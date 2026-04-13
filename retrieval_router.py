"""
Retrieval Router — multi-source agentic orchestration.

The model suggests source-specific queries, Python executes each enabled
source deterministically, then a dedupe/rerank/select pipeline builds a
compact evidence pack for mandatory synthesis.
"""

import re
from difflib import SequenceMatcher

import ollama

from sources.clinical_trials import search_clinical_trials
from sources.europe_pmc import search_europe_pmc
from sources.pubmed import search_pubmed


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_NAME = "gemma4:e2b"
ROUTING_OPTIONS = {"temperature": 0.1}
SYNTHESIS_OPTIONS = {"temperature": 0.2}
SOURCE_DISPLAY_ORDER = [
    "Europe PMC",
    "PubMed",
    "ClinicalTrials.gov",
]
DEFAULT_LIMIT = 5
MAX_SELECTED_EVIDENCE = 8
MAX_SYNTHESIS_ATTEMPTS = 3
MAX_RESERVED_TRIALS = 3
STOPWORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "into", "is", "it", "its", "of", "on", "or", "the", "their",
    "there", "these", "this", "to", "what", "which", "with",
}
ROUTER_SYSTEM_PROMPT = (
    "You are a clinical research routing agent. Your job is to decide the best "
    "search query for each available tool.\n\n"
    "Available tools:\n"
    "  • search_europe_pmc — published biomedical literature from Europe PMC\n"
    "  • search_pubmed — published biomedical literature from PubMed/MEDLINE\n"
    "  • search_clinical_trials — clinical trial records from ClinicalTrials.gov\n\n"
    "IMPORTANT RULES:\n"
    "  1. Call each available tool at most once.\n"
    "  2. Use the user's request to craft precise source-specific search queries.\n"
    "  3. If one generic query works for all tools, reuse it.\n"
    "  4. Do not answer the user directly in this step."
)

FINAL_SYNTHESIS_PROMPT = (
    "You are a clinical research assistant. Use only the curated evidence pack "
    "provided in the user message.\n\n"
    "OUTPUT CONTRACT:\n"
    "1. Start with a direct answer to the user's question.\n"
    "2. Synthesize across evidence; do not summarize one citation at a time.\n"
    "3. Use short sections when helpful, especially for published evidence and "
    "clinical-trial evidence.\n"
    "4. Every factual paragraph or bullet must include inline citations like [1], [3].\n"
    "5. If clinical-trial records are included and relevant, discuss them explicitly.\n"
    "6. If evidence is limited or conflicting, say so clearly.\n"
    "7. Do not apologize, refuse, or say that evidence is missing unless the pack "
    "itself lacks the requested evidence.\n"
    "8. Do not output a References section.\n"
)

# ──────────────────────────────────────────────
# Tool Schemas
# ──────────────────────────────────────────────
TOOL_SEARCH_EUROPE_PMC = {
    "type": "function",
    "function": {
        "name": "search_europe_pmc",
        "description": (
            "Search Europe PMC for published biomedical and life-sciences "
            "literature. Returns article titles, abstracts, authors, and DOIs. "
            "Use for drug-target interactions, molecular mechanisms, and "
            "published research papers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., drug names, gene targets, disease mechanisms).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 5.",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SEARCH_PUBMED = {
    "type": "function",
    "function": {
        "name": "search_pubmed",
        "description": (
            "Search PubMed/MEDLINE for published biomedical literature. "
            "Returns article titles, abstracts, authors, and DOIs. "
            "Use for peer-reviewed medical research, pharmacology, "
            "and clinical studies."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., drug names, gene targets, disease mechanisms).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 5.",
                },
            },
            "required": ["query"],
        },
    },
}

TOOL_SEARCH_CLINICAL_TRIALS = {
    "type": "function",
    "function": {
        "name": "search_clinical_trials",
        "description": (
            "Search ClinicalTrials.gov for clinical trial records. "
            "Returns trial titles, sponsors, summaries, and links. "
            "Use for questions about ongoing/completed clinical trials, "
            "drug efficacy studies, phases, and recruitment status."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query (e.g., drug names, conditions, interventions).",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return. Default is 5.",
                },
            },
            "required": ["query"],
        },
    },
}

# Map source display names to their tool schemas
SOURCE_TOOL_MAP = {
    "Europe PMC": TOOL_SEARCH_EUROPE_PMC,
    "PubMed": TOOL_SEARCH_PUBMED,
    "ClinicalTrials.gov": TOOL_SEARCH_CLINICAL_TRIALS,
}

# Map function names to their implementations
FUNCTION_DISPATCH = {
    "search_europe_pmc": search_europe_pmc,
    "search_pubmed": search_pubmed,
    "search_clinical_trials": search_clinical_trials,
}

# Map source display names to their function names
SOURCE_FUNCTION_MAP = {
    "Europe PMC": "search_europe_pmc",
    "PubMed": "search_pubmed",
    "ClinicalTrials.gov": "search_clinical_trials",
}


# ──────────────────────────────────────────────
# LLM Helpers
# ──────────────────────────────────────────────
def _chat(messages: list[dict], tools: list[dict] | None = None, options: dict | None = None):
    """Thin wrapper for Ollama chat with shared model/options handling."""
    kwargs = {
        "model": MODEL_NAME,
        "messages": messages,
    }
    if tools is not None:
        kwargs["tools"] = tools
    if options is not None:
        kwargs["options"] = options
    return ollama.chat(**kwargs)


# ──────────────────────────────────────────────
# Query and Evidence Helpers
# ──────────────────────────────────────────────
def _normalize_enabled_sources(enabled_sources) -> list[str]:
    """Normalize enabled sources into a stable display order."""
    enabled_set = set(enabled_sources)
    ordered = [source for source in SOURCE_DISPLAY_ORDER if source in enabled_set]
    extras = sorted(enabled_set - set(SOURCE_DISPLAY_ORDER))
    return ordered + extras


def _source_type(source_name: str) -> str:
    """Classify a source as literature vs trial evidence."""
    return "trial" if source_name == "ClinicalTrials.gov" else "literature"


def _normalize_text(text: str) -> str:
    """Normalize text for rough matching."""
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _normalize_title(title: str) -> str:
    """Normalize titles for deduplication."""
    return _normalize_text(title)


def _canonical_doi(doi: str | None) -> str:
    """Normalize DOI values so duplicate papers collapse cleanly."""
    normalized = (doi or "").strip().lower()
    normalized = normalized.removeprefix("https://doi.org/")
    normalized = normalized.removeprefix("http://doi.org/")
    normalized = normalized.removeprefix("doi:")
    return normalized


def _tokenize(text: str) -> set[str]:
    """Tokenize text into a compact relevance set."""
    return {
        token for token in _normalize_text(text).split()
        if len(token) >= 3 and token not in STOPWORDS
    }


def _query_wants_trials(user_query: str) -> bool:
    """Detect explicit trial-focused queries."""
    lowered = user_query.lower()
    trial_terms = [
        "clinical trial", "clinical trials", "ongoing", "recruiting",
        "enrolling", "phase ", "nct", "study", "studies",
    ]
    return any(term in lowered for term in trial_terms)


def _query_wants_clinical(user_query: str) -> bool:
    """Detect clinically oriented queries that may benefit from trial evidence."""
    lowered = user_query.lower()
    clinical_terms = [
        "clinical", "efficacy", "safety", "outcome", "survival", "response",
        "patients", "human", "treatment", "therapy",
    ]
    return any(term in lowered for term in clinical_terms)


def _split_sentences(text: str) -> list[str]:
    """Split text into rough sentences for snippet selection."""
    parts = re.split(r"(?<=[.!?])\s+", (text or "").strip())
    return [part.strip() for part in parts if part.strip()]


def _select_key_snippet(text: str, query_tokens: set[str], max_sentences: int = 2, max_chars: int = 360) -> str:
    """Choose the most query-relevant sentence window from a longer abstract."""
    sentences = _split_sentences(text)
    if not sentences:
        return "No abstract or summary available."

    scored = []
    for idx, sentence in enumerate(sentences):
        overlap = len(query_tokens & _tokenize(sentence))
        scored.append((overlap, -idx, sentence))
    scored.sort(reverse=True)

    chosen = [entry[2] for entry in scored[:max_sentences]]
    snippet = " ".join(chosen).strip()
    if len(snippet) > max_chars:
        snippet = snippet[: max_chars - 3].rstrip() + "..."
    return snippet


def _flatten_results(results_by_source: dict[str, list[dict]]) -> list[dict]:
    """Flatten per-source results into a single evidence pool."""
    pool = []
    pool_id = 1

    for source_name in _normalize_enabled_sources(results_by_source.keys()):
        for source_rank, paper in enumerate(results_by_source.get(source_name, []), start=1):
            pool.append({
                "pool_id": pool_id,
                "title": paper.get("title", "No Title"),
                "authors": paper.get("authors", []),
                "year": paper.get("year"),
                "abstract": paper.get("abstract", ""),
                "doi": paper.get("doi"),
                "url": paper.get("url", ""),
                "source_names": [source_name],
                "source_type": _source_type(source_name),
                "source_rank": source_rank,
            })
            pool_id += 1

    return pool


def _merge_evidence(existing: dict, incoming: dict) -> None:
    """Merge duplicate records while preserving the richest metadata."""
    for source_name in incoming["source_names"]:
        if source_name not in existing["source_names"]:
            existing["source_names"].append(source_name)
    existing["source_names"] = _normalize_enabled_sources(existing["source_names"])

    if not existing.get("doi") and incoming.get("doi"):
        existing["doi"] = incoming["doi"]
    if not existing.get("url") and incoming.get("url"):
        existing["url"] = incoming["url"]
    if not existing.get("year") and incoming.get("year"):
        existing["year"] = incoming["year"]
    if len(incoming.get("authors", [])) > len(existing.get("authors", [])):
        existing["authors"] = incoming.get("authors", [])
    if len(incoming.get("abstract", "")) > len(existing.get("abstract", "")):
        existing["abstract"] = incoming.get("abstract", "")

    existing["source_rank"] = min(existing.get("source_rank", 99), incoming.get("source_rank", 99))
    existing["source_type"] = "trial" if "ClinicalTrials.gov" in existing["source_names"] else "literature"


def _same_record(existing: dict, incoming: dict) -> bool:
    """Detect duplicate records across sources."""
    doi_a = _canonical_doi(existing.get("doi"))
    doi_b = _canonical_doi(incoming.get("doi"))
    if doi_a and doi_b and doi_a == doi_b:
        return True

    title_a = _normalize_title(existing.get("title", ""))
    title_b = _normalize_title(incoming.get("title", ""))
    if title_a and title_b and title_a == title_b:
        return True

    year_a = existing.get("year")
    year_b = incoming.get("year")
    same_year = year_a == year_b or year_a is None or year_b is None
    if same_year and title_a and title_b:
        if SequenceMatcher(None, title_a, title_b).ratio() >= 0.96:
            return True

    return False


def _deduplicate_evidence(pool: list[dict]) -> list[dict]:
    """Collapse duplicate papers that appear across Europe PMC and PubMed."""
    deduped: list[dict] = []

    for item in pool:
        match = None
        for candidate in deduped:
            if _same_record(candidate, item):
                match = candidate
                break
        if match is None:
            deduped.append(item)
        else:
            _merge_evidence(match, item)

    return deduped


def _relevance_score(item: dict, user_query: str) -> float:
    """Score an evidence item against the user query."""
    query_tokens = _tokenize(user_query)
    title_tokens = _tokenize(item.get("title", ""))
    abstract_tokens = _tokenize(item.get("abstract", ""))
    lowered_query = user_query.lower()
    lowered_title = item.get("title", "").lower()
    lowered_abstract = item.get("abstract", "").lower()

    title_overlap = len(query_tokens & title_tokens)
    abstract_overlap = len(query_tokens & abstract_tokens)

    phrase_bonus = 0.0
    for phrase in re.findall(r"[a-z0-9+\-/]{3,}(?:\s+[a-z0-9+\-/]{3,})+", lowered_query):
        if phrase in lowered_title:
            phrase_bonus += 2.5
        elif phrase in lowered_abstract:
            phrase_bonus += 1.0

    score = (title_overlap * 4.0) + (abstract_overlap * 1.5) + phrase_bonus

    year = item.get("year")
    if isinstance(year, int):
        score += max(min((year - 2018) * 0.2, 1.6), 0.0)

    score += max(0, DEFAULT_LIMIT + 1 - item.get("source_rank", DEFAULT_LIMIT + 1)) * 0.3

    if item.get("doi"):
        score += 0.2
    if len(item.get("source_names", [])) > 1:
        score += 0.5

    wants_trials = _query_wants_trials(user_query)
    wants_clinical = _query_wants_clinical(user_query)
    if item["source_type"] == "trial":
        if wants_trials:
            score += 4.0
        elif wants_clinical:
            score += 1.5
        else:
            score -= 1.0
    else:
        if wants_trials:
            score -= 0.5
        else:
            score += 1.0

    return score


def _selection_sort_key(item: dict) -> tuple:
    """Stable sort key for selected evidence."""
    return (
        item.get("score", 0.0),
        item.get("year") or 0,
        -item.get("source_rank", DEFAULT_LIMIT + 1),
    )


def _greedy_fill(pool: list[dict], slots: int, selected: list[dict]) -> list[dict]:
    """Greedy selection with a small diversity bonus."""
    chosen = list(selected)
    chosen_ids = {item["pool_id"] for item in chosen}

    while len(chosen) < slots:
        remaining = [item for item in pool if item["pool_id"] not in chosen_ids]
        if not remaining:
            break

        used_sources = {
            source_name
            for item in chosen
            for source_name in item.get("source_names", [])
        }
        best = max(
            remaining,
            key=lambda item: (
                item.get("score", 0.0)
                + (0.35 if any(src not in used_sources for src in item.get("source_names", [])) else 0.0),
                item.get("year") or 0,
            ),
        )
        chosen.append(best)
        chosen_ids.add(best["pool_id"])

    return chosen


def _select_evidence(pool: list[dict], user_query: str) -> list[dict]:
    """Rerank globally and keep a compact, source-aware evidence pack."""
    if not pool:
        return []

    for item in pool:
        item["score"] = _relevance_score(item, user_query)

    sorted_pool = sorted(pool, key=_selection_sort_key, reverse=True)
    trials = [item for item in sorted_pool if item["source_type"] == "trial"]

    reserved_trials = 0
    if _query_wants_trials(user_query):
        reserved_trials = min(MAX_RESERVED_TRIALS, len(trials))
    elif _query_wants_clinical(user_query) and trials:
        reserved_trials = 1

    selected = trials[:reserved_trials]
    selected = _greedy_fill(sorted_pool, MAX_SELECTED_EVIDENCE, selected)
    selected = sorted(selected, key=_selection_sort_key, reverse=True)[:MAX_SELECTED_EVIDENCE]

    for citation_id, item in enumerate(selected, start=1):
        item["citation_id"] = citation_id

    return selected


def _build_evidence_packet(selected_evidence: list[dict], user_query: str) -> tuple[str, dict[int, dict]]:
    """
    Stage 1: build compact, extractive evidence briefs from selected items.

    These briefs become the synthesis input instead of raw full abstracts.
    """
    query_tokens = _tokenize(user_query)
    literature_lines = ["=== Published Literature ==="]
    trial_lines = ["=== Clinical Trial Evidence ==="]
    reference_map: dict[int, dict] = {}

    for item in selected_evidence:
        citation_id = item["citation_id"]
        source_label = " / ".join(item.get("source_names", []))
        year_str = str(item.get("year")) if item.get("year") else "N/A"
        authors = item.get("authors", [])
        authors_str = ", ".join(authors[:4]) if authors else "Unknown"
        if len(authors) > 4:
            authors_str += " et al."
        snippet = _select_key_snippet(item.get("abstract", ""), query_tokens)

        brief = [
            f"[{citation_id}] Source: {source_label} | Year: {year_str}",
            f"Title: {item.get('title', 'No Title')}",
            f"Authors/Sponsor: {authors_str}",
            f"Key evidence: {snippet}",
        ]
        if item["source_type"] == "trial":
            trial_lines.append("\n".join(brief))
        else:
            literature_lines.append("\n".join(brief))

        reference_map[citation_id] = {
            "source": source_label,
            "source_type": item["source_type"],
            "title": item.get("title", "No Title"),
            "authors": authors,
            "year": item.get("year"),
            "doi": item.get("doi"),
            "url": item.get("url", ""),
        }

    sections = []
    if len(literature_lines) > 1:
        sections.append("\n\n".join(literature_lines))
    if len(trial_lines) > 1:
        sections.append("\n\n".join(trial_lines))

    packet = (
        "Selected evidence pack. These items are deduplicated, reranked, and "
        "condensed from the retrieved sources. Use them as the sole basis for synthesis.\n\n"
        + "\n\n".join(sections)
    )
    return packet, reference_map


# ──────────────────────────────────────────────
# Output Validation and Formatting
# ──────────────────────────────────────────────
def _build_reference_link(paper: dict) -> str:
    """Choose the best clickable link for a result."""
    doi = _canonical_doi(paper.get("doi"))
    if doi:
        return f"https://doi.org/{doi}"
    return (paper.get("url") or "").strip()


def _extract_citation_numbers(text: str) -> list[int]:
    """Extract unique citation numbers in order of appearance."""
    seen: set[int] = set()
    citations: list[int] = []
    for match in re.finditer(r"\[(\d+)\]", text or ""):
        num = int(match.group(1))
        if num not in seen:
            seen.add(num)
            citations.append(num)
    return citations


def _format_reference_entry(num: int, paper: dict) -> str:
    """Format a single reference entry as a Markdown list item."""
    authors = paper.get("authors") or []
    authors_str = ", ".join(authors[:6]) if authors else "Unknown"
    if len(authors) > 6:
        authors_str += " et al."

    year = paper.get("year") or "N/A"
    title = paper.get("title", "No Title")
    source = paper.get("source", "")
    link = _build_reference_link(paper)
    title_link = f"[{title}]({link})" if link else title
    return f"{num}. {title_link}. {authors_str}. {year}. {source}."


def _build_references_section(citation_numbers: list[int], reference_map: dict[int, dict]) -> str:
    """
    Render a Markdown references section for ALL evidence in reference_map.

    Cited items are listed first (in order of appearance in the text),
    followed by any uncited items (in ascending ID order).
    """
    if not reference_map:
        return ""

    lines = ["## References", ""]

    # Cited references first (in appearance order)
    emitted: set[int] = set()
    for num in citation_numbers:
        paper = reference_map.get(num)
        if paper is None:
            continue
        lines.append(_format_reference_entry(num, paper))
        emitted.add(num)

    # Then uncited references (ascending ID order)
    for num in sorted(reference_map.keys()):
        if num in emitted:
            continue
        lines.append(_format_reference_entry(num, reference_map[num]))

    return "\n".join(lines)


def _strip_existing_references(text: str) -> str:
    """Remove any model-generated references section so code can append a clean one."""
    if not text:
        return text
    return re.split(r"\n##\s+References\b", text, maxsplit=1, flags=re.IGNORECASE)[0].rstrip()


def _renumber_citations(text: str, reference_map: dict[int, dict]) -> tuple[str, dict[int, dict]]:
    """
    Remap arbitrary citation numbers in `text` to a contiguous 1-based
    sequence and rebuild reference_map accordingly.

    Also renumbers any uncited entries in reference_map so every item
    gets a clean sequential ID.
    """
    cited = _extract_citation_numbers(text)
    # Build old->new mapping: cited items first (in appearance order),
    # then uncited items (in ascending original ID order)
    old_to_new: dict[int, int] = {}
    next_id = 1
    for old_id in cited:
        if old_id in reference_map:
            old_to_new[old_id] = next_id
            next_id += 1
    for old_id in sorted(reference_map.keys()):
        if old_id not in old_to_new:
            old_to_new[old_id] = next_id
            next_id += 1

    # Rewrite inline citations in the text
    def _replace_cite(match: re.Match) -> str:
        old_num = int(match.group(1))
        new_num = old_to_new.get(old_num, old_num)
        return f"[{new_num}]"

    new_text = re.sub(r"\[(\d+)\]", _replace_cite, text)

    # Rebuild reference_map with new IDs
    new_map: dict[int, dict] = {}
    for old_id, new_id in old_to_new.items():
        if old_id in reference_map:
            new_map[new_id] = reference_map[old_id]

    return new_text, new_map


def _format_retrieval_error(error_msg: str) -> str:
    """Convert raw exception strings into concise, user-friendly messages."""
    # Strip HTTP URLs from error messages
    clean = re.sub(r"https?://\S+", "", error_msg).strip()
    # Remove trailing colons or whitespace left by URL stripping
    clean = re.sub(r":\s*$", "", clean).strip()
    # Collapse multiple spaces
    clean = re.sub(r"\s{2,}", " ", clean).strip()
    if not clean:
        return error_msg.split(":", 1)[0] + ": request failed"
    return clean


def _is_invalid_answer(text: str, reference_map: dict[int, dict], user_query: str) -> bool:
    """Detect refusal-like or structurally invalid synthesis outputs."""
    if not text or len(text.strip()) < 160:
        return True

    lowered = text.lower()
    invalid_patterns = [
        "i apologize",
        "i'm sorry",
        "i cannot fulfill",
        "cannot fulfill the request",
        "cannot comply",
        "no specific evidence",
        "no retrieved results were provided",
        "please provide the evidence",
        "evidence you are referring to",
        "retrieved evidence was found across the enabled sources",
        "title:",
        "key evidence:",
    ]
    if any(pattern in lowered for pattern in invalid_patterns):
        return True

    citation_numbers = _extract_citation_numbers(text)
    if not citation_numbers:
        return True
    if any(num not in reference_map for num in citation_numbers):
        return True
    if len(reference_map) >= 3 and len(citation_numbers) < 2:
        return True

    if _query_wants_trials(user_query):
        has_trials = any(paper["source_type"] == "trial" for paper in reference_map.values())
        cites_trials = any(reference_map[num]["source_type"] == "trial" for num in citation_numbers)
        if has_trials and not cites_trials:
            return True

    return False


# ──────────────────────────────────────────────
# Synthesis
# ──────────────────────────────────────────────
def _synthesis_messages(
    user_query: str,
    evidence_packet: str,
    reference_map: dict[int, dict],
    feedback: str = "",
) -> list[dict]:
    """Build a self-contained synthesis request."""
    feedback_block = f"{feedback}\n\n" if feedback else ""
    has_trials = any(paper["source_type"] == "trial" for paper in reference_map.values())
    trial_instruction = (
        "Clinical-trial evidence is present in the evidence pack. Discuss it explicitly and "
        "cite it if it is relevant to the question.\n"
        if has_trials
        else "No clinical-trial evidence was selected into the evidence pack.\n"
    )

    return [
        {"role": "system", "content": FINAL_SYNTHESIS_PROMPT},
        {
            "role": "user",
            "content": (
                f"{feedback_block}"
                f"User question:\n{user_query}\n\n"
                f"{trial_instruction}"
                "Instructions:\n"
                "- Give a direct answer first.\n"
                "- Integrate evidence across the cited items instead of listing titles.\n"
                "- Every factual paragraph or bullet must end with inline citations.\n"
                "- Use only citation numbers that exist in the evidence pack.\n"
                "- Do not include a References section.\n\n"
                "Curated evidence pack:\n\n"
                f"{evidence_packet}"
            ),
        },
    ]


def _run_synthesis(user_query: str, evidence_packet: str, reference_map: dict[int, dict]) -> str:
    """Run mandatory synthesis with retries. Returns empty string on failure."""
    feedbacks = [
        "",
        (
            "The previous draft was invalid. Rewrite it as a direct evidence-based synthesis "
            "with grounded inline citations and explicit cross-source reasoning."
        ),
        (
            "Final attempt. Write a concise clinical synthesis that answers the question directly, "
            "uses multiple grounded citations, and explicitly separates trial evidence when present."
        ),
    ]

    for attempt in range(MAX_SYNTHESIS_ATTEMPTS):
        messages = _synthesis_messages(
            user_query=user_query,
            evidence_packet=evidence_packet,
            reference_map=reference_map,
            feedback=feedbacks[attempt],
        )
        response = _chat(messages=messages, options=SYNTHESIS_OPTIONS)
        final_text = _strip_existing_references(response["message"].get("content", ""))
        if not _is_invalid_answer(final_text, reference_map, user_query):
            return final_text

    return ""


def _call_source(fn_name: str, query: str, limit: int,
                 pubmed_tool_name: str, pubmed_email: str) -> list[dict]:
    """Call a source function by its function name."""
    fn = FUNCTION_DISPATCH.get(fn_name)
    if fn is None:
        return []
    if fn_name == "search_pubmed":
        return fn(query=query, limit=limit, tool_name=pubmed_tool_name, email=pubmed_email)
    return fn(query=query, limit=limit)


def _plan_tool_calls(user_query: str, tools: list[dict]) -> dict[str, dict]:
    """Ask the model for source-specific search queries."""
    response = _chat(
        messages=[
            {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
            {"role": "user", "content": user_query},
        ],
        tools=tools,
        options=ROUTING_OPTIONS,
    )

    planned_calls: dict[str, dict] = {}
    for tool_call in response.get("message", {}).get("tool_calls", []):
        fn_name = tool_call["function"]["name"]
        args = tool_call["function"].get("arguments", {}) or {}
        planned_calls[fn_name] = {
            "query": args.get("query", user_query),
            "limit": args.get("limit", DEFAULT_LIMIT),
        }

    return planned_calls


# ──────────────────────────────────────────────
# Agent Loop
# ──────────────────────────────────────────────
def route_and_retrieve(
    user_query: str,
    enabled_sources,
    pubmed_tool_name: str = "DrugTargetAgent",
    pubmed_email: str = "user@example.com",
) -> tuple[str, dict[str, int], list[str]]:
    """
    Agentic retrieval with mandatory synthesis:
      1. Let the model propose source-specific search queries
      2. Execute every enabled source in Python
      3. Deduplicate and rerank the evidence globally
      4. Build a compact evidence pack
      5. Require a valid synthesized answer, otherwise return an explicit error
    """
    ordered_sources = _normalize_enabled_sources(enabled_sources)

    tools = []
    for source_name in ordered_sources:
        tool_schema = SOURCE_TOOL_MAP.get(source_name)
        if tool_schema:
            tools.append(tool_schema)

    if not tools:
        return ("⚠️ No data sources are enabled. Please enable at least one source in the sidebar.", {}, [])

    results_by_source: dict[str, list[dict]] = {}
    source_counts: dict[str, int] = {}
    tool_queries: list[str] = []

    retrieval_errors: list[str] = []
    planned_calls = _plan_tool_calls(user_query, tools)

    for source_name in ordered_sources:
        fn_name = SOURCE_FUNCTION_MAP.get(source_name)
        if fn_name is None:
            continue

        planned = planned_calls.get(fn_name, {})
        query = planned.get("query", user_query)
        limit = planned.get("limit", DEFAULT_LIMIT)
        tool_queries.append(query)

        try:
            results = _call_source(fn_name, query, limit, pubmed_tool_name, pubmed_email)
        except Exception as exc:
            results = []
            retrieval_errors.append(f"{source_name}: {exc}")

        results_by_source[source_name] = results
        source_counts[source_name] = len(results)

    raw_pool = _flatten_results(results_by_source)
    deduped_pool = _deduplicate_evidence(raw_pool)
    selected_evidence = _select_evidence(deduped_pool, user_query)

    if not selected_evidence:
        final_text = "No results were retrieved from the enabled sources."
    else:
        evidence_packet, reference_map = _build_evidence_packet(selected_evidence, user_query)
        final_text = _run_synthesis(user_query, evidence_packet, reference_map)

        if not final_text:
            final_text = (
                "⚠️ Retrieval succeeded, but the model failed to produce a valid "
                "evidence-grounded synthesis after multiple attempts. "
                "Please narrow the query or reduce the enabled sources and try again."
            )
        else:
            # Renumber citations to a contiguous 1-based sequence
            final_text, reference_map = _renumber_citations(final_text, reference_map)
            citation_numbers = _extract_citation_numbers(final_text)
            references_section = _build_references_section(citation_numbers, reference_map)
            if references_section:
                final_text = f"{final_text.rstrip()}\n\n{references_section}"

    if retrieval_errors:
        clean_errors = [_format_retrieval_error(err) for err in retrieval_errors]
        final_text = (
            f"{final_text}\n\n"
            "## Retrieval Notes\n"
            + "\n".join(f"- {err}" for err in clean_errors)
        )

    return final_text, source_counts, tool_queries
