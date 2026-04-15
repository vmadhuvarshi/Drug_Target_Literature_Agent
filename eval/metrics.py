"""Metric computation for benchmarked drug-target interaction answers."""

from __future__ import annotations

import json
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Any

import ollama


JUDGE_OPTIONS = {"temperature": 0}
MAX_ABSTRACT_CHARS = 1800
MAX_CONTEXT_CHARS = 12000


def call_ollama_with_retries(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None = None,
    options: dict[str, Any] | None = None,
    timeout: int = 120,
    retries: int = 1,
) -> dict[str, Any]:
    """Call Ollama with a per-call timeout and bounded retries."""

    last_error: Exception | None = None
    merged_options = dict(options or {})
    merged_options["temperature"] = 0

    for attempt in range(retries + 1):
        try:
            return _call_ollama_once(
                model=model,
                messages=messages,
                tools=tools,
                options=merged_options,
                timeout=timeout,
            )
        except Exception as exc:  # pragma: no cover - depends on local Ollama
            last_error = exc
            if attempt < retries:
                time.sleep(min(2**attempt, 5))

    raise RuntimeError(f"Ollama call failed after {retries + 1} attempts: {last_error}")


def _call_ollama_once(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    options: dict[str, Any],
    timeout: int,
) -> dict[str, Any]:
    """Run one Ollama call in a worker so benchmark execution can time out."""

    def _invoke() -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "options": options,
        }
        if tools is not None:
            kwargs["tools"] = tools

        try:
            client = ollama.Client(timeout=timeout)
            return client.chat(**kwargs)
        except TypeError:
            return ollama.chat(**kwargs)

    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_invoke)
    try:
        return future.result(timeout=timeout)
    except FutureTimeout as exc:
        future.cancel()
        raise TimeoutError(f"Ollama call timed out after {timeout}s") from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def retrieval_precision(
    question: str,
    retrieved_papers: list[dict[str, Any]],
    *,
    model: str,
    timeout: int = 120,
    retries: int = 1,
) -> dict[str, Any]:
    """
    Fraction of retrieved papers judged relevant to the query by Gemma.

    The harness judges the deduplicated retrieval pool. This avoids counting
    the same paper twice when it is returned by both Europe PMC and PubMed.
    """

    if not retrieved_papers:
        return {
            "score": 0.0,
            "relevant_count": 0,
            "total_count": 0,
            "judgments": [],
            "error": None,
        }

    papers = [
        {
            "index": idx,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "doi": paper.get("doi"),
            "source": paper.get("source") or " / ".join(paper.get("source_names", [])),
            "abstract": _truncate(paper.get("abstract", ""), MAX_ABSTRACT_CHARS),
        }
        for idx, paper in enumerate(retrieved_papers, start=1)
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict biomedical literature relevance judge. "
                "Return JSON only. Mark a paper relevant only if its title or "
                "abstract directly helps answer the user's drug-target question."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                "Retrieved papers:\n"
                f"{json.dumps(papers, indent=2)}\n\n"
                "Return exactly this schema:\n"
                '{"judgments":[{"index":1,"relevant":true,"rationale":"short reason"}]}'
            ),
        },
    ]

    try:
        response = call_ollama_with_retries(
            model=model,
            messages=messages,
            options=JUDGE_OPTIONS,
            timeout=timeout,
            retries=retries,
        )
        parsed = _extract_json(response.get("message", {}).get("content", ""))
        judgments = parsed.get("judgments", [])
        normalized = _normalize_boolean_judgments(judgments, len(retrieved_papers))
        relevant_count = sum(1 for item in normalized if item["relevant"])
        return {
            "score": _safe_divide(relevant_count, len(retrieved_papers)),
            "relevant_count": relevant_count,
            "total_count": len(retrieved_papers),
            "judgments": normalized,
            "error": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "relevant_count": None,
            "total_count": len(retrieved_papers),
            "judgments": [],
            "error": str(exc),
        }


def retrieval_recall(
    expected_sources: list[str],
    retrieved_papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fraction of expected landmark DOI/PMID/NCT identifiers retrieved."""

    expected = [_normalize_identifier(source) for source in expected_sources]
    expected = [source for source in expected if source]
    retrieved_ids = _collect_identifiers(retrieved_papers)

    matched: list[str] = []
    missing: list[str] = []
    for source in expected:
        if source in retrieved_ids:
            matched.append(source)
        else:
            missing.append(source)

    return {
        "score": _safe_divide(len(matched), len(expected)) if expected else 0.0,
        "matched_count": len(matched),
        "expected_count": len(expected),
        "matched_sources": matched,
        "missing_sources": missing,
        "error": None,
    }


def citation_accuracy(
    synthesis_text: str,
    reference_map: dict[int, dict[str, Any]],
    retrieved_papers: list[dict[str, Any]],
) -> dict[str, Any]:
    """Fraction of inline citation IDs that resolve to retrieved records."""

    citation_ids = _extract_inline_citation_ids(synthesis_text)
    unique_ids = sorted(set(citation_ids))
    if not unique_ids:
        return {
            "score": 0.0,
            "valid_count": 0,
            "citation_count": 0,
            "invalid_citations": [],
            "error": None,
        }

    retrieved_ids = _collect_identifiers(retrieved_papers)
    retrieved_titles = {_normalize_title(paper.get("title", "")) for paper in retrieved_papers}
    invalid: list[int] = []

    for citation_id in unique_ids:
        ref = reference_map.get(citation_id)
        if ref is None:
            invalid.append(citation_id)
            continue

        ref_ids = _record_identifiers(ref)
        ref_title = _normalize_title(ref.get("title", ""))
        points_to_retrieved = bool(ref_ids & retrieved_ids) or (
            bool(ref_title) and ref_title in retrieved_titles
        )
        has_real_locator = bool(ref_ids or ref.get("url") or ref.get("doi"))

        if not (points_to_retrieved and has_real_locator):
            invalid.append(citation_id)

    valid_count = len(unique_ids) - len(invalid)
    return {
        "score": _safe_divide(valid_count, len(unique_ids)),
        "valid_count": valid_count,
        "citation_count": len(unique_ids),
        "citation_occurrences": len(citation_ids),
        "invalid_citations": invalid,
        "error": None,
    }


def factual_coverage(
    synthesis_text: str,
    expected_key_findings: list[str],
    *,
    model: str,
    timeout: int = 120,
    retries: int = 1,
) -> dict[str, Any]:
    """Fraction of expected key findings addressed in the synthesis."""

    if not expected_key_findings:
        return {
            "score": 0.0,
            "covered_count": 0,
            "expected_count": 0,
            "judgments": [],
            "error": None,
        }

    findings = [
        {"index": idx, "finding": finding}
        for idx, finding in enumerate(expected_key_findings, start=1)
    ]
    clean_synthesis = _strip_generated_sections(synthesis_text)

    messages = [
        {
            "role": "system",
            "content": (
                "You judge factual coverage in biomedical answers. Return JSON only. "
                "Mark a finding addressed if the answer states it directly or clearly "
                "covers the same factual point, even with different wording."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Synthesis:\n{clean_synthesis}\n\n"
                f"Expected findings:\n{json.dumps(findings, indent=2)}\n\n"
                "Return exactly this schema:\n"
                '{"coverage":[{"index":1,"addressed":true,"rationale":"short reason"}]}'
            ),
        },
    ]

    try:
        response = call_ollama_with_retries(
            model=model,
            messages=messages,
            options=JUDGE_OPTIONS,
            timeout=timeout,
            retries=retries,
        )
        parsed = _extract_json(response.get("message", {}).get("content", ""))
        judgments = parsed.get("coverage", [])
        normalized = _normalize_boolean_judgments(
            judgments,
            len(expected_key_findings),
            truth_key="addressed",
        )
        covered_count = sum(1 for item in normalized if item["addressed"])
        return {
            "score": _safe_divide(covered_count, len(expected_key_findings)),
            "covered_count": covered_count,
            "expected_count": len(expected_key_findings),
            "judgments": normalized,
            "error": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "covered_count": None,
            "expected_count": len(expected_key_findings),
            "judgments": [],
            "error": str(exc),
        }


def hallucination_rate(
    synthesis_text: str,
    retrieved_papers: list[dict[str, Any]],
    *,
    model: str,
    timeout: int = 120,
    retries: int = 1,
) -> dict[str, Any]:
    """Fraction of generated claims not traceable to retrieved abstracts."""

    claims = _extract_candidate_claims(synthesis_text)
    if not claims:
        return {
            "score": 0.0,
            "unsupported_count": 0,
            "claim_count": 0,
            "judgments": [],
            "error": None,
        }

    if not retrieved_papers:
        return {
            "score": 1.0,
            "unsupported_count": len(claims),
            "claim_count": len(claims),
            "judgments": [
                {"index": idx, "supported": False, "rationale": "No retrieved evidence."}
                for idx, _claim in enumerate(claims, start=1)
            ],
            "error": None,
        }

    evidence_context = _build_evidence_context(retrieved_papers)
    indexed_claims = [
        {"index": idx, "claim": claim}
        for idx, claim in enumerate(claims, start=1)
    ]

    messages = [
        {
            "role": "system",
            "content": (
                "You are a strict biomedical claim support judge. Return JSON only. "
                "A claim is supported only when it can be traced to the supplied "
                "retrieved title or abstract context. Do not use outside knowledge."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Retrieved evidence:\n{evidence_context}\n\n"
                f"Claims:\n{json.dumps(indexed_claims, indent=2)}\n\n"
                "Return exactly this schema:\n"
                '{"claims":[{"index":1,"supported":true,"rationale":"short reason"}]}'
            ),
        },
    ]

    try:
        response = call_ollama_with_retries(
            model=model,
            messages=messages,
            options=JUDGE_OPTIONS,
            timeout=timeout,
            retries=retries,
        )
        parsed = _extract_json(response.get("message", {}).get("content", ""))
        judgments = parsed.get("claims", [])
        normalized = _normalize_boolean_judgments(
            judgments,
            len(claims),
            truth_key="supported",
        )
        unsupported_count = sum(1 for item in normalized if not item["supported"])
        return {
            "score": _safe_divide(unsupported_count, len(claims)),
            "unsupported_count": unsupported_count,
            "claim_count": len(claims),
            "judgments": normalized,
            "error": None,
        }
    except Exception as exc:
        return {
            "score": None,
            "unsupported_count": None,
            "claim_count": len(claims),
            "judgments": [],
            "error": str(exc),
        }


def compute_all_metrics(
    *,
    question: str,
    synthesis_text: str,
    retrieved_papers: list[dict[str, Any]],
    reference_map: dict[int, dict[str, Any]],
    expected_sources: list[str],
    expected_key_findings: list[str],
    model: str,
    timeout: int = 120,
    retries: int = 1,
) -> dict[str, dict[str, Any]]:
    """Compute every benchmark metric for one question."""

    return {
        "retrieval_precision": retrieval_precision(
            question,
            retrieved_papers,
            model=model,
            timeout=timeout,
            retries=retries,
        ),
        "retrieval_recall": retrieval_recall(expected_sources, retrieved_papers),
        "citation_accuracy": citation_accuracy(
            synthesis_text,
            reference_map,
            retrieved_papers,
        ),
        "factual_coverage": factual_coverage(
            synthesis_text,
            expected_key_findings,
            model=model,
            timeout=timeout,
            retries=retries,
        ),
        "hallucination_rate": hallucination_rate(
            synthesis_text,
            retrieved_papers,
            model=model,
            timeout=timeout,
            retries=retries,
        ),
    }


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.IGNORECASE | re.DOTALL)
    if fenced:
        cleaned = fenced.group(1).strip()

    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    first = cleaned.find("{")
    last = cleaned.rfind("}")
    if first >= 0 and last > first:
        parsed = json.loads(cleaned[first : last + 1])
        return parsed if isinstance(parsed, dict) else {}

    return {}


def _normalize_boolean_judgments(
    judgments: list[Any],
    expected_count: int,
    *,
    truth_key: str = "relevant",
) -> list[dict[str, Any]]:
    by_index: dict[int, dict[str, Any]] = {}
    for item in judgments:
        if not isinstance(item, dict):
            continue
        try:
            idx = int(item.get("index"))
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= expected_count:
            by_index[idx] = {
                "index": idx,
                truth_key: bool(item.get(truth_key)),
                "rationale": str(item.get("rationale", ""))[:300],
            }

    normalized: list[dict[str, Any]] = []
    for idx in range(1, expected_count + 1):
        if idx in by_index:
            normalized.append(by_index[idx])
        else:
            normalized.append({
                "index": idx,
                truth_key: False,
                "rationale": "Judge did not return a valid item for this index.",
            })
    return normalized


def _strip_generated_sections(text: str) -> str:
    if not text:
        return ""
    stripped = re.split(r"\n##\s+References\b", text, maxsplit=1, flags=re.IGNORECASE)[0]
    stripped = re.split(r"\n##\s+Retrieval Notes\b", stripped, maxsplit=1, flags=re.IGNORECASE)[0]
    return stripped.strip()


def _extract_inline_citation_ids(text: str) -> list[int]:
    stripped = _strip_generated_sections(text)
    ids: list[int] = []
    for content in re.findall(r"\[([0-9,\s]+)\]", stripped):
        for part in content.split(","):
            try:
                ids.append(int(part.strip()))
            except ValueError:
                continue
    return ids


def _extract_candidate_claims(text: str) -> list[str]:
    stripped = _strip_generated_sections(text)
    stripped = re.sub(r"\[[0-9,\s]+\]", "", stripped)

    claims: list[str] = []
    for raw_line in stripped.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        line = re.sub(r"^[-*]\s+", "", line)
        parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", line)
        for part in parts:
            claim = part.strip()
            if len(claim) < 30:
                continue
            if claim.endswith(":"):
                continue
            claims.append(claim)
    return claims


def _build_evidence_context(retrieved_papers: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    used = 0
    for idx, paper in enumerate(retrieved_papers, start=1):
        title = paper.get("title", "Unknown title")
        abstract = _truncate(paper.get("abstract", ""), MAX_ABSTRACT_CHARS)
        doi = paper.get("doi") or ""
        year = paper.get("year") or ""
        source = paper.get("source") or " / ".join(paper.get("source_names", []))
        block = f"[{idx}] {title}\nSource: {source}; Year: {year}; DOI: {doi}\nAbstract: {abstract}"
        if used + len(block) > MAX_CONTEXT_CHARS:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def _collect_identifiers(records: list[dict[str, Any]]) -> set[str]:
    identifiers: set[str] = set()
    for record in records:
        identifiers.update(_record_identifiers(record))
    return identifiers


def _record_identifiers(record: dict[str, Any]) -> set[str]:
    identifiers: set[str] = set()
    doi = _normalize_doi(record.get("doi"))
    if doi:
        identifiers.add(f"doi:{doi}")

    for value in (record.get("url"), record.get("pmid"), record.get("nct_id")):
        if not value:
            continue
        text = str(value)
        pmid_match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)|\bPMID[:\s]*(\d+)\b", text, re.I)
        if pmid_match:
            identifiers.add(f"pmid:{pmid_match.group(1) or pmid_match.group(2)}")
        elif str(value).isdigit():
            identifiers.add(f"pmid:{value}")
        nct_match = re.search(r"\b(NCT\d{8})\b", text, re.I)
        if nct_match:
            identifiers.add(f"nct:{nct_match.group(1).upper()}")
    return identifiers


def _normalize_identifier(value: str) -> str | None:
    text = (value or "").strip()
    doi = _normalize_doi(text)
    if doi:
        return f"doi:{doi}"

    pmid_match = re.search(r"\bPMID[:\s]*(\d+)\b", text, re.I)
    if pmid_match:
        return f"pmid:{pmid_match.group(1)}"
    if text.isdigit():
        return f"pmid:{text}"

    nct_match = re.search(r"\b(NCT\d{8})\b", text, re.I)
    if nct_match:
        return f"nct:{nct_match.group(1).upper()}"
    return None


def _normalize_doi(value: Any) -> str | None:
    text = str(value or "").strip().lower()
    text = text.removeprefix("https://doi.org/")
    text = text.removeprefix("http://doi.org/")
    text = text.removeprefix("doi:")
    match = re.search(r"10\.\d{4,9}/[^\s]+", text)
    if not match:
        return None
    return match.group(0).rstrip(".,);]")


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


def _truncate(text: str, limit: int) -> str:
    if not text:
        return ""
    text = str(text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _safe_divide(numerator: int | float, denominator: int | float) -> float:
    if not denominator or math.isclose(float(denominator), 0.0):
        return 0.0
    return round(float(numerator) / float(denominator), 4)

