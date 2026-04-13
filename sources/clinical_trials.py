"""
ClinicalTrials.gov retrieval source.

Queries the ClinicalTrials.gov v2 API and returns results in the
standardized schema used across all retrieval sources.
"""

import re

import requests


CTGOV_API = "https://clinicaltrials.gov/api/v2/studies"

# Words to strip when building a concise API query
_STRIP_WORDS = {
    "a", "an", "and", "any", "are", "as", "at", "be", "been", "being", "by",
    "can", "could", "do", "does", "for", "from", "has", "have", "how", "if",
    "in", "into", "is", "it", "its", "like", "may", "of", "on", "or", "our",
    "out", "over", "so", "some", "such", "than", "that", "the", "their",
    "them", "then", "there", "these", "they", "this", "those", "to", "was",
    "we", "were", "what", "when", "where", "which", "while", "who", "why",
    "will", "with", "would",
    # Domain-generic verbs/adjectives that don't help API search
    "about", "against", "between", "confer", "confers", "compared",
    "describe", "discussed", "effect", "effects", "explored", "found",
    "given", "known", "mutations", "mutation", "next", "generation",
    "new", "novel", "overcome", "recent", "role", "shown", "through",
    "used", "using", "via",
}

# Question-opening fragments to remove
_QUESTION_PREFIXES = [
    r"^what\s+are\s+",
    r"^how\s+do\s+",
    r"^how\s+does\s+",
    r"^can\s+you\s+",
    r"^describe\s+",
    r"^explain\s+",
    r"^summarize\s+",
    r"^summarise\s+",
    r"^find\s+",
    r"^search\s+for\s+",
    r"^tell\s+me\s+about\s+",
]


def _sanitize_ct_query(raw_query: str, max_keywords: int = 10, max_length: int = 80) -> str:
    """
    Convert a natural-language question into a concise keyword query
    suitable for the ClinicalTrials.gov v2 API.

    The API returns 400 Bad Request on long free-text strings and on
    queries containing hyphens.  It also returns 0 results when too
    many terms are combined (implicit AND), so we aggressively trim.
    """
    text = raw_query.strip()

    # Remove question-opening fragments
    for pattern in _QUESTION_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Remove trailing question marks and extra punctuation
    text = re.sub(r"[?!]+$", "", text)

    # Replace hyphens with spaces (the API rejects hyphenated tokens)
    text = text.replace("-", " ")

    # Tokenize (alphanumeric, slashes, plus signs only — no hyphens)
    tokens = re.findall(r"[A-Za-z0-9/+]{2,}", text)
    keywords = [t for t in tokens if t.lower() not in _STRIP_WORDS]

    # Cap number of keywords so the implicit AND doesn't zero-out results
    keywords = keywords[:max_keywords]

    # Rebuild and truncate to max_length on word boundaries
    result = " ".join(keywords)
    if len(result) > max_length:
        truncated = result[:max_length]
        last_space = truncated.rfind(" ")
        if last_space > 0:
            truncated = truncated[:last_space]
        result = truncated

    return result or raw_query[:max_length]


def search_clinical_trials(query: str, limit: int = 5) -> list[dict]:
    """
    Search ClinicalTrials.gov for clinical trial records.

    Returns a list of dicts following the standardized result schema.
    """
    clean_query = _sanitize_ct_query(query)

    params = {
        "query.term": clean_query,
        "pageSize": limit,
        "format": "json",
    }

    response = requests.get(CTGOV_API, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = []
    for study in data.get("studies", []):
        protocol = study.get("protocolSection", {})

        # Identification
        id_module = protocol.get("identificationModule", {})
        nct_id = id_module.get("nctId", "")
        title = (
            id_module.get("officialTitle")
            or id_module.get("briefTitle")
            or "No Title"
        )

        # Sponsor as "author"
        authors = []
        sponsor_module = protocol.get("sponsorCollaboratorsModule", {})
        lead_sponsor = sponsor_module.get("leadSponsor", {})
        sponsor_name = lead_sponsor.get("name", "")
        if sponsor_name:
            authors.append(sponsor_name)

        # Year from start date
        year = None
        status_module = protocol.get("statusModule", {})
        start_date = status_module.get("startDateStruct", {}).get("date", "")
        if start_date:
            try:
                # Format is typically "YYYY-MM-DD" or "YYYY-MM" or "YYYY"
                year = int(start_date[:4])
            except (ValueError, IndexError):
                pass

        # Brief summary as abstract
        desc_module = protocol.get("descriptionModule", {})
        abstract = desc_module.get("briefSummary", "No summary available")

        # URL
        url = f"https://clinicaltrials.gov/study/{nct_id}" if nct_id else ""

        results.append({
            "source": "ClinicalTrials.gov",
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": None,  # Clinical trials do not have DOIs
            "url": url,
            "relevance_snippet": abstract[:300] if abstract else "",
        })

    return results
