"""
Europe PMC retrieval source.

Queries the Europe PMC REST API and returns results in the
standardized schema used across all retrieval sources.
"""

import requests


EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def search_europe_pmc(query: str, limit: int = 5) -> list[dict]:
    """
    Search Europe PMC for biomedical literature.

    Returns a list of dicts following the standardized result schema.
    """
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": limit,
    }

    response = requests.get(EUROPE_PMC_API, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("resultList", {}).get("result", []):
        # Extract author names from the authorList structure
        authors = []
        author_list = item.get("authorList", {}).get("author", [])
        if isinstance(author_list, list):
            authors = [a.get("fullName", "") for a in author_list if a.get("fullName")]

        # Build a direct URL
        pmid = item.get("pmid", "")
        source_db = item.get("source", "MED")
        url = f"https://europepmc.org/article/{source_db}/{pmid}" if pmid else ""

        # Year
        year_raw = item.get("pubYear", "")
        try:
            year = int(year_raw)
        except (ValueError, TypeError):
            year = None

        # DOI
        doi = item.get("doi") or None

        # Abstract
        abstract = item.get("abstractText", "No abstract available")

        results.append({
            "source": "Europe PMC",
            "title": item.get("title", "No Title"),
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "url": url,
            "relevance_snippet": abstract[:300] if abstract else "",
        })

    return results
