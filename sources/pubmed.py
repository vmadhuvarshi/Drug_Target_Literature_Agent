"""
PubMed retrieval source.

Uses the NCBI E-utilities (ESearch + EFetch) to search PubMed/MEDLINE
and returns results in the standardized schema.
"""

import xml.etree.ElementTree as ET

import requests


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# Defaults — configurable via sidebar
DEFAULT_TOOL_NAME = "DrugTargetAgent"
DEFAULT_EMAIL = "user@example.com"


def search_pubmed(
    query: str,
    limit: int = 5,
    sort: str = "relevance",
    tool_name: str = DEFAULT_TOOL_NAME,
    email: str = DEFAULT_EMAIL,
) -> list[dict]:
    """
    Search PubMed via NCBI E-utilities.

    Two-step process:
      1. ESearch — get PMIDs matching the query
      2. EFetch  — retrieve full article metadata as XML

    Returns a list of dicts following the standardized result schema.
    """
    # ── Step 1: ESearch ──────────────────────────────
    esearch_params = {
        "db": "pubmed",
        "term": query,
        "retmode": "json",
        "retmax": limit,
        "sort": sort,
        "tool": tool_name,
        "email": email,
    }

    resp = requests.get(ESEARCH_URL, params=esearch_params, timeout=30)
    resp.raise_for_status()
    search_data = resp.json()

    id_list = search_data.get("esearchresult", {}).get("idlist", [])
    if not id_list:
        return []

    # ── Step 2: EFetch ───────────────────────────────
    efetch_params = {
        "db": "pubmed",
        "id": ",".join(id_list),
        "rettype": "xml",
        "retmode": "xml",
        "tool": tool_name,
        "email": email,
    }

    resp = requests.get(EFETCH_URL, params=efetch_params, timeout=30)
    resp.raise_for_status()

    return _parse_pubmed_xml(resp.text)


def _parse_pubmed_xml(xml_text: str) -> list[dict]:
    """Parse PubMed EFetch XML into standardized result dicts."""
    root = ET.fromstring(xml_text)
    results = []

    for article_node in root.findall(".//PubmedArticle"):
        medline = article_node.find("MedlineCitation")
        if medline is None:
            continue

        article = medline.find("Article")
        if article is None:
            continue

        # PMID
        pmid_el = medline.find("PMID")
        pmid = pmid_el.text if pmid_el is not None else ""

        # Title
        title_el = article.find("ArticleTitle")
        title = title_el.text if title_el is not None else "No Title"

        # Authors
        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{fore} {last}".strip())

        # Year
        year = None
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    pass
            # Fallback: MedlineDate like "2024 Jan-Feb"
            if year is None:
                medline_date = pub_date.findtext("MedlineDate", "")
                if medline_date:
                    try:
                        year = int(medline_date[:4])
                    except ValueError:
                        pass

        # Abstract
        abstract_parts = []
        abstract_node = article.find("Abstract")
        if abstract_node is not None:
            for abs_text in abstract_node.findall("AbstractText"):
                label = abs_text.get("Label", "")
                text = "".join(abs_text.itertext())
                if label:
                    abstract_parts.append(f"{label}: {text}")
                else:
                    abstract_parts.append(text)
        abstract = " ".join(abstract_parts) if abstract_parts else "No abstract available"

        # DOI
        doi = None
        article_id_list = article_node.find(".//ArticleIdList")
        if article_id_list is not None:
            for aid in article_id_list.findall("ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text
                    break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/" if pmid else ""

        results.append({
            "source": "PubMed",
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "url": url,
            "relevance_snippet": abstract[:300] if abstract else "",
        })

    return results
