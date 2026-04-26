"""
DOI / PMID direct-lookup utilities.

Used by the benchmark harness to fetch specific landmark papers that
keyword search may miss.  Not used by the live Streamlit app.
"""

from __future__ import annotations

from typing import Any

import requests


EUROPE_PMC_API = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"


def lookup_by_doi(doi: str) -> dict[str, Any] | None:
    """Fetch a single paper from Europe PMC by exact DOI.

    Returns a standardized paper dict, or None if not found.
    """
    clean_doi = doi.strip().removeprefix("DOI:").removeprefix("doi:")
    params = {
        "query": f'DOI:"{clean_doi}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }
    try:
        resp = requests.get(EUROPE_PMC_API, params=params, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("resultList", {}).get("result", [])
        if not results:
            return None
        item = results[0]
        authors = []
        author_list = item.get("authorList", {}).get("author", [])
        if isinstance(author_list, list):
            authors = [a.get("fullName", "") for a in author_list if a.get("fullName")]
        pmid = item.get("pmid", "")
        source_db = item.get("source", "MED")
        url = f"https://europepmc.org/article/{source_db}/{pmid}" if pmid else ""
        year = None
        try:
            year = int(item.get("pubYear", ""))
        except (ValueError, TypeError):
            pass
        return {
            "source": "Europe PMC",
            "title": item.get("title", "No Title"),
            "authors": authors,
            "year": year,
            "abstract": item.get("abstractText", "No abstract available"),
            "doi": item.get("doi") or clean_doi,
            "url": url,
            "pmid": pmid,
            "relevance_snippet": (item.get("abstractText", "") or "")[:300],
        }
    except Exception:
        return None


def lookup_by_pmid(
    pmid: str,
    tool_name: str = "DrugTargetAgent",
    email: str = "user@example.com",
) -> dict[str, Any] | None:
    """Fetch a single paper from PubMed by exact PMID.

    Returns a standardized paper dict, or None if not found.
    """
    import xml.etree.ElementTree as ET

    clean_pmid = pmid.strip().removeprefix("PMID:").removeprefix("pmid:")
    params = {
        "db": "pubmed",
        "id": clean_pmid,
        "rettype": "xml",
        "retmode": "xml",
        "tool": tool_name,
        "email": email,
    }
    try:
        resp = requests.get(EFETCH_URL, params=params, timeout=15)
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        article_node = root.find(".//PubmedArticle")
        if article_node is None:
            return None
        medline = article_node.find("MedlineCitation")
        if medline is None:
            return None
        article = medline.find("Article")
        if article is None:
            return None

        pmid_el = medline.find("PMID")
        pmid_val = pmid_el.text if pmid_el is not None else clean_pmid

        title_el = article.find("ArticleTitle")
        title = title_el.text if title_el is not None else "No Title"

        authors = []
        author_list = article.find("AuthorList")
        if author_list is not None:
            for author in author_list.findall("Author"):
                last = author.findtext("LastName", "")
                fore = author.findtext("ForeName", "")
                if last:
                    authors.append(f"{fore} {last}".strip())

        year = None
        pub_date = article.find(".//PubDate")
        if pub_date is not None:
            year_el = pub_date.find("Year")
            if year_el is not None and year_el.text:
                try:
                    year = int(year_el.text)
                except ValueError:
                    pass
            if year is None:
                medline_date = pub_date.findtext("MedlineDate", "")
                if medline_date:
                    try:
                        year = int(medline_date[:4])
                    except ValueError:
                        pass

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

        doi = None
        article_id_list = article_node.find(".//ArticleIdList")
        if article_id_list is not None:
            for aid in article_id_list.findall("ArticleId"):
                if aid.get("IdType") == "doi":
                    doi = aid.text
                    break

        url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid_val}/" if pmid_val else ""

        return {
            "source": "PubMed",
            "title": title,
            "authors": authors,
            "year": year,
            "abstract": abstract,
            "doi": doi,
            "url": url,
            "pmid": pmid_val,
            "relevance_snippet": abstract[:300] if abstract else "",
        }
    except Exception:
        return None
