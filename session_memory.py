import os
from pathlib import Path
import chromadb

DATA_DIR = Path("./data/sessions")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Module-level singleton — reuse across all ResearchSession instances
_chroma_client = None

def _get_chroma_client() -> chromadb.ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path=str(DATA_DIR))
    return _chroma_client

class ResearchSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.client = _get_chroma_client()
        self.collection = self.client.get_or_create_collection(
            name=f"session_{self.session_id.replace('-', '_')}"
        )

    def add_papers(self, reference_map: dict[int, dict]):
        """Embeds and saves retrieved papers representing them by DOI or Title."""
        if not reference_map:
            return

        documents = []
        metadatas = []
        ids = []
        
        for k, paper in reference_map.items():
            # Use DOI if available, else a hash of title
            paper_id = paper.get("doi", paper.get("title", f"paper_{k}"))
            
            # Form rich text for embedding: abstract + title
            text = f"Title: {paper.get('title', '')}\nAbstract: {paper.get('abstract', '')}"
            documents.append(text)
            
            meta = {
                "title": paper.get("title", ""),
                "doi": paper.get("doi", ""),
                "abstract": paper.get("abstract", ""),
                "source_type": paper.get("source_type", "")
            }
            metadatas.append(meta)
            ids.append(paper_id)
            
        # Chroma handles deduplication silently if IDs match, but let's upsert
        self.collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )

    def search_papers(self, query: str, n_results: int = 5) -> list[dict]:
        """Search the session for relevant papers using semantic similarity. 
        Returns parsed results suitable for inclusion in prompts."""
        # If collection is empty, querying will throw an error or return empty
        if self.collection.count() == 0:
            return []
            
        # Ensure we don't request more results than we have
        count = min(n_results, self.collection.count())
        
        results = self.collection.query(
            query_texts=[query],
            n_results=count
        )
        
        papers = []
        if results and results.get("documents") and results.get("documents")[0]:
            metas = (results.get("metadatas") or [[{}]])[0] or []
            for meta in metas:
                if not meta: continue
                papers.append({
                    "title": meta.get("title", ""),
                    "doi": meta.get("doi", ""),
                    "abstract": meta.get("abstract", ""),
                    "source_type": meta.get("source_type", "memory"),
                    "source": "Session Memory"
                })
        return papers
