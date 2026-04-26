# Pattern Adaptation Guide

This architecture is designed to be adapted. The five-layer structure (LLM runtime → routing → retrieval → session memory → verification) applies to any domain where you need LLM-powered synthesis over sensitive queries without cloud exposure.

Below are three concrete scenarios showing how to modify the reference pattern.

---

## Scenario 1: Pharmacovigilance Literature Monitoring

**Goal:** Automatically scan biomedical literature for emerging adverse drug reactions or off-label use reports, without revealing which products you're monitoring.

### What to change:

- **`sources/europe_pmc.py`** — Modify the search query builder to scope results to pharmacovigilance-relevant terms. Add date filtering (e.g., `pub_date > 30 days`) to focus on recent publications only.
- **`retrieval_router.py`** — Update the `ROUTER_SYSTEM_PROMPT` to instruct the LLM to always include adverse event terminology in its search queries.
- **`prompts/verification_system_prompt.txt`** — Rewrite the verification prompt to check for "adverse event extracted" rather than "drug mechanism supported."
- **`session_memory.py`** — Use the ChromaDB session to track previously reviewed paper DOIs, so subsequent runs skip already-seen articles.

---

## Scenario 2: Regulatory Document Q&A (Internal Stores)

**Goal:** Search through internal IND submissions, Clinical Study Reports, and trial protocols to answer regulatory questions — while keeping everything local.

### What to change:

- **`sources/`** — Replace the public API modules entirely. Create a new `search_internal_csr_store.py` targeting your internal search engine (Elasticsearch, Milvus, or a SharePoint REST integration). Return results in the same `{title, abstract, doi, url, authors, year}` schema.
- **Security posture** — Since everything is internal, you can upgrade the retrieval tools to fetch full-text documents rather than just abstracts.
- **`retrieval_router.py`** — Update `SOURCE_TOOL_MAP` and `FUNCTION_DISPATCH` to register your new source. Update the tool schema descriptions so the LLM knows when to use it.
- **`TOOL_SCHEMAS`** — Allow the agent to use chain-of-thought tool calls, drilling down from directory search to specific paragraph extractions.

---

## Scenario 3: Pre-Clinical Competitive Intelligence

**Goal:** Monitor early-phase biotech activity — press releases, patent filings, clinical trial registrations — to map competitor strategies without exposing your own focus areas.

### What to change:

- **`sources/`** — Add API integrations for USPTO (patent search), financial news APIs (AlphaSense, Bloomberg, etc.), and preprint servers (bioRxiv, medRxiv).
- **`retrieval_router.py`** — Update the deduplication logic (`SequenceMatcher` in `_deduplicate_evidence`) to handle boilerplate text common in patent filings and press releases.
- **`verification_agent.py`** — Adapt the verification prompt to assess claims based on "competitive signal strength" rather than "biological mechanism support."
- **`prompts/verification_system_prompt.txt`** — Rewrite to focus on factual extraction from financial and patent language.

---

## Getting Started (Adapting the Pattern)

1. Clone this repo and get the base app running with `streamlit run app.py`.
2. Decide which `sources/` modules you need. Disable or remove the ones you don't.
3. Write your new retrieval module in `sources/`. Return a list of dicts with at minimum `title`, `abstract`, and an identifier (`doi`, `url`, or similar).
4. Update `SOURCE_TOOL_MAP`, `FUNCTION_DISPATCH`, and `SOURCE_FUNCTION_MAP` in `retrieval_router.py` to register your new source. Add a tool schema so the LLM knows when to use it.
5. Run the app on an internal server accessible to your team.
