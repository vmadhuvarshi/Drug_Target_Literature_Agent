# Pattern Adaptation Guide

The "IP-Safe Agentic Retrieval" architecture is designed as a reusable reference pattern. While originally formulated to analyze Drug Target mechanisms across broad public clinical APIs, its five-layer architecture can be rapidly reforked and integrated into distinct regulated areas.

Below are three adaptation scenarios demonstrating how to morph the reference pattern.

---

## Scenario 1: Pharmacovigilance Literature Monitoring
**Goal:** Automating the real-time scanning of medical literature to identify emerging adverse drug reactions (ADRs) or off-label use cases without compromising internal focus points.

### Required Changes:
- **Retrieval Layer Modification:** Swap generic `search_europe_pmc` prompts to strictly query specialized pharmacovigilance streams or filtered E-Utilities subsets matching known product lines.
- **Agent Orchestrator:** Force the LLM JSON routing to permanently scope queries to time-boxed windows (e.g., `pub_date > 30 days`).
- **Verification Rules:** Shift the Verification Agent's prompt (`prompts/verification_system.txt`) away from testing for 'drug mechanisms' toward checking explicitly for "Adverse Event Extracted" validation metrics.
- **Session Intelligence:** Use ChromaDB to track already-reviewed papers so subsequent runs explicitly filter out previously seen article DOIs.

---

## Scenario 2: Regulatory Document Q&A (Internal Stores)
**Goal:** Searching through historical IND submissions, internal Clinical Study Reports (CSRs), and trial protocols to answer regulatory queries while keeping proprietary secrets localized.

### Required Changes:
- **Retrieval Layer Modification:** Completely sever ties with public URLs. Build a new tool (`search_internal_csr_store`) targeting an internal enterprise search engine (like Milvus, Elasticsearch, or a localized SharePoint REST integration).
- **Security Posture:** Since everything is internal, you can upgrade the retrieval tools to fetch full-text documents rather than just abstracts.
- **Orchestrator:** Allow the agent to use chain-of-thought tool calls, drilling down from directory search to specific paragraph extractions.

---

## Scenario 3: Pre-Clinical Competitive Intelligence
**Goal:** Scraping early-phase biotech press releases, financial disclosures, and experimental patent filings to map competitor strategies securely. 

### Required Changes:
- **Retrieval Layer Modification:** Integrate APIs such as the USPTO (Patent Office) or financial news scrapers (e.g., AlphaSense/Bloomberg integrations if permitted under compliance).
- **Data Deduplication:** Since patents and press releases often mirror identical text, enhance the `SequenceMatcher` deduplication logic located in `retrieval_router.py` to strip out boilerplates.
- **Verification:** Since competitive intelligence relies heavily on sentiment analysis, adapt the verification agent to grade claims based on "Financial Impact" versus "Biological Efficacy."

---

## Getting Started (Forking the Pattern)
To rapidly adapt this pattern for your own organization:
1. Clone the core logic surrounding `retrieval_router.py`.
2. Disable or comment out the `sources/` configurations you do not intend to use.
3. Write your new API integration inside the `sources/` directory. Be sure to return a strict list of dictionaries containing `id`, `title`, and `abstract`.
4. Update `TOOL_SCHEMAS` in the router module with description hints teaching the LLM when to use your new source.
5. Deploy `app.py` onto an isolated internal server handling HTTPS requests over your organizational VPN.
