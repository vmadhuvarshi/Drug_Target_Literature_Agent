# Architecture: IP-Safe Agentic Retrieval

How this system works, why it's built this way, and where the security boundaries are.

## Problem

A researcher exploring an unpatented kinase target can't type that query into a cloud LLM without potentially exposing their R&D direction to the model provider. The same applies to competitive intelligence queries, pre-IND compound names, and any unpublished biological hypothesis. Legal and InfoSec teams in pharma increasingly block cloud LLM tools for this reason.

Researchers still need LLM-grade synthesis to navigate the biomedical literature efficiently. The question is whether you can get that capability without sending sensitive queries off-network.

**This system's answer:** run the LLM locally. The only data that leaves the machine is a search keyword hitting a public API — the same kind of request PubMed has processed from browsers for decades.

## Five-Layer Architecture

### 1. Local LLM Runtime

Ollama runs Gemma 4 locally. All reasoning, synthesis, and tool-call planning happen on the researcher's machine — nothing is sent to an external model provider. The model is configured in `models/config.py` and can be swapped via the `OLLAMA_MODEL` environment variable.

### 2. Routing Agent

The retrieval router (`retrieval_router.py`) prompts the LLM with a list of available search tools using function calling. The model decides which databases to query and how to phrase the search, without generating the final answer yet. If the model doesn't support function calling natively, the router falls back to a prompt-based JSON approach.

### 3. Multi-Source Retrieval

Python functions execute the planned queries against public APIs: Europe PMC, PubMed, and ClinicalTrials.gov. Europe PMC results are sorted by citation count (`sort=cited`) and PubMed by relevance score (`sort=relevance`), so landmark papers surface alongside recent publications. Results are parsed, deduplicated, and reranked before being loaded into the LLM's context window.

### 4. Session Memory

A local ChromaDB vector store persists retrieved abstracts and trial records tied to a session ID. This gives researchers persistent context — follow-up questions can draw on semantically related prior results without re-querying the external APIs.

### 5. Citation Verification

A second-pass verification agent extracts every cited claim from the synthesis and checks whether the cited paper's abstract actually supports it. The result is a confidence badge shown in the UI. This doesn't guarantee correctness, but it catches obvious hallucinations and fabricated citations.

---

## Data Flow

1. **Query:** Researcher asks *"What are the resistance patterns for Imatinib?"*
2. **Routing:** The agent generates function calls targeting Europe PMC + ClinicalTrials.gov.
3. **Retrieval:** HTTP requests go to the public APIs. Only search keywords leave the machine.
4. **Evidence packing:** Raw API responses are standardized, deduplicated, and cached to ChromaDB.
5. **Synthesis:** The evidence pack + original query go to the local LLM. It generates a response with inline citations (`[1]`, `[3]`).
6. **Verification:** The verification agent checks each cited claim against the source abstract and produces a confidence score.

## Security Model

The only outbound traffic is HTTP GET requests to public APIs (Europe PMC, PubMed, ClinicalTrials.gov) containing search keywords. No prompt text, no synthesized output, no session history leaves the machine.

- **Inbound:** Public abstracts and metadata from standard REST endpoints.
- **Outbound:** Keyword search strings only, constructed by the routing agent.
- **LLM context:** Stays entirely within the local Ollama process. No logs, prompts, or completions are transmitted externally.

## Extensibility

- **Internal document stores:** Replace the `sources/` modules with Elasticsearch, Milvus, or any internal search API. The router doesn't care where the evidence comes from as long as it follows the standard schema.
- **Verification rules:** Extend `verification_agent.py` with domain-specific checks (regulatory alignment, adverse event detection, etc.).
- **Model swaps:** Change `OLLAMA_MODEL` in environment or config. The architecture is model-agnostic — `llama3`, `mistral`, `gemma`, or any Ollama-supported model works.

## Trade-offs & Limitations

- **Model quality vs. privacy:** Local models at 4B parameters produce noticeably weaker synthesis than GPT-4-class models. For production use, run a larger model (70B+) on a GPU server. The architecture doesn't change.
- **Latency:** Routing, retrieval, synthesis, and verification all run sequentially and locally. Expect 15–60 seconds per query depending on model size and hardware.
- **Retrieval ceiling:** The agent relies on keyword search against public APIs. It will miss papers that don't match the keywords, and can't access paywalled full text. For specific known papers, the benchmark uses a DOI/PMID direct-lookup fallback.
