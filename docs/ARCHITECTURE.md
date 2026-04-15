# Reference Architecture: IP-Safe Agentic Retrieval

This document outlines the architectural blueprint for deploying agentic retrieval systems in heavily regulated environments (Pharmaceuticals, Biotechnology, Healthcare) where Intellectual Property (IP) sensitivity and data sovereignty are paramount.

## Problem Statement
In pre-IND (Investigational New Drug) research, competitive intelligence, and early-stage target discovery, pharmaceutical companies cannot risk querying cloud-based Large Language Models. Doing so poses a severe risk of IP leakage: sending proprietary chemical structures, unpatented biological targets, or strategic research queries to external third-party models could expose a company's confidential R&D pipeline.

However, researchers still need the synthesis, extraction, and reasoning capabilities of modern LLMs to navigate massive swaths of biomedical literature.

**The Solution:** A 100% locally homed, air-gapped LLM runtime paired with deterministic federated search plugins. The LLM processes all data internally, while only highly regulated, anonymized keyword searches cross the network boundary to public databases.

## Architecture Overview
The system is built as a generic abstraction consisting of five distinct layers:

### 1. Local LLM Runtime
The foundational engine running the open-weights models (`gemma4:e2b`). Executed via `Ollama`, this layer ensures that all natural language understanding, generative output, and tool-call planning occur purely within the enterprise perimeter.

### 2. Agentic Orchestration Layer
A deterministic routing script that prompts the LLM with a list of available search tools. Using strict JSON schemas, the LLM decides *which* databases to query and *how* to construct the search strings based on the user's input, without directly synthesizing the final answer yet.

### 3. Multi-Source Retrieval Layer
The execution boundary. Python functions execute the LLM's planned JSON queries against parallel APIs (e.g., Europe PMC, PubMed, ClinicalTrials.gov). Responses are parsed, deeply aggregated, and deduplicated before being passed back *into* the local LLM's context window.

### 4. Session Intelligence Layer
Uses a local Vector Database (`ChromaDB`) to persist retrieved abstracts and trial records tied to a unique session ID. This allows for persistent "Session Memory" where researchers can query follow-up questions mathematically linked to prior outputs without repeatedly hitting external APIs.

### 5. Quality Assurance (Verification) Layer
Due to the non-deterministic nature of generative models, a parallel Verification Agent extracts every factual claim made in the generated synthesis. It compares each claim against the specific retrieved abstracts cited, producing a confidence badge and forcing the application to be transparent about its sources.

---

## Data Flow
1. **Input Submission:** A researcher submits query: *"What are the resistance patterns for Imatinib?"*
2. **Orchestration (Tooling):** The Router Agent analyzes the request and generates a JSON function call array limiting queries to `EuropePMC` and `ClinicalTrials`.
3. **Retrieval (Network Boundary):** The system connects to the external REST APIs, fetching the top 10 results from both sources.
4. **Context Injection:** Raw JSON is standardized into a localized markdown format ("Evidence Pack") and cached to ChromaDB.
5. **Synthesis:** The evidence pack is pushed to the local LLM alongside the original query. The LLM generates a comprehensive response heavily punctuated with inline citations (e.g., `[1]`, `[3]`).
6. **Validation:** In the background, the Verification Agent slices the synthesis by citation indices and mathematically scores whether the provided text aligns with the cited context.

## Security Model
**Strict Boundary Constraints:**
- **Inbound Data:** Only public abstracts and metadata are pulled from standard REST endpoints.
- **Outbound Data:** Only keyword strings (parsed and minimized by the Router Agent) leave the network.
- **LLM Context:** No search histories or generation logs are exposed externally. Inference bounds are locked entirely within the host GPU/CPU via Ollama.

## Extensibility Points
This pattern is designed to be easily modified for internal systems:
- **Internal Document Stores:** External APIs (`pubmed.py`) can be seamlessly swapped out with Elasticsearch or internal document embeddings referencing proprietary, in-house PDFs.
- **Verification Rules:** The `verification_agent.py` can be extended to utilize custom enterprise rulebooks or regulatory alignment checks.
- **Model Swaps:** The infrastructure leverages an abstract Ollama client, meaning `llama3`, `mistral`, or any other high-capability open-weight model can drop in seamlessly.

## Trade-offs & Limitations
- **Hardware Bottlenecks:** Local LLMs demand extensive, dedicated accelerated compute architectures (GPUs/NPU clusters).
- **Latency:** Because routing, parsing, synthetic generation, and post-verification happen sequentially and locally, responses often take anywhere from 10 to 60 seconds.
- **Retrieval Ceiling:** The agent's knowledge mapping relies heavily on the quality of basic Keyword APIs; without semantic internal graph representations, it may occasionally fail to fetch hyper-specific studies unless exact keyword matches happen.
