# 🧬 Reference Implementation: IP-Safe Agentic Retrieval

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Require: Ollama](https://img.shields.io/badge/Runtime-Ollama-purple.svg)](https://ollama.com/)

A reference implementation of a completely localized, IP-safe agentic retrieval system designed for pharmaceutical and clinical research. Powered by open-weights local inferencing tools (**Gemma 4** via Ollama) and an orchestrating **Streamlit** dashboard.

> **Why Local?**
> Pharmaceutical R&D (such as pre-IND molecule targeting, competitive intelligence pipelines, and pharmacovigilance) involves extremely sensitive intellectual property. Commercial off-the-shelf cloud LLMs process query history remotely, creating severe IP leakage risks. This architecture enforces a strict boundary: the LLM, the prompts, and the synthetic orchestration stay entirely air-gapped on localized enterprise hardware. The only artifacts leaving the physical network boundary are normalized search terms hitting public clinical REST APIs.

## Architecture Topology

![System Architecture](docs/diagrams/system_architecture.mmd)

This blueprint relies heavily on a 5-layer retrieval model described comprehensively in the [Architecture Documentation](docs/ARCHITECTURE.md).

## Key Features

- **Local LLM Runtime**: Entirely powered by Ollama bindings running Gemma algorithms locally, requiring zero cloud dependency for reasoning.
- **Agentic Orchestration**: Models autonomously negotiate keyword routing, filtering, and API endpoint selection utilizing deterministic JSON tool schemas.
- **Multi-Source Retrieval Federator**: Seamlessly retrieves, aggregates, and deduplicates metadata across public databases (**Europe PMC**, **PubMed**, and **ClinicalTrials.gov**).
- **Quality Assurance Verification**: A built-in second-pass extraction agent verifies generated claims against raw literature texts to rigorously guard against hallucinated citations.
- **Session Intelligence**: Uses **ChromaDB** caching to vector-track previous research conversations and evidence packets for seamless UI and logic continuity.

## Documentation Navigation

This repository heavily emphasizes documentation and structural mapping over code itself, specifically targeting technology leaders evaluating agentic architectures.

* [**Architecture Details**](docs/ARCHITECTURE.md) - Deep dive into boundary restrictions, network data flow, and limitations.
* [**Pattern Adaptation Guide**](docs/PATTERN_GUIDE.md) - Instructions for how to rapidly re-fork this pattern for Regulatory Document Search, Pharmacovigilance, or Competitive Intelligence pipelines.
* [**Evaluation Methodology**](docs/EVAL_METHODOLOGY.md) - Explainers on the MLOps automated local benchmarking logic scoring Factual Coverage and Hallucination Control.
* **Diagrams Directory:** (`docs/diagrams/`)
  * `system_architecture.mmd`
  * `agent_loop.mmd`
  * `session_flow.mmd`
  * `deployment.mmd`

## Quick Start

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/) installed and natively running on your acceleration hardware.
- The default reference model downloaded:
  ```bash
  ollama pull gemma4:e2b
  ```

### Install & Run

```bash
# Clone the enterprise architecture reference repo
git clone https://github.com/vmadhuvarshi/Drug_Target_Literature_Agent.git
cd Drug_Target_Literature_Agent

# Install the Python dependencies (Streamlit, Pydantic, ChromaDB, etc.)
pip install -r requirements.txt

# Launch the visual interface dashboard
streamlit run app.py
```

Clicking into the sidebar will also offer native execution capabilities for the local Evaluation Benchmarks.

---
**License**: MIT
