# 🧬 Drug Target Literature Agent

A locally-run clinical research assistant powered by **Gemma 4** (via Ollama) that uses **agentic retrieval** to fetch real-time data from multiple literature and clinical trial databases. It produces highly accurate, synthesized summaries with in-line citations and a second-pass quality gate.

> **Why local?** Pharmaceutical and biotech research often involves IP-sensitive queries. This tool keeps everything on your machine — no data leaves your network.

## Key Features

- **Multi-Source Retrieval Router**: Intelligently routes natural language queries to search APIs across **Europe PMC**, **PubMed**, and **ClinicalTrials.gov**.
- **Automated Deduplication**: Aggregates duplicate papers and redundant trial results globally across all enabled data sources.
- **Exhaustive Synthesis**: Gemma 4 integrates all selected evidence into a unified response with in-line citations linking back to the precise data sources.
- **Citation Verification Agent**: An autonomous second-pass LLM pipeline checks every generated factual claim against its cited source abstract. Prevents hallucinations and provides a clear Confidence Badge + Claim details UI.

## How It Works

1. You ask a natural-language question targeting a drug-target interaction.
2. Gemma 4's primary agent identifies the most appropriate queries and limits and executes search tool calls against the enabled databases.
3. The retrieved results are pooled, deduplicated, and fed back directly into the context window.
4. Gemma synthesizes a detailed summary with **in-line citations** referring to a clearly enumerated **References** section (grouped by "Cited" and "Uncited").
5. The **Verification Agent** spins up in parallel threads, verifying each extracted synthetic claim against the specific cited abstract text, generating a confidence score.

## Quick Start

### Prerequisites

- [Python 3.10+](https://www.python.org/downloads/)
- [Ollama](https://ollama.com/) installed and running
- The Gemma 4 model pulled locally:
  ```bash
  ollama pull gemma4:e2b
  ```

### Install & Run

```bash
# Clone the repo
git clone https://github.com/vmadhuvarshi/Drug_Target_Literature_Agent.git
cd Drug_Target_Literature_Agent

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

## Project Structure

| File/Folder | Description |
|---|---|
| `app.py` | Streamlit chat UI — the main application |
| `retrieval_router.py` | Core agentic routing, deduplication, and synthesis formatting |
| `verification_agent.py` | Multi-threaded claim extraction and verification pipeline |
| `sources/` | Search integration functions (`pubmed.py`, `europe_pmc.py`, `clinical_trials.py`) |
| `models/` | Pydantic data schemas representing Verification and Output logic |
| `prompts/` | Prompt injection templates (e.g., `verification_system_prompt.txt`) |
| `requirements.txt` | Python dependencies (streamlit, ollama, requests, pydantic) |

## Example Queries

- *What are the mechanisms of resistance to Imatinib in BCR-ABL+ CML?*
- *Find recent studies on PARP inhibitor Olaparib resistance in BRCA cancers.*
- *Summarise clinical evidence for CDK4/6 inhibitors in HR+ breast cancer.*
- *Are there any ongoing clinical trials for CAR-T therapy in glioblastoma?*

## Evaluation Harness

Run the benchmark suite after Ollama is running and dependencies are installed:

```bash
pip install -r requirements.txt
bash run_eval.sh
```

The harness evaluates 25 curated drug-target questions across mechanism,
resistance, clinical evidence, safety, and emerging-target categories. Results
are timestamped under `eval/results/` as JSON, Markdown, CSV, and a radar chart.

Useful filters:

```bash
bash run_eval.sh --question-id moa_01
bash run_eval.sh --category "Clinical evidence"
make eval
```

## License

MIT
