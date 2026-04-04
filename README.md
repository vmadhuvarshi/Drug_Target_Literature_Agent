# 🧬 Drug Target Literature Agent

A locally-run clinical research assistant powered by **Gemma 4** (via Ollama) that uses **agentic retrieval** to search [Europe PMC](https://europepmc.org/) in real time and produce cited literature summaries.

> **Why local?** Pharmaceutical and biotech research often involves IP-sensitive queries. This tool keeps everything on your machine — no data leaves your network.

## How It Works

1. You ask a natural-language question about a drug-target interaction.
2. Gemma 4 autonomously decides to call the `search_literature` tool.
3. The tool fetches real-time results from the **Europe PMC REST API**.
4. Gemma synthesises a summary with **in-line citations** and a **References** section.

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

| File | Description |
|---|---|
| `app.py` | Streamlit chat UI — the main application |
| `prompt_gemma.py` | Standalone CLI script for testing the agent loop |
| `requirements.txt` | Python dependencies |

## Example Queries

- *What are the mechanisms of resistance to Imatinib in BCR-ABL+ CML?*
- *Find recent studies on PARP inhibitor Olaparib resistance in BRCA cancers.*
- *Summarise clinical evidence for CDK4/6 inhibitors in HR+ breast cancer.*

## License

MIT
