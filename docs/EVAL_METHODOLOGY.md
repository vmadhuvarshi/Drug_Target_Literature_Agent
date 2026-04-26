# Evaluation Methodology

Regulated industries require measurable, reproducible validation of any tool used in their workflows. Generative AI introduces non-deterministic outputs, which means **you can't deploy what you can't systematically measure.**

The evaluation harness provides automated, reproducible scoring against the agentic retrieval pipeline, tracking whether hallucination rates drift as you iterate on prompts, models, or retrieval strategies.

## Metrics

The benchmark measures the agent across five dimensions:

1. **Retrieval Precision (LLM as a judge)**
   - *What it measures:* The fraction of retrieved papers that are actually relevant to the question being asked.
   - *Why it matters:* Low precision means the routing agent generated poor search queries, filling the context window with noise and degrading synthesis quality.

2. **Retrieval Recall (Pattern Matching)**
   - *What it measures:* Whether the retrieved results contain specific landmark papers (identified by DOI or PMID) that should appear for a given topic.
   - *Why it matters:* Ensures the synthesis is built on comprehensive literature, not just whatever the API happened to return first.

3. **Citation Accuracy (Deterministic Parse)**
   - *What it measures:* Whether every bracketed citation (e.g., `[1]`) maps to a real document in the reference list for that run.
   - *Why it matters:* Catches fabricated citations — a common failure mode in retrieval-augmented generation.

4. **Factual Coverage (LLM as a judge)**
   - *What it measures:* Whether the synthesized response covers all the key clinical findings expected for the topic.
   - *Why it matters:* LLMs frequently drop important nuance (adverse events, resistance subtypes, dosing details) in shorter responses.

5. **Hallucination Control (LLM as a judge)**
   - *What it measures:* Extracts only claims that carry inline citations (sentences containing `[1]`, `[3]`, etc.) and checks whether each claim is topically consistent with the cited paper's title and abstract. Scored as a rate, then converted to a control ratio `(1.0 - rate)` for the composite.
   - *Why it matters:* This is the final accuracy gatekeeper. By evaluating only cited claims, the metric avoids false positives from correct background knowledge the model uses in connecting prose.

## Interpreting Results

Results are saved to `eval/results/` and visualized as a radar chart in the Evaluations dashboard.

- **Baseline expectations:** Composite reliability scores will vary significantly by model size. A 4B model (Gemma 4) will typically score 0.5–0.8 on well-known drug-target topics. A 70B model will score higher. This is expected — the architecture is model-agnostic, so you can upgrade the model without changing anything else.
- **Composite Reliability:** An equally-weighted average of all five metrics. Scores above 0.8 indicate strong performance for the given model and domain.

## Landmark Paper Lookup

The benchmark dataset specifies `expected_sources` (landmark DOIs and PMIDs) for each question. Since keyword search may not always surface specific older papers, the harness includes a **direct-lookup fallback** (`sources/doi_lookup.py`) that fetches missing expected papers via the Europe PMC and PubMed APIs. These are injected into the pool before scoring, so retrieval recall measures the pipeline's coverage fairly rather than penalizing API search ranking behavior.

This fallback is **benchmark-only** and does not affect the live application.

## Extending the Harness

The benchmark dataset (`eval/datasets/benchmark_questions.json`) is fully customizable. To adapt for a different domain (e.g., pharmacovigilance), replace the default questions with scenarios matching your evaluation needs:

```json
{
  "id": "scenario_01",
  "category": "Adverse Events",
  "question": "What are the common hepatic side effects of molecule X?",
  "expected_key_findings": ["Liver toxicity", "Elevated ALT", "Hepatitis onset"],
  "expected_sources": ["PubMed"],
  "difficulty": "medium"
}
```

Run `python -m eval.benchmark` after editing the dataset to score your changes immediately.
