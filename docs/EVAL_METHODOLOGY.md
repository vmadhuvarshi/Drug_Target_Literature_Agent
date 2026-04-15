# Evaluation Methodology in Regulated Contexts

In enterprise, heavily regulated settings such as clinical and pharmaceutical tech, deterministic software controls are legally and functionally mandated. Generative AI fundamentally introduces non-deterministic outputs. Therefore, **you cannot deploy what you cannot systematically measure.**

The automated evaluation harness provides mathematical, time-stamped rigor directly against the agentic retrieval architecture, scoring pipeline iterations to ensure hallucination rates do not drift.

## Evaluated Dimensions and Metrics

The benchmark mathematically evaluates the underlying agent across 5 isolated dimensions:

1. **Retrieval Precision (LLM as a judge)**
   - *What it measures:* The fraction of the evidence pulled by the search APIs that was actually clinically relevant to the drug and target in question.
   - *Why it matters:* Poor precision indicates the routing agent generated bad tool calls, burying the LLM context window in noise and reducing synthesis quality.

2. **Retrieval Recall (Pattern Matching)**
   - *What it measures:* Whether the search results contain specific landmark identifiers (like DOIs or PMIDs) expected to be found for a baseline understanding of the question.
   - *Why it matters:* Ensures the LLM is synthesizing over comprehensive literature rather than missing the underlying gold-standard papers.

3. **Citation Accuracy (Deterministic Parse)**
   - *What it measures:* Verifies that every single bracketed citation (e.g., `[1]`) corresponds to a valid document index injected during that specific run.
   - *Why it matters:* Stops the model from faking citations or fabricating index numbers, a common failure mode in retrieval generation.

4. **Factual Coverage (LLM as a judge)**
   - *What it measures:* Grades whether the final synthesized output explicitly explicitly covered every single underlying clinical observation expected for the topic.
   - *Why it matters:* Over-constrained LLMs frequently drop critical nuance such as adverse events or mild resistance trends in shorter output responses.

5. **Hallucination Control (LLM as a judge)**
   - *What it measures:* Scores the ratio by which generated factual statements can securely trace back to an identified abstract excerpt. Scored initially as a rate, and converted to a 'control' ratio `(1.0 - rate)` for the composite graph.
   - *Why it matters:* This acts as the final gatekeeper for generative accuracy.

## Interpreting Benchmark Results

Results are saved to `eval/results/` and visualized automatically onto a radar chart interface located through the UI's Evaluations dashboard.

- **Baseline:** Expect most open-source models (like Gemma 4:e2b) to achieve high citation accuracy due to rigid system prompts, but experience variance in Factual Coverage depending on API limitations.
- **Composite Reliability:** A blended aggregate mathematical score merging all 5 metrics equally. High 0.8+ suggests enterprise-readiness on the specific scenario base.

## Extending the Harness

The JSON structure housing the evaluations (`eval/datasets/benchmark_questions.json`) is fully customizable.
When adapting this pattern for unique architectures (like Pharmacovigilance), you should delete the default clinical trial questions and write scenarios matching your specific evaluation standards:

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
Run `python -m eval.benchmark` sequentially after altering the dataset to instantly assess your new domain adjustments.
