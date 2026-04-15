import json
import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

# Ensure we can import from the eval module
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmark import (
    DEFAULT_DATASET,
    DEFAULT_RESULTS_DIR,
    DEFAULT_MODEL,
    DEFAULT_SOURCES,
    load_dataset,
    filter_dataset,
    run_question,
    summarize_run,
    current_git_branch,
    current_git_commit,
)
from eval.report_generator import (
    write_markdown_report,
    write_csv_export,
    write_radar_chart,
)

st.set_page_config(page_title="Evaluations", page_icon="📊", layout="wide")

st.markdown("# 📊 Clinical Agent Evaluations")
st.caption("Run deterministic benchmarks against the underlying agent pipeline.")

# Load available questions
try:
    dataset = load_dataset(DEFAULT_DATASET)
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

categories = sorted({item["category"] for item in dataset})

with st.sidebar:
    st.header("⚙️ Evaluation Settings")
    st.markdown("---")
    
    selected_categories = st.multiselect(
        "Select Categories to Run",
        options=categories,
        default=categories,
        help="Leave empty to run all categories if no specific questions are selected."
    )
    
    run_limit = st.number_input("Limit Questions", min_value=1, max_value=len(dataset), value=len(dataset), step=1)
    
    st.markdown("### Agent Configuration")
    src_europe_pmc = st.checkbox("Europe PMC", value=True)
    src_pubmed = st.checkbox("PubMed", value=True)
    src_clinical_trials = st.checkbox("ClinicalTrials.gov", value=True)
    
    enabled_sources = []
    if src_europe_pmc: enabled_sources.append("Europe PMC")
    if src_pubmed: enabled_sources.append("PubMed")
    if src_clinical_trials: enabled_sources.append("ClinicalTrials.gov")
    
    model_name = st.text_input("Model", value=DEFAULT_MODEL)
    llm_timeout = st.number_input("LLM Timeout (s)", min_value=30, max_value=600, value=300, step=30)
    retries = st.number_input("Retries", min_value=0, max_value=5, value=1, step=1)
    
    st.markdown("---")
    run_btn = st.button("🚀 Run Benchmark Suite", use_container_width=True, type="primary")

if run_btn:
    # Build list of questions
    selected_q = filter_dataset(
        dataset,
        question_ids=None,
        categories=selected_categories if len(selected_categories) < len(categories) else None,
        limit=run_limit
    )
    
    if not selected_q:
        st.warning("No questions matched the current filters.")
        st.stop()
        
    st.info(f"Loaded {len(selected_q)} questions to evaluate across {len(enabled_sources)} sources.")
    
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"benchmark_{timestamp}"
    results_dir = Path(DEFAULT_RESULTS_DIR)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    progress_bar = st.progress(0, text="Initializing evaluation...")
    status_container = st.container()
    
    with status_container:
        with st.status("Running evaluations in progress...", expanded=True) as status:
            for idx, item in enumerate(selected_q, 1):
                progress_bar.progress((idx - 1) / len(selected_q), text=f"Evaluating [{idx}/{len(selected_q)}]: {item['id']}")
                st.write(f"⏳ Running `{item['id']}`: _{item['question']}_")
                
                try:
                    result = run_question(
                        item,
                        model=model_name,
                        enabled_sources=enabled_sources,
                        llm_timeout=llm_timeout,
                        retries=retries,
                        pubmed_tool_name="DrugTargetAgent",
                        pubmed_email="user@example.com"
                    )
                    results.append(result)
                    
                    comp = result.get('composite_reliability')
                    comp_str = f"{comp:.2f}" if comp is not None else "N/A"
                    
                    if result.get('status') == 'ok':
                        st.write(f"✅ Completed `{item['id']}` — Composite Reliability: **{comp_str}**")
                    else:
                        st.write(f"⚠️ Warning on `{item['id']}` — Status: `{result.get('status')}`")
                        
                except Exception as e:
                    st.error(f"Failed on `{item['id']}`: {e}")
            
            status.update(label="Evaluation completed!", state="complete", expanded=False)
    
    progress_bar.progress(1.0, text="Finalizing reports...")
    
    # Generate reports
    summary = summarize_run(results)
    run_result = {
        "run_id": run_id,
        "timestamp": timestamp,
        "git_branch": current_git_branch(),
        "git_commit": current_git_commit(),
        "model": model_name,
        "enabled_sources": enabled_sources,
        "dataset": str(Path(DEFAULT_DATASET).resolve()),
        "filters": {
            "category": selected_categories,
            "limit": run_limit,
        },
        "settings": {
            "llm_timeout": llm_timeout,
            "retries": retries,
            "pubmed_tool_name": "DrugTargetAgent",
            "pubmed_email": "user@example.com",
        },
        "summary": summary,
        "results": results,
    }
    
    json_path = results_dir / f"{run_id}.json"
    md_path = results_dir / f"{run_id}.md"
    csv_path = results_dir / f"{run_id}.csv"
    chart_path = results_dir / f"{run_id}_radar.png"
    
    write_markdown_report(run_result, md_path)
    write_csv_export(run_result, csv_path)
    json_path.write_text(json.dumps(run_result, indent=2), encoding="utf-8")
    
    chart_success = False
    try:
        write_radar_chart(run_result, chart_path)
        chart_success = True
    except Exception as e:
        st.error(f"Radar chart failed: {e}")
    
    st.divider()
    st.subheader("🏁 Run Summary")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        if chart_success and chart_path.exists():
            st.image(str(chart_path), caption="Evaluation Performance Radar", use_container_width=True)
    
    with col2:
        st.markdown(f"**Composite Reliability:** `{summary.get('composite_reliability', 0):.2f}`")
        if md_path.exists():
            with st.expander("View Full Markdown Report", expanded=False):
                st.markdown(md_path.read_text(encoding="utf-8"))
    
    st.markdown("### Artifacts")
    dl_col1, dl_col2, dl_col3 = st.columns(3)
    
    with dl_col1:
        st.download_button(
            label="📄 Download JSON Results",
            data=json_path.read_text(encoding="utf-8"),
            file_name=f"{run_id}.json",
            mime="application/json"
        )
    with dl_col2:
        st.download_button(
            label="📊 Download CSV Data",
            data=csv_path.read_text(encoding="utf-8"),
            file_name=f"{run_id}.csv",
            mime="text/csv"
        )
    with dl_col3:
        st.download_button(
            label="📝 Download Markdown Report",
            data=md_path.read_text(encoding="utf-8"),
            file_name=f"{run_id}.md",
            mime="text/markdown"
        )
    
    st.success(f"All artifacts saved locally to `{results_dir}`")
else:
    st.info("Configure your settings in the sidebar and press **Run Benchmark Suite** to start.")
