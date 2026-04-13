import threading

import streamlit as st

from models.verification import VerificationReport, VerificationStatus
from retrieval_router import route_and_retrieve
from sources.pubmed import DEFAULT_TOOL_NAME, DEFAULT_EMAIL
from verification_agent import verify_all

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Gemma 4 Clinical Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_NAME = "gemma4:e2b"

# Badge colors per source
SOURCE_COLORS = {
    "Europe PMC": "#6366f1",       # Indigo
    "PubMed": "#10b981",           # Emerald
    "ClinicalTrials.gov": "#f59e0b",  # Amber
}

# Status display config
_STATUS_CONFIG = {
    VerificationStatus.SUPPORTED: {
        "icon": "✅",
        "label": "Supported",
        "bg": "rgba(16, 185, 129, 0.08)",
        "border": "#10b981",
    },
    VerificationStatus.PARTIALLY_SUPPORTED: {
        "icon": "⚠️",
        "label": "Partially supported",
        "bg": "rgba(245, 158, 11, 0.08)",
        "border": "#f59e0b",
    },
    VerificationStatus.NOT_SUPPORTED: {
        "icon": "❌",
        "label": "Not supported",
        "bg": "rgba(239, 68, 68, 0.08)",
        "border": "#ef4444",
    },
}

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Global ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    }
    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(148, 163, 184, 0.2);
    }

    /* ── Green checkboxes ── */
    section[data-testid="stSidebar"] [data-baseweb="checkbox"] > span:first-child {
        background-color: #10b981 !important;
        border-color: #10b981 !important;
    }

    /* ── Sidebar text inputs ── */
    section[data-testid="stSidebar"] [data-baseweb="input"] {
        background-color: #1e293b !important;
        border-color: rgba(148, 163, 184, 0.3) !important;
    }
    section[data-testid="stSidebar"] [data-baseweb="input"] input {
        color: #e2e8f0 !important;
        -webkit-text-fill-color: #e2e8f0 !important;
    }

    /* ── Chat input ── */
    .stChatInput > div {
        border: 1px solid rgba(99, 102, 241, 0.4) !important;
        border-radius: 12px !important;
        transition: border-color 0.3s ease;
    }
    .stChatInput > div:focus-within {
        border-color: #818cf8 !important;
        box-shadow: 0 0 0 3px rgba(129, 140, 248, 0.15) !important;
    }

    /* ── Header badge ── */
    .header-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        color: white;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }

    /* ── Tool call info box ── */
    .tool-call-box {
        background: rgba(99, 102, 241, 0.06);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin-bottom: 12px;
        font-size: 0.85rem;
    }
    .tool-call-box code {
        background: rgba(99, 102, 241, 0.12);
        padding: 2px 6px;
        border-radius: 4px;
        font-size: 0.8rem;
    }

    /* ── Source badges ── */
    .source-badge {
        display: inline-flex;
        align-items: center;
        gap: 4px;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        color: white;
        margin-right: 6px;
        margin-bottom: 6px;
    }
    .source-badges-row {
        display: flex;
        flex-wrap: wrap;
        align-items: center;
        gap: 4px;
        margin-bottom: 12px;
    }

    /* ── Verification badge ── */
    .verification-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        color: white;
        margin-bottom: 12px;
        letter-spacing: 0.02em;
    }
    .verification-badge .score {
        font-size: 0.85rem;
        font-weight: 700;
    }

    /* ── Verification details ── */
    .claim-row {
        padding: 10px 14px;
        margin-bottom: 8px;
        border-radius: 8px;
        border-left: 3px solid;
        font-size: 0.84rem;
        line-height: 1.5;
    }
    .claim-text {
        font-style: italic;
        color: #475569;
        margin-bottom: 4px;
    }
    .claim-explanation {
        color: #64748b;
        font-size: 0.8rem;
    }
    .claim-sources {
        color: #94a3b8;
        font-size: 0.75rem;
        margin-top: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Helper: render source badges
# ──────────────────────────────────────────────
def render_source_badges(source_counts: dict[str, int], unique_count: int = 0) -> str:
    """Build HTML for colored source badges showing result counts."""
    if not source_counts:
        return ""
    total_raw = sum(source_counts.values())
    badges = []
    for source, count in source_counts.items():
        color = SOURCE_COLORS.get(source, "#64748b")
        badges.append(
            f'<span class="source-badge" style="background:{color};">'
            f"{source} · {count}</span>"
        )
    if unique_count > 0 and unique_count < total_raw:
        badges.append(
            f'<span class="source-badge" style="background:#475569;">'
            f"Unique Sources · {unique_count}</span>"
        )
    return f'<div class="source-badges-row">{"".join(badges)}</div>'


def render_tool_calls(tool_queries: list[str]) -> str:
    """Build HTML for tool-call info boxes."""
    if not tool_queries:
        return ""
    parts = []
    for q in tool_queries:
        parts.append(
            f'<div class="tool-call-box">🔧 <strong>Tool called</strong> — '
            f"query: <em>{q}</em></div>"
        )
    return "".join(parts)


def render_verification_badge(report: VerificationReport) -> str:
    """Build HTML for the verification confidence badge."""
    score = report.confidence_score
    color = report.badge_color
    emoji = report.badge_emoji
    total = len(report.results)
    return (
        f'<div class="verification-badge" style="background:{color};">'
        f'{emoji} Citation Verification: <span class="score">{score}%</span> '
        f'({total} claims checked)</div>'
    )


def render_verification_details(report: VerificationReport) -> str:
    """Build HTML for the expandable verification details."""
    if not report.results:
        return ""

    rows = []
    for result in report.results:
        cfg = _STATUS_CONFIG[result.status]
        icon = cfg["icon"]
        label = cfg["label"]
        bg = cfg["bg"]
        border = cfg["border"]

        # Truncate claim for display
        claim_display = result.claim
        if len(claim_display) > 250:
            claim_display = claim_display[:247] + "..."

        sources_str = ", ".join(
            f"[{cid}] {title[:60]}{'...' if len(title) > 60 else ''}"
            for cid, title in zip(result.citation_ids, result.source_titles)
        )

        rows.append(
            f'<div class="claim-row" style="background:{bg}; border-left-color:{border};">'
            f'<div><strong>{icon} {label}</strong></div>'
            f'<div class="claim-text">"{claim_display}"</div>'
            f'<div class="claim-explanation">{result.explanation}</div>'
            f'<div class="claim-sources">Sources: {sources_str}</div>'
            f'</div>'
        )

    return "\n".join(rows)


def render_verification_summary(report: VerificationReport) -> str:
    """Build a one-line summary of verification counts."""
    parts = []
    if report.supported_count:
        parts.append(f"✅ {report.supported_count} supported")
    if report.partial_count:
        parts.append(f"⚠️ {report.partial_count} partial")
    if report.unsupported_count:
        parts.append(f"❌ {report.unsupported_count} unsupported")
    return " &nbsp;·&nbsp; ".join(parts)


# ──────────────────────────────────────────────
# Background Verification Runner
# ──────────────────────────────────────────────
def _run_verification_thread(
    synthesis_text: str,
    reference_map: dict[int, dict],
    result_container: dict,
):
    """Run verification in a background thread and store results."""
    try:
        report = verify_all(synthesis_text, reference_map)
        result_container["report"] = report
    except Exception as e:
        result_container["error"] = str(e)
    result_container["done"] = True


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧬 Clinical Research Agent")
    st.markdown(
        '<span class="header-badge">LOCAL · PRIVATE · AGENTIC</span>',
        unsafe_allow_html=True,
    )
    st.markdown("---")

    st.markdown(
        "This tool runs **100 % locally** on your machine using "
        f"**{MODEL_NAME}** via [Ollama](https://ollama.com). "
        "Your queries and data never leave your network — ideal for "
        "**IP-sensitive pharmaceutical research**."
    )

    st.markdown("---")

    # ── Data Sources ─────────────────────────────
    st.markdown("#### 📚 Data Sources")
    st.caption("Toggle which databases the agent can query.")

    src_europe_pmc = st.checkbox("Europe PMC", value=True, key="src_epmc")
    src_pubmed = st.checkbox("PubMed", value=True, key="src_pubmed")
    src_clinical_trials = st.checkbox("ClinicalTrials.gov", value=True, key="src_ct")

    # Build the enabled set
    enabled_sources: list[str] = []
    if src_europe_pmc:
        enabled_sources.append("Europe PMC")
    if src_pubmed:
        enabled_sources.append("PubMed")
    if src_clinical_trials:
        enabled_sources.append("ClinicalTrials.gov")

    st.markdown("---")

    # ── Advanced Settings ────────────────────────
    with st.expander("⚙️ Advanced Settings"):
        pubmed_tool_name = st.text_input(
            "PubMed tool name (NCBI)",
            value=DEFAULT_TOOL_NAME,
            help="Identifies your application to NCBI. Required for E-utilities.",
        )
        pubmed_email = st.text_input(
            "PubMed email (NCBI)",
            value=DEFAULT_EMAIL,
            help="Contact email sent to NCBI with each request.",
        )

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        "1. You ask a question about a drug-target interaction.\n"
        "2. Gemma 4 decides **which sources** to query using function calling.\n"
        "3. The agent fetches real-time results from **up to 3 databases**.\n"
        "4. Gemma synthesises a cited summary with references.\n"
        "5. A **verification agent** checks each claim against its source."
    )

    st.markdown("---")
    st.markdown("#### Example queries")
    examples = [
        "What are the mechanisms of resistance to Imatinib in BCR-ABL+ CML?",
        "Find recent studies on PARP inhibitor Olaparib resistance in BRCA cancers.",
        "Summarise clinical evidence for CDK4/6 inhibitors in HR+ breast cancer.",
        "Are there any ongoing clinical trials for CAR-T therapy in glioblastoma?",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

    st.markdown("---")
    st.caption("Powered by Gemma 4 · Europe PMC · PubMed · ClinicalTrials.gov · Streamlit")


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ──────────────────────────────────────────────
# Rendering Helpers
# ──────────────────────────────────────────────
def _render_entry(entry: dict):
    """Render all parts of a chat history entry."""
    # Tool call boxes
    if entry.get("tool_queries"):
        st.markdown(render_tool_calls(entry["tool_queries"]), unsafe_allow_html=True)
    # Source badges
    if entry.get("source_counts"):
        unique_len = len(entry.get("reference_map", {}))
        st.markdown(render_source_badges(entry["source_counts"], unique_len), unsafe_allow_html=True)

    # Verification badge (if verification is done)
    report = entry.get("verification_report")
    if report is not None:
        st.markdown(render_verification_badge(report), unsafe_allow_html=True)

    # Main content
    st.markdown(entry["content"])

    # Verification details expander (if verification is done)
    if report is not None and report.results:
        summary = render_verification_summary(report)
        with st.expander(f"🔍 Verification Details — {summary}", expanded=False):
            st.markdown(
                render_verification_details(report),
                unsafe_allow_html=True,
            )


# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.markdown("## 🔬 Gemma 4 Clinical Research Agent")
st.caption("Ask any question about drug-target interactions and get a cited literature summary.")

# Render existing history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"], avatar="🧑‍🔬" if entry["role"] == "user" else "🧬"):
        if entry["role"] == "assistant":
            _render_entry(entry)
        else:
            st.markdown(entry["content"])

# Chat input
user_input = st.chat_input("e.g. What are the mechanisms of Imatinib resistance in BCR-ABL+ CML?")

if user_input:
    # Append & render user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(user_input)

    # Build spinner text from enabled sources
    source_names = ", ".join(enabled_sources) if enabled_sources else "no sources"
    spinner_text = f"Agent is querying {source_names}…"

    # Run agent with spinner
    with st.chat_message("assistant", avatar="🧬"):
        # ── Phase 1: Retrieval + Synthesis ───────
        with st.spinner(spinner_text):
            try:
                answer, source_counts, tool_queries, reference_map = route_and_retrieve(
                    user_query=user_input,
                    enabled_sources=enabled_sources,
                    pubmed_tool_name=pubmed_tool_name,
                    pubmed_email=pubmed_email,
                )
            except Exception as e:
                answer = (
                    f"⚠️ **Error** — could not complete the request.\n\n"
                    f"```\n{e}\n```\n\n"
                    f"Make sure Ollama is running and `{MODEL_NAME}` is pulled."
                )
                source_counts = {}
                tool_queries = []
                reference_map = {}

        # Display tool calls
        if tool_queries:
            st.markdown(render_tool_calls(tool_queries), unsafe_allow_html=True)

        # Display source badges
        if source_counts:
            st.markdown(render_source_badges(source_counts, len(reference_map)), unsafe_allow_html=True)

        # ── Phase 2: Verification ────────────────
        verification_report = None
        if reference_map:
            # Show the unverified response and a progress bar while verifying
            verification_placeholder = st.empty()
            verification_placeholder.markdown(
                '<div class="verification-badge" style="background:#64748b;">'
                '🔍 Verifying citations…</div>',
                unsafe_allow_html=True,
            )

            # Show the response text immediately (user can read while verification runs)
            st.markdown(answer)

            # Run verification with progress tracking
            progress_bar = st.progress(0, text="Verifying claims against sources…")
            try:
                def _update_progress(completed: int, total: int):
                    progress_bar.progress(
                        completed / total,
                        text=f"Verified {completed}/{total} claims…",
                    )

                verification_report = verify_all(
                    synthesis_text=answer,
                    reference_map=reference_map,
                    progress_callback=_update_progress,
                )

                # Replace the placeholder with the actual badge
                progress_bar.empty()
                verification_placeholder.markdown(
                    render_verification_badge(verification_report),
                    unsafe_allow_html=True,
                )

                # Show verification details expander
                if verification_report.results:
                    summary = render_verification_summary(verification_report)
                    with st.expander(f"🔍 Verification Details — {summary}", expanded=False):
                        st.markdown(
                            render_verification_details(verification_report),
                            unsafe_allow_html=True,
                        )

            except Exception as e:
                progress_bar.empty()
                verification_placeholder.markdown(
                    '<div class="verification-badge" style="background:#64748b;">'
                    f'⚠️ Verification failed: {e}</div>',
                    unsafe_allow_html=True,
                )
        else:
            # No reference map — just show the response
            st.markdown(answer)

    # Append assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "source_counts": source_counts,
        "tool_queries": tool_queries,
        "reference_map": reference_map,
        "verification_report": verification_report,
    })
