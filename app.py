import streamlit as st
import ollama
import requests

# ──────────────────────────────────────────────
# Page Configuration
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Gemma 4 Clinical Research Agent",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────
MODEL_NAME = "gemma4:e2b"

SYSTEM_PROMPT = (
    "You are a clinical research assistant. When summarizing literature, "
    "you MUST use in-line citations (e.g., [1], [2]) and include a "
    "'References' section at the bottom containing the Title and DOI "
    "of the papers you used."
)

SEARCH_LITERATURE_TOOL = {
    "type": "function",
    "function": {
        "name": "search_literature",
        "description": (
            "Use this tool to search Europe PMC for scientific literature "
            "when asked about drug-target interactions. Returns a list of "
            "article titles, abstracts, and DOIs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The drug-target search query string (e.g., drug names, gene targets, interactions)."
                },
                "limit": {
                    "type": "integer",
                    "description": "The maximum number of literature results to return. Default is 5."
                }
            },
            "required": ["query"]
        }
    }
}

# ──────────────────────────────────────────────
# Tool Implementation
# ──────────────────────────────────────────────
def search_literature(query: str, limit: int = 5):
    """
    Calls the Europe PMC REST API to search for literature.
    Returns a list of dicts with title, abstract, and DOI.
    """
    url = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    params = {
        "query": query,
        "format": "json",
        "resultType": "core",
        "pageSize": limit
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("resultList", {}).get("result", []):
        results.append({
            "title": item.get("title", "No Title"),
            "abstract": item.get("abstractText", "No abstract available"),
            "doi": item.get("doi", "No DOI available")
        })
    return results


def format_results(literature_results):
    """Format literature results with numbered references for the model."""
    formatted = ""
    for idx, paper in enumerate(literature_results, 1):
        formatted += (
            f"[{idx}] Title: {paper['title']}\n"
            f"    DOI: {paper['doi']}\n"
            f"    Abstract: {paper['abstract']}\n\n"
        )
    return formatted


# ──────────────────────────────────────────────
# Agent Logic
# ──────────────────────────────────────────────
def run_agent(user_query: str):
    """
    Runs the full agent loop:
      1. Send user query + tool schema to Gemma
      2. If a tool call is triggered, execute search_literature locally
      3. Feed results back to Gemma for a cited summary
    Returns (final_text, tool_query) where tool_query is None if no tool was called.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]

    response = ollama.chat(
        model=MODEL_NAME,
        messages=messages,
        tools=[SEARCH_LITERATURE_TOOL]
    )

    message = response.get("message", {})

    if "tool_calls" in message and message["tool_calls"]:
        tool_call = message["tool_calls"][0]
        fn_name = tool_call["function"]["name"]
        args = tool_call["function"]["arguments"]

        if fn_name == "search_literature":
            query = args.get("query")
            limit = args.get("limit", 5)
            literature_results = search_literature(query=query, limit=limit)
            formatted = format_results(literature_results)

            messages.append(message)
            messages.append({
                "role": "tool",
                "content": f"Search results:\n\n{formatted}"
            })

            final_response = ollama.chat(
                model=MODEL_NAME,
                messages=messages,
                tools=[SEARCH_LITERATURE_TOOL]
            )
            return final_response["message"]["content"], query
    
    # No tool call — model answered directly
    return message.get("content", ""), None


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
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("# 🧬 Clinical Research Agent")
    st.markdown(
        '<span class="header-badge">LOCAL · PRIVATE · AGENTIC</span>',
        unsafe_allow_html=True
    )
    st.markdown("---")

    st.markdown(
        "This tool runs **100 % locally** on your machine using "
        f"**{MODEL_NAME}** via [Ollama](https://ollama.com). "
        "Your queries and data never leave your network — ideal for "
        "**IP-sensitive pharmaceutical research**."
    )

    st.markdown("---")
    st.markdown("#### How it works")
    st.markdown(
        "1. You ask a question about a drug-target interaction.\n"
        "2. Gemma 4 decides to call the **search_literature** tool.\n"
        "3. The tool fetches real-time results from **Europe PMC**.\n"
        "4. Gemma synthesises a cited summary with references."
    )

    st.markdown("---")
    st.markdown("#### Example queries")
    examples = [
        "What are the mechanisms of resistance to Imatinib in BCR-ABL+ CML?",
        "Find recent studies on PARP inhibitor Olaparib resistance in BRCA cancers.",
        "Summarise clinical evidence for CDK4/6 inhibitors in HR+ breast cancer.",
    ]
    for ex in examples:
        st.markdown(f"- *{ex}*")

    st.markdown("---")
    st.caption("Powered by Gemma 4 · Europe PMC · Streamlit")


# ──────────────────────────────────────────────
# Session State
# ──────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ──────────────────────────────────────────────
# Main Chat Area
# ──────────────────────────────────────────────
st.markdown("## 🔬 Gemma 4 Clinical Research Agent")
st.caption("Ask any question about drug-target interactions and get a cited literature summary.")

# Render existing history
for entry in st.session_state.chat_history:
    with st.chat_message(entry["role"], avatar="🧑‍🔬" if entry["role"] == "user" else "🧬"):
        if entry.get("tool_query"):
            st.markdown(
                f'<div class="tool-call-box">🔧 <strong>Tool called</strong> — '
                f'<code>search_literature</code> with query: <em>{entry["tool_query"]}</em></div>',
                unsafe_allow_html=True
            )
        st.markdown(entry["content"])

# Chat input
user_input = st.chat_input("e.g. What are the mechanisms of Imatinib resistance in BCR-ABL+ CML?")

if user_input:
    # Append & render user message
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍🔬"):
        st.markdown(user_input)

    # Run agent with spinner
    with st.chat_message("assistant", avatar="🧬"):
        with st.spinner("Agent is querying Europe PMC database..."):
            try:
                answer, tool_query = run_agent(user_input)
            except Exception as e:
                answer = (
                    f"⚠️ **Error** — could not complete the request.\n\n"
                    f"```\n{e}\n```\n\n"
                    f"Make sure Ollama is running and `{MODEL_NAME}` is pulled."
                )
                tool_query = None

        if tool_query:
            st.markdown(
                f'<div class="tool-call-box">🔧 <strong>Tool called</strong> — '
                f'<code>search_literature</code> with query: <em>{tool_query}</em></div>',
                unsafe_allow_html=True
            )

        st.markdown(answer)

    # Append assistant message to history
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "tool_query": tool_query
    })
