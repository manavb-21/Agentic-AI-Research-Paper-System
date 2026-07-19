"""Streamlit frontend for the research-paper agent."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal, TypedDict

import streamlit as st

from src.agentic.workflow import process_user_query

APP_TITLE = "Agentic AI Research Paper Intelligence System"
APP_DESCRIPTION = (
    "A portfolio-ready research assistant that searches machine learning "
    "papers with semantic retrieval, then uses an agentic LLM workflow to "
    "produce grounded research answers."
)
TECHNOLOGIES = [
    "LangChain",
    "ChatGroq",
    "FAISS",
    "SentenceTransformers",
    "Streamlit",
]
SUGGESTED_QUESTIONS = [
    "Explain Vision Transformers",
    "Recent diffusion model papers",
    "Papers about RLHF",
    "Graph Neural Networks",
    "Explain LoRA",
]
EMBEDDING_MODEL_DISPLAY = "all-MiniLM-L6-v2"
FAISS_INDEX_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "index" / "faiss.index"
)
SUGGESTIONS_PER_ROW = 3


class ChatMessage(TypedDict):
    """Message stored in Streamlit session state."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: str


def _inject_theme() -> None:
    """Apply the custom Streamlit theme for the premium Agentic AI portfolio UI."""
    st.markdown(
        """
        <style>
            /* Premium Dark Glassmorphism Variables */
            :root {
                --bg-primary: #0b0f19;
                --bg-secondary: #111827;
                --surface: rgba(17, 24, 39, 0.75);
                --surface-card: #1f2937;
                --text-main: #f8fafc;
                --text-muted: #94a3b8;
                --accent: #3b82f6;
                --accent-glow: rgba(59, 130, 246, 0.15);
                --teal: #14b8a6;
                --teal-glow: rgba(20, 184, 166, 0.15);
                --radius: 14px;
                --border: rgba(255, 255, 255, 0.08);
            }

            /* Global Background & Text */
            .stApp, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
                background-color: var(--bg-primary) !important;
                background: radial-gradient(circle at 85% 15%, rgba(59, 130, 246, 0.12), transparent 40%),
                            radial-gradient(circle at 15% 80%, rgba(20, 184, 166, 0.08), transparent 40%),
                            var(--bg-primary) !important;
                color: var(--text-main) !important;
            }

            .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 5rem;
            }

            h1, h2, h3, h4, h5, p, span, li, label {
                color: var(--text-main) !important;
            }

            /* Sidebar Override */
            section[data-testid="stSidebar"], 
            div[data-testid="stSidebarContent"] {
                background-color: var(--bg-secondary) !important;
                background: linear-gradient(180deg, #111827, #0b0f19) !important;
                border-right: 1px solid var(--border) !important;
            }
            
            section[data-testid="stSidebar"] * {
                color: var(--text-main) !important;
            }

            /* Hero Section & Header Cards */
            .hero {
                padding: 2rem 2.2rem;
                border: 1px solid var(--border);
                border-radius: 20px;
                background: var(--surface);
                backdrop-filter: blur(16px);
                box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
                margin-bottom: 1.5rem;
                position: relative;
                overflow: hidden;
            }
            
            .hero::before {
                content: "";
                position: absolute;
                top: 0; right: 0; width: 150px; height: 150px;
                background: radial-gradient(circle, var(--accent-glow), transparent 70%);
                pointer-events: none;
            }

            .hero-title {
                margin: 0 0 0.75rem 0;
                font-size: 2.6rem;
                line-height: 1.2;
                font-weight: 800;
                background: linear-gradient(to right, #ffffff, #94a3b8);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
            }

            .hero-copy {
                margin: 0;
                color: var(--text-muted) !important;
                font-size: 1.1rem;
                line-height: 1.6;
                max-width: 800px;
            }

            /* Technology Pills */
            .tech-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.65rem;
                margin-top: 1.2rem;
            }

            .tech-pill {
                border: 1px solid rgba(59, 130, 246, 0.3);
                border-radius: 999px;
                background: rgba(59, 130, 246, 0.08);
                color: #60a5fa !important;
                padding: 0.35rem 0.85rem;
                font-size: 0.85rem;
                font-weight: 600;
                letter-spacing: 0.02em;
            }

            /* Section Headers */
            .section-card {
                padding: 0.5rem 0 1rem 0;
            }

            .section-title {
                margin: 0;
                color: var(--text-main);
                font-size: 1.15rem;
                font-weight: 700;
                letter-spacing: 0.03em;
                text-transform: uppercase;
            }

            /* Chat Flow Styling */
            div[data-testid="stChatMessage"] {
                background: transparent !important;
                padding: 0.75rem 0;
                border: none !important;
            }

            .message-meta {
                color: var(--text-muted) !important;
                font-size: 0.8rem;
                font-weight: 600;
                margin-bottom: 0.5rem;
                letter-spacing: 0.03em;
            }

            /* Chat Cards */
            .user-card {
                border: 1px solid rgba(20, 184, 166, 0.25);
                border-radius: var(--radius);
                background: rgba(20, 184, 166, 0.05);
                padding: 1.2rem 1.4rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }

            .assistant-card {
                border: 1px solid var(--border);
                border-radius: var(--radius);
                background: var(--surface-card);
                padding: 1.2rem 1.4rem;
                box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            }

            /* Status Grid Configuration */
            .status-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 10px;
                margin-top: 5px;
            }

            .status-item {
                background: rgba(17, 24, 39, 0.5);
                border: 1px solid var(--border);
                border-radius: 12px;
                padding: 12px;
                transition: transform 0.2s ease, border-color 0.2s ease;
            }
            
            .status-item:hover {
                border-color: rgba(59, 130, 246, 0.4);
                transform: translateY(-2px);
            }

            .status-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                color: var(--text-muted) !important;
                margin-bottom: 4px;
            }

            .status-value {
                font-size: 0.95rem;
                font-weight: 700;
                color: var(--accent) !important;
            }

            /* Streamlit Input Customization */
            div[data-testid="stChatInput"] {
                padding-bottom: 2rem;
            }
            
            div[data-testid="stChatInput"] textarea {
                background-color: var(--surface-card) !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border) !important;
                border-radius: 12px !important;
            }

            div[data-testid="stChatInput"] textarea:focus {
                border-color: var(--accent) !important;
                box-shadow: 0 0 0 2px var(--accent-glow) !important;
            }

            /* Buttons Overrides */
            button[kind="secondary"], 
            button[data-testid="baseButton-secondary"],
            .stButton > button {
                background: var(--surface-card) !important;
                color: var(--text-main) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
                font-weight: 600 !important;
                transition: all 0.2s ease !important;
            }

            button[kind="secondary"]:hover,
            .stButton > button:hover {
                background: linear-gradient(135deg, #3b82f6, #1d4ed8) !important;
                border-color: transparent !important;
                transform: translateY(-1px) !important;
                box-shadow: 0 6px 16px rgba(59, 130, 246, 0.4) !important;
            }
            
            /* Metric Overrides */
            [data-testid="stMetricValue"] {
                color: var(--text-main) !important;
                font-weight: 700 !important;
            }
            
            [data-testid="stMetricLabel"] {
                color: var(--text-muted) !important;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            /* Expander Overrides */
            div[data-testid="stExpander"] {
                background-color: rgba(17, 24, 39, 0.4) !important;
                border: 1px solid var(--border) !important;
                border-radius: 10px !important;
            }
            
            div[data-testid="stExpander"] summary {
                color: var(--text-main) !important;
            }

            /* Footer Stylings */
            .footer {
                border-top: 1px solid var(--border);
                color: var(--text-muted) !important;
                margin-top: 3rem;
                padding-top: 1.5rem;
                text-align: center;
                font-size: 0.85rem;
                letter-spacing: 0.02em;
            }
            
            /* Status / Spinner Overrides */
            div[data-testid="stStatusWidget"], details {
                background-color: var(--surface-card) !important;
                border: 1px solid var(--border) !important;
                color: var(--text-main) !important;
                border-radius: 12px !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _now_label() -> str:
    """Return a compact timestamp for chat messages."""
    return datetime.now().strftime("%I:%M %p")


def _initialize_session_state() -> None:
    """Create chat history storage for the current Streamlit session."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pending_query" not in st.session_state:
        st.session_state.pending_query = None


def _get_config_display() -> tuple[str, str]:
    """Read display-only backend configuration values."""
    try:
        from src.agentic.config import DEFAULT_TOP_K, MODEL_NAME

        return MODEL_NAME, str(DEFAULT_TOP_K)
    except Exception:
        return "Unavailable", "Unavailable"


def _message_counts(messages: list[ChatMessage]) -> tuple[int, int]:
    """Count user and assistant messages."""
    user_count = sum(1 for message in messages if message["role"] == "user")
    assistant_count = sum(1 for message in messages if message["role"] == "assistant")
    return user_count, assistant_count


def _conversation_text(messages: list[ChatMessage]) -> str:
    """Format the current conversation as plain text for export."""
    lines = [APP_TITLE, ""]
    for message in messages:
        role = message["role"].title()
        timestamp = message.get("timestamp", "")
        lines.append(f"[{timestamp}] {role}")
        lines.append(message["content"])
        lines.append("")
    return "\n".join(lines).strip() + "\n"


def _render_sidebar(messages: list[ChatMessage]) -> None:
    """Render project metadata, controls, and architecture notes."""
    model_name, top_k = _get_config_display()
    user_count, assistant_count = _message_counts(messages)
    faiss_status = "Ready" if FAISS_INDEX_PATH.exists() else "Index not found"

    with st.sidebar:
        st.markdown("### Project Information")
        st.write(
            "A modular agentic research assistant with a Streamlit presentation "
            "layer over a LangChain workflow and FAISS retrieval backend."
        )
        st.divider()

        st.markdown("### System Status")
        st.markdown(
            f"""
            <div class="status-grid">
                <div class="status-item">
                    <div class="status-label">Current LLM</div>
                    <div class="status-value">{model_name}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Embedding Model</div>
                    <div class="status-value">{EMBEDDING_MODEL_DISPLAY}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">FAISS Status</div>
                    <div class="status-value">{faiss_status}</div>
                </div>
                <div class="status-item">
                    <div class="status-label">Top-K Retrieval</div>
                    <div class="status-value">{top_k}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()
        st.markdown("### Conversation Controls")
        clear_col, export_col = st.columns(2)
        with clear_col:
            if st.button("Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.pending_query = None
                st.rerun()
        with export_col:
            st.download_button(
                "Export .txt",
                data=_conversation_text(messages),
                file_name="research_assistant_conversation.txt",
                mime="text/plain",
                use_container_width=True,
                disabled=not messages,
            )

        st.divider()
        st.markdown("### Chat Statistics")
        c1, c2 = st.columns(2)
        c1.metric("User Messages", user_count)
        c2.metric("AI Messages", assistant_count)

        st.divider()
        st.markdown("### Project Architecture")
        st.markdown(
            """
            ```mermaid
            User Input
               │
               ▼
            LangChain Workflow
               │
               ▼
            Agentic Reasoner
               │
               ▼
            FAISS Vector Store
            ```
            """
        )


def _render_homepage() -> None:
    """Render the welcome section shown before the first query."""
    tech_markup = "".join(
        f'<span class="tech-pill">{technology}</span>' for technology in TECHNOLOGIES
    )
    st.markdown(
        f"""
        <div class="hero">
            <h1 class="hero-title">{APP_TITLE}</h1>
            <p class="hero-copy">{APP_DESCRIPTION}</p>
            <div class="tech-row">{tech_markup}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="section-card">
            <h3 class="section-title">Suggested Research Topics</h3>
        </div>
        """,
        unsafe_allow_html=True,
    )
    for row_start in range(0, len(SUGGESTED_QUESTIONS), SUGGESTIONS_PER_ROW):
        row_questions = SUGGESTED_QUESTIONS[
            row_start : row_start + SUGGESTIONS_PER_ROW
        ]
        columns = st.columns(SUGGESTIONS_PER_ROW)
        for column, question in zip(columns, row_questions):
            with column:
                if st.button(question, use_container_width=True):
                    st.session_state.pending_query = question
                    st.rerun()


def _render_header() -> None:
    """Render compact page header once a conversation has started."""
    st.markdown(
        f"""
        <div class="hero" style="padding: 1.5rem 1.8rem; margin-bottom: 1rem;">
            <h1 class="hero-title" style="font-size: 1.8rem; margin-bottom: 0.25rem;">{APP_TITLE}</h1>
            <p class="hero-copy" style="font-size: 0.95rem;">
                Ask a research question and the agent will search relevant papers,
                inspect retrieved metadata, and produce a grounded answer.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_message(message: ChatMessage) -> None:
    """Display a single chat message with portfolio styling."""
    role = message["role"]
    timestamp = message.get("timestamp", "")
    card_class = "user-card" if role == "user" else "assistant-card"
    label = "You" if role == "user" else "Research Assistant"

    with st.chat_message(role):
        st.markdown(
            f'<div class="message-meta">{label} • {timestamp}</div>',
            unsafe_allow_html=True,
        )
        st.markdown(f'<div class="{card_class}">', unsafe_allow_html=True)
        
        if role == "assistant":
            with st.expander("Final AI Answer", expanded=True):
                st.markdown(message["content"])
            with st.expander("Retrieved Information", expanded=False):
                st.caption(
                    "The current backend keeps retrieval details inside the "
                    "LangChain agent execution. This section is reserved for "
                    "future structured retrieval traces."
                )
        else:
            st.markdown(message["content"])
            
        st.markdown("</div>", unsafe_allow_html=True)


def _render_chat_history(messages: list[ChatMessage]) -> None:
    """Display previous chat messages."""
    for message in messages:
        if "timestamp" not in message:
            message["timestamp"] = _now_label()
        _render_message(message)


def _append_message(role: Literal["user", "assistant"], content: str) -> None:
    """Append a message to the active chat session."""
    st.session_state.messages.append(
        {"role": role, "content": content, "timestamp": _now_label()}
    )


def _resolve_submitted_query() -> str | None:
    """Return a query from a suggestion click or the chat input."""
    pending_query = st.session_state.get("pending_query")
    if pending_query:
        st.session_state.pending_query = None
        return str(pending_query)

    return st.chat_input("Ask about a machine learning research topic")


def _run_query(query: str) -> None:
    """Send a query through the workflow and render the assistant response."""
    _append_message("user", query)
    _render_message(st.session_state.messages[-1])

    with st.chat_message("assistant"):
        st.markdown(
            f'<div class="message-meta">Research Assistant • {_now_label()}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Searching papers and preparing a response..."):
            with st.status("Research workflow in progress", expanded=True) as status:
                status.write("Searching semantic embeddings...")
                status.write("Retrieving papers from FAISS...")
                status.write("Reasoning with Agent...")
                status.write("Generating final response...")
                try:
                    response = process_user_query(query)
                except Exception as exc:
                    response = (
                        "I could not complete that request. Please check your "
                        f"configuration and try again. Details: {exc}"
                    )
                    status.update(label="Request could not be completed", state="error")
                    st.error(response)
                else:
                    status.update(label="Research response ready", state="complete")
                    st.markdown('<div class="assistant-card">', unsafe_allow_html=True)
                    with st.expander("Final AI Answer", expanded=True):
                        st.markdown(response)
                    with st.expander("Retrieved Information", expanded=False):
                        st.caption(
                            "The current backend keeps retrieval details inside the "
                            "LangChain agent execution. This section is reserved for "
                            "future structured retrieval traces."
                        )
                    st.markdown("</div>", unsafe_allow_html=True)

    _append_message("assistant", response)


def _render_footer() -> None:
    """Render the app footer."""
    st.markdown(
        """
        <div class="footer">
            <strong>Agentic AI Research Platform</strong><br/>
            Built with LangChain • ChatGroq • FAISS • Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Streamlit chat interface."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
        initial_sidebar_state="expanded"
    )

    _initialize_session_state()
    _inject_theme()
    _render_sidebar(st.session_state.messages)

    if st.session_state.messages:
        _render_header()
    else:
        _render_homepage()

    _render_chat_history(st.session_state.messages)

    query = _resolve_submitted_query()
    if query:
        _run_query(query)

    _render_footer()


if __name__ == "__main__":
    main()