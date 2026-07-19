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
FAISS_INDEX_PATH = Path(__file__).resolve().parents[2] / "data" / "index" / "faiss.index"
SUGGESTIONS_PER_ROW = 3


class ChatMessage(TypedDict):
    """Message stored in Streamlit session state."""

    role: Literal["user", "assistant"]
    content: str
    timestamp: str


def _inject_theme() -> None:
    """Apply the custom Streamlit theme for the portfolio UI."""
    st.markdown(
        """
        <style>
            :root {
                --paper-bg: #f7f8fb;
                --paper-panel: rgba(255, 255, 255, 0.86);
                --paper-border: rgba(31, 41, 55, 0.12);
                --paper-text: #172033;
                --paper-muted: #5b6475;
                --paper-accent: #2563eb;
                --paper-accent-soft: #e8f0ff;
                --paper-success: #0f766e;
            }

            .stApp {
                background:
                    radial-gradient(circle at top left, rgba(37, 99, 235, 0.10), transparent 32rem),
                    linear-gradient(180deg, #fbfcff 0%, var(--paper-bg) 100%);
                color: var(--paper-text);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 2.2rem;
                padding-bottom: 3.5rem;
            }

            h1, h2, h3 {
                letter-spacing: 0;
            }

            .hero {
                padding: 1.5rem 1.6rem;
                border: 1px solid var(--paper-border);
                border-radius: 18px;
                background:
                    linear-gradient(135deg, rgba(255, 255, 255, 0.94), rgba(232, 240, 255, 0.62));
                box-shadow: 0 18px 40px rgba(15, 23, 42, 0.07);
                margin-bottom: 1.2rem;
            }

            .hero-title {
                margin: 0 0 0.45rem 0;
                font-size: 2.35rem;
                line-height: 1.12;
                font-weight: 760;
                color: var(--paper-text);
            }

            .hero-copy {
                margin: 0;
                color: var(--paper-muted);
                font-size: 1.03rem;
                line-height: 1.62;
                max-width: 760px;
            }

            .tech-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.55rem;
                margin-top: 1rem;
            }

            .tech-pill {
                border: 1px solid rgba(37, 99, 235, 0.16);
                border-radius: 999px;
                background: rgba(255, 255, 255, 0.74);
                color: #1f3f77;
                padding: 0.34rem 0.72rem;
                font-size: 0.86rem;
                font-weight: 650;
            }

            .section-card {
                border: 1px solid var(--paper-border);
                border-radius: 16px;
                background: var(--paper-panel);
                box-shadow: 0 12px 30px rgba(15, 23, 42, 0.055);
                padding: 1.05rem 1.1rem;
                margin: 1rem 0;
            }

            .section-title {
                margin: 0 0 0.65rem 0;
                color: var(--paper-text);
                font-size: 1rem;
                font-weight: 720;
            }

            .message-meta {
                color: var(--paper-muted);
                font-size: 0.78rem;
                margin-bottom: 0.35rem;
            }

            .assistant-card {
                border: 1px solid rgba(37, 99, 235, 0.13);
                border-radius: 16px;
                background: linear-gradient(180deg, #ffffff 0%, #f8fbff 100%);
                box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
                padding: 1rem 1.05rem;
                margin-top: 0.15rem;
            }

            .user-card {
                border: 1px solid rgba(15, 118, 110, 0.16);
                border-radius: 16px;
                background: linear-gradient(180deg, #ffffff 0%, #f6fffd 100%);
                padding: 0.85rem 1rem;
                margin-top: 0.15rem;
            }

            .status-grid {
                display: grid;
                grid-template-columns: repeat(2, minmax(0, 1fr));
                gap: 0.55rem;
            }

            .status-item {
                border: 1px solid var(--paper-border);
                border-radius: 12px;
                background: rgba(255, 255, 255, 0.72);
                padding: 0.65rem 0.7rem;
            }

            .status-label {
                color: var(--paper-muted);
                font-size: 0.74rem;
                margin-bottom: 0.2rem;
            }

            .status-value {
                color: var(--paper-text);
                font-size: 0.9rem;
                font-weight: 700;
            }

            .footer {
                border-top: 1px solid var(--paper-border);
                color: var(--paper-muted);
                margin-top: 2rem;
                padding-top: 1rem;
                text-align: center;
                font-size: 0.9rem;
            }

            div[data-testid="stSidebarContent"] {
                background: linear-gradient(180deg, rgba(255,255,255,0.96), rgba(247,248,251,0.96));
            }

            div[data-testid="stChatMessage"] {
                background: transparent;
                padding: 0.6rem 0;
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
        st.metric("User messages", user_count)
        st.metric("Assistant messages", assistant_count)

        st.divider()
        st.markdown("### Project Architecture")
        st.code(
            "User\n"
            "  |\n"
            "  v\n"
            "Workflow\n"
            "  |\n"
            "  v\n"
            "Research Agent\n"
            "  |\n"
            "  v\n"
            "Tools\n"
            "  |\n"
            "  v\n"
            "FAISS Retrieval"
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
            <div class="section-title">Suggested research questions</div>
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
        <div class="hero">
            <h1 class="hero-title">{APP_TITLE}</h1>
            <p class="hero-copy">
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
            f'<div class="message-meta">{label} - {timestamp}</div>',
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
            f'<div class="message-meta">Research Assistant - {_now_label()}</div>',
            unsafe_allow_html=True,
        )
        with st.spinner("Searching papers and preparing a response..."):
            with st.status("Research workflow in progress", expanded=True) as status:
                status.write("Searching semantic embeddings...")
                status.write("Retrieving papers from FAISS...")
                status.write("Reasoning with Llama...")
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
            <strong>Agentic AI Research Paper Intelligence System</strong><br/>
            Powered by LangChain · ChatGroq · FAISS · SentenceTransformers
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    """Render the Streamlit chat interface."""
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
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
