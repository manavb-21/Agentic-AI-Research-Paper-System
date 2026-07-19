"""LLM initialization for the agentic research assistant layer."""

from __future__ import annotations

from langchain_groq import ChatGroq

from src.agentic.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE

_llm: ChatGroq | None = None


def get_llm() -> ChatGroq:
    """Return a lazily initialized reusable ChatGroq instance."""
    global _llm

    if _llm is None:
        _llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )

    return _llm
