"""LLM initialization for the agentic research assistant layer."""

from __future__ import annotations

from typing import Any

from langchain.agents import create_agent
from langchain_groq import ChatGroq

from src.agentic.config import GROQ_API_KEY, MODEL_NAME, TEMPERATURE
from src.agentic.prompts import GENERAL_RESEARCH_ASSISTANT_SYSTEM_PROMPT
from src.agentic.tools import RESEARCH_TOOLS

_llm: ChatGroq | None = None
_research_agent: Any | None = None


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


def create_research_agent() -> Any:
    """Create and cache the LangChain research-paper agent."""
    global _research_agent

    if _research_agent is None:
        _research_agent = create_agent(
            model=get_llm(),
            tools=RESEARCH_TOOLS,
            system_prompt=GENERAL_RESEARCH_ASSISTANT_SYSTEM_PROMPT,
        )

    return _research_agent
