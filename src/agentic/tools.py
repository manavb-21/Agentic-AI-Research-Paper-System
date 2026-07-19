"""LangChain tool wrappers for retrieval and prompt preparation."""

from __future__ import annotations

from langchain_core.tools import tool

from src.agentic.config import DEFAULT_TOP_K
from src.agentic.prompts import (
    build_difficulty_estimation_prompt,
    build_paper_comparison_prompt,
    build_paper_summarization_prompt,
    build_research_gap_analysis_prompt,
    build_research_recommendation_prompt,
)


def _require_text(value: str, field_name: str) -> str:
    """Return stripped text or raise a clear validation error."""
    cleaned_value = value.strip() if value else ""
    if not cleaned_value:
        raise ValueError(f"{field_name} cannot be empty.")
    return cleaned_value


@tool
def search_papers(query: str, k: int = DEFAULT_TOP_K) -> dict[str, object]:
    """Search ML-ArXiv papers by semantic similarity and return raw results."""
    from src.base.engine import search_papers as engine_search_papers

    cleaned_query = _require_text(query, "query")
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    return engine_search_papers(query=cleaned_query, k=k)


@tool
def summarize_paper(title: str, abstract: str) -> str:
    """Build a summarization prompt for a single paper."""
    return build_paper_summarization_prompt(
        title=_require_text(title, "title"),
        abstract=_require_text(abstract, "abstract"),
    )


@tool
def compare_papers(
    title_a: str,
    abstract_a: str,
    title_b: str,
    abstract_b: str,
) -> str:
    """Build a comparison prompt for two papers."""
    return build_paper_comparison_prompt(
        title_a=_require_text(title_a, "title_a"),
        abstract_a=_require_text(abstract_a, "abstract_a"),
        title_b=_require_text(title_b, "title_b"),
        abstract_b=_require_text(abstract_b, "abstract_b"),
    )


@tool
def recommend_papers(query: str, papers: str) -> str:
    """Build a recommendation prompt from a query and retrieved papers."""
    return build_research_recommendation_prompt(
        query=_require_text(query, "query"),
        papers=_require_text(papers, "papers"),
    )


@tool
def estimate_difficulty(title: str, abstract: str) -> str:
    """Build a difficulty-estimation prompt for a single paper."""
    return build_difficulty_estimation_prompt(
        title=_require_text(title, "title"),
        abstract=_require_text(abstract, "abstract"),
    )


@tool
def analyze_research_gap(topic: str, papers: str) -> str:
    """Build a research-gap analysis prompt from a topic and papers."""
    return build_research_gap_analysis_prompt(
        topic=_require_text(topic, "topic"),
        papers=_require_text(papers, "papers"),
    )
