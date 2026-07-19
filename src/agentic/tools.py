"""LangChain tools for research-paper retrieval actions."""

from __future__ import annotations

from langchain_core.tools import BaseTool, tool

from src.agentic.config import DEFAULT_TOP_K


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
def fetch_paper_metadata(dataset_index: int) -> dict[str, object]:
    """Fetch title and abstract for a paper by its dataset index."""
    from src.base.engine import load_paper_dataset

    dataframe = load_paper_dataset()
    if dataset_index < 0 or dataset_index >= len(dataframe):
        raise ValueError(
            f"dataset_index must be between 0 and {len(dataframe) - 1}."
        )

    paper = dataframe.iloc[dataset_index]
    return {
        "dataset_index": dataset_index,
        "title": str(paper["title"]),
        "abstract": str(paper["abstract"]),
    }


RESEARCH_TOOLS: list[BaseTool] = [
    search_papers,
    fetch_paper_metadata,
]
