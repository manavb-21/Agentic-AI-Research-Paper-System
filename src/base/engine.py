"""Retrieval engine for the ML-ArXiv research paper dataset.

This module intentionally contains only retrieval concerns:

- load the SentenceTransformer embedding model
- load and prepare the ML-ArXiv dataset
- load or build the FAISS index
- run semantic search
- return structured paper records

Agent reasoning, summarization, LangChain tool wrapping, API routes, and UI code
belong in higher-level modules.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, TypedDict, cast

import faiss
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "CShorten/ML-ArXiv-Papers"
DATASET_SPLIT = "train"
REQUIRED_DATASET_COLUMNS = ("title", "abstract")
MAX_PAPERS = 15_000

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
INDEX_PATH = PROJECT_ROOT / "data" / "index" / "faiss.index"


class EngineError(Exception):
    """Base exception for recoverable retrieval engine failures."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PaperSearchResult(TypedDict):
    """Single paper returned by semantic search."""

    rank: int
    dataset_index: int
    title: str
    abstract: str
    similarity_score: float
    authors: list[str] | None
    metadata: dict[str, Any]


class PaperMetadata(TypedDict):
    """Structured paper metadata independent of a search score."""

    dataset_index: int
    title: str
    abstract: str
    authors: list[str] | None
    metadata: dict[str, Any]


class SearchResponse(TypedDict):
    """Structured retrieval response suitable for a LangChain tool."""

    query: str
    k: int
    results: list[PaperSearchResult]
    metadata: dict[str, Any]


_embedding_model: SentenceTransformer | None = None
_dataframe: pd.DataFrame | None = None
_faiss_index: Any | None = None


def load_embedding_model() -> SentenceTransformer:
    """Load and cache the sentence-transformer used for query embeddings."""
    global _embedding_model

    if _embedding_model is None:
        logger.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
        try:
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        except Exception as exc:
            logger.exception("Embedding model loading failed")
            raise EngineError("Failed to load embedding model") from exc

    return _embedding_model


def load_models() -> SentenceTransformer:
    """Backward-compatible wrapper that loads only the embedding model."""
    return load_embedding_model()


def load_paper_dataset(max_papers: int = MAX_PAPERS) -> pd.DataFrame:
    """Load and cache the ML-ArXiv dataset prepared for semantic retrieval.

    The returned frame preserves all source dataset columns and adds
    ``combined_text``. ``combined_text`` preserves the existing retrieval input
    format used when building FAISS embeddings.
    """
    global _dataframe

    if _dataframe is not None:
        return _dataframe

    if max_papers <= 0:
        raise EngineError("max_papers must be greater than zero", status_code=400)

    logger.info(
        "Loading dataset %s split=%s with max_papers=%d",
        DATASET_NAME,
        DATASET_SPLIT,
        max_papers,
    )

    try:
        dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT)
        df = dataset.to_pandas()
        missing_columns = [
            column for column in REQUIRED_DATASET_COLUMNS if column not in df.columns
        ]
        if missing_columns:
            raise EngineError(
                f"Dataset is missing required columns: {', '.join(missing_columns)}"
            )

        df = df.dropna(subset=list(REQUIRED_DATASET_COLUMNS)).reset_index(drop=True)
        df = df.iloc[:max_papers].reset_index(drop=True)
        df["combined_text"] = df["title"].astype(str) + ". " + df["abstract"].astype(str)
    except Exception as exc:
        logger.exception("Dataset loading failed")
        raise EngineError("Failed to load research paper dataset") from exc

    _dataframe = df
    logger.info("Loaded %d papers", len(_dataframe))
    return _dataframe


def _is_missing_value(value: Any) -> bool:
    """Return whether a dataset value should be treated as missing."""
    if value is None:
        return True

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        return False

    if isinstance(missing, bool):
        return missing

    if hasattr(missing, "item"):
        try:
            return bool(missing.item())
        except (TypeError, ValueError):
            return False

    return False


def _to_json_safe(value: Any) -> Any:
    """Convert dataset values to JSON-friendly Python objects."""
    if _is_missing_value(value):
        return None

    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass

    if isinstance(value, dict):
        return {str(key): _to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_to_json_safe(item) for item in value]

    return value


def _normalize_authors(value: Any) -> list[str] | None:
    """Normalize an authors field from the source dataset when available."""
    if _is_missing_value(value):
        return None

    if isinstance(value, str):
        authors = [
            author.strip()
            for author in value.replace(" and ", ",").split(",")
            if author.strip()
        ]
        return authors or None

    if isinstance(value, (list, tuple, set)):
        authors = [str(author).strip() for author in value if str(author).strip()]
        return authors or None

    return [str(value).strip()] if str(value).strip() else None


def _extract_authors(metadata: Mapping[str, Any]) -> list[str] | None:
    """Extract authors from known metadata key variants when present."""
    for key in ("authors", "author", "creator", "creators"):
        if key in metadata:
            authors = _normalize_authors(metadata[key])
            if authors:
                return authors

    return None


def _paper_from_row(
    row: pd.Series,
    dataset_index: int,
    *,
    rank: int | None = None,
    similarity_score: float | None = None,
) -> PaperMetadata | PaperSearchResult:
    """Build a structured paper payload from a dataset row."""
    metadata = {
        str(column): _to_json_safe(value)
        for column, value in row.items()
        if column not in {"title", "abstract", "combined_text"}
    }
    authors = _extract_authors(metadata)

    paper: PaperMetadata = {
        "dataset_index": dataset_index,
        "title": str(row["title"]),
        "abstract": str(row["abstract"]),
        "authors": authors,
        "metadata": metadata,
    }

    if rank is None or similarity_score is None:
        return paper

    return {
        **paper,
        "rank": rank,
        "similarity_score": similarity_score,
    }


def get_paper_metadata(dataset_index: int) -> PaperMetadata:
    """Return structured paper data for a dataset index."""
    dataframe = load_paper_dataset()
    if dataset_index < 0 or dataset_index >= len(dataframe):
        raise EngineError(
            f"dataset_index must be between 0 and {len(dataframe) - 1}.",
            status_code=400,
        )

    paper = _paper_from_row(dataframe.iloc[dataset_index], dataset_index)
    return paper


def load_faiss_index(index_path: Path = INDEX_PATH) -> Any:
    """Load and cache the FAISS index, creating it if no index file exists.

    The index keeps the existing retrieval behavior: combined title/abstract
    embeddings are L2-normalized and stored in an inner-product FAISS index.
    """
    global _faiss_index

    if _faiss_index is not None:
        return _faiss_index

    dataframe = load_paper_dataset()

    try:
        index_path.parent.mkdir(parents=True, exist_ok=True)

        if index_path.exists():
            logger.info("Loading FAISS index from %s", index_path)
            _faiss_index = faiss.read_index(str(index_path))
            return _faiss_index

        logger.info("Creating FAISS index at %s", index_path)
        embedding_model = load_embedding_model()
        embeddings = embedding_model.encode(
            dataframe["combined_text"].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype("float32")

        faiss.normalize_L2(embeddings)

        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, str(index_path))

        _faiss_index = index
        return _faiss_index
    except Exception as exc:
        logger.exception("FAISS index loading failed")
        raise EngineError("Failed to load or create FAISS index") from exc


def initialize_engine() -> tuple[SentenceTransformer, pd.DataFrame, Any]:
    """Initialize all retrieval resources and return them.

    This is optional because public search functions initialize lazily, but it
    is useful for application startup hooks that want failures to occur early.
    """
    embedding_model = load_embedding_model()
    dataframe = load_paper_dataset()
    index = load_faiss_index()
    return embedding_model, dataframe, index


def initialize_system() -> tuple[pd.DataFrame, Any]:
    """Backward-compatible wrapper that initializes retrieval resources only."""
    _, dataframe, index = initialize_engine()
    return dataframe, index


def semantic_search(query: str, k: int = 5) -> list[PaperSearchResult]:
    """Return the top-k semantically similar papers for a query."""
    normalized_query = query.strip() if query else ""
    if not normalized_query:
        raise EngineError("Query parameter cannot be empty", status_code=400)
    if k <= 0:
        raise EngineError("k must be greater than zero", status_code=400)

    embedding_model = load_embedding_model()
    dataframe = load_paper_dataset()
    index = load_faiss_index()

    if index.ntotal == 0:
        raise EngineError("FAISS index is empty")

    search_k = min(k, index.ntotal)

    try:
        query_embedding = embedding_model.encode(
            [normalized_query],
            convert_to_numpy=True,
        ).astype("float32")
        faiss.normalize_L2(query_embedding)

        scores, indices = index.search(query_embedding, search_k)
    except Exception as exc:
        logger.exception("Semantic search failed")
        raise EngineError("Failed to search research papers") from exc

    results: list[PaperSearchResult] = []
    for score, dataset_index in zip(scores[0], indices[0]):
        row_index = int(dataset_index)
        if row_index < 0:
            continue
        if row_index >= len(dataframe):
            logger.warning(
                "FAISS result index %d is outside dataset size %d",
                row_index,
                len(dataframe),
            )
            continue

        paper = _paper_from_row(
            dataframe.iloc[row_index],
            row_index,
            rank=len(results) + 1,
            similarity_score=float(score),
        )
        results.append(cast(PaperSearchResult, paper))

    return results


def search_papers(query: str, k: int = 5) -> SearchResponse:
    """Search papers and return a structured response for downstream tools."""
    return {
        "query": query.strip() if query else "",
        "k": k,
        "results": semantic_search(query=query, k=k),
        "metadata": {
            "dataset": DATASET_NAME,
            "split": DATASET_SPLIT,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "index_path": str(INDEX_PATH),
        },
    }


def process_query(query: str, k: int = 5) -> SearchResponse:
    """Backward-compatible alias for retrieval-only query processing.

    Older code used this name for search plus summarization. The engine now
    returns retrieval results only; summarization should be handled by the
    agent layer with ChatGroq.
    """
    return search_papers(query=query, k=k)
