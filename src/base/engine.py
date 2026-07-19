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
from typing import Any, TypedDict

import faiss
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_NAME = "CShorten/ML-ArXiv-Papers"
DATASET_SPLIT = "train"
DATASET_COLUMNS = ("title", "abstract")
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


class SearchResponse(TypedDict):
    """Structured retrieval response suitable for a LangChain tool."""

    query: str
    k: int
    results: list[PaperSearchResult]


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

    The returned frame contains ``title``, ``abstract``, and ``combined_text``.
    ``combined_text`` preserves the existing retrieval input format used when
    building FAISS embeddings.
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
        df = df.loc[:, list(DATASET_COLUMNS)].dropna().reset_index(drop=True)
        df = df.iloc[:max_papers].reset_index(drop=True)
        df["combined_text"] = df["title"].astype(str) + ". " + df["abstract"].astype(str)
    except Exception as exc:
        logger.exception("Dataset loading failed")
        raise EngineError("Failed to load research paper dataset") from exc

    _dataframe = df
    logger.info("Loaded %d papers", len(_dataframe))
    return _dataframe


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

        row = dataframe.iloc[row_index]
        results.append(
            {
                "rank": len(results) + 1,
                "dataset_index": row_index,
                "title": str(row["title"]),
                "abstract": str(row["abstract"]),
                "similarity_score": float(score),
            }
        )

    return results


def search_papers(query: str, k: int = 5) -> SearchResponse:
    """Search papers and return a structured response for downstream tools."""
    return {
        "query": query.strip() if query else "",
        "k": k,
        "results": semantic_search(query=query, k=k),
    }


def process_query(query: str, k: int = 5) -> SearchResponse:
    """Backward-compatible alias for retrieval-only query processing.

    Older code used this name for search plus summarization. The engine now
    returns retrieval results only; summarization should be handled by the
    agent layer with ChatGroq.
    """
    return search_papers(query=query, k=k)
