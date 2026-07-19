import logging
import os
import time
from typing import Any, TypedDict

import faiss
import pandas as pd
import torch
from datasets import load_dataset
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'
SUMMARIZER_MODEL_NAME = 'facebook/bart-large-cnn'
NER_MODEL_NAME = 'dslim/bert-base-NER'

INDEX_PATH = 'data/index/faiss.index'


class EngineError(Exception):
    """Base exception for recoverable engine failures."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class PaperResult(TypedDict):
    title: str
    abstract: str
    score: float
    keywords: list[tuple[str, float]]
    entities: list[tuple[str, str]]


class TopPaperResponse(TypedDict):
    title: str
    similarity_score: float
    keywords: list[tuple[str, float]]
    entities: list[tuple[str, str]]


class QueryResponse(TypedDict):
    query: str
    generative_summary: str
    top_papers: list[TopPaperResponse]


_dataframe: pd.DataFrame | None = None
_faiss_index: Any | None = None
_embedding_model: SentenceTransformer | None = None
_keyword_model: KeyBERT | None = None
_summarizer_tokenizer: Any | None = None
_summarizer_model: AutoModelForSeq2SeqLM | None = None
_ner_tokenizer: Any | None = None
_ner_model: AutoModelForTokenClassification | None = None


def load_models() -> None:
    """Load all ML models required by the research paper intelligence engine."""
    global _embedding_model
    global _keyword_model
    global _summarizer_tokenizer
    global _summarizer_model
    global _ner_tokenizer
    global _ner_model

    if all([
        _embedding_model,
        _keyword_model,
        _summarizer_tokenizer,
        _summarizer_model,
        _ner_tokenizer,
        _ner_model,
    ]):
        return

    logger.info('Loading models on device: %s', DEVICE)

    try:
        if _embedding_model is None:
            logger.info('Loading embedding model: %s', EMBEDDING_MODEL_NAME)
            _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME, device=DEVICE.type)

        if _keyword_model is None:
            logger.info('Initializing KeyBERT keyword model')
            _keyword_model = KeyBERT(model=_embedding_model)

        if _summarizer_tokenizer is None:
            logger.info('Loading summarizer tokenizer: %s', SUMMARIZER_MODEL_NAME)
            _summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZER_MODEL_NAME)

        if _summarizer_model is None:
            logger.info('Loading summarizer model: %s', SUMMARIZER_MODEL_NAME)
            _summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZER_MODEL_NAME).to(DEVICE)
            _summarizer_model.eval()

        if _ner_tokenizer is None:
            logger.info('Loading NER tokenizer: %s', NER_MODEL_NAME)
            _ner_tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME)

        if _ner_model is None:
            logger.info('Loading NER model: %s', NER_MODEL_NAME)
            _ner_model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_NAME).to(DEVICE)
            _ner_model.eval()

        logger.info('Model loading complete')
    except Exception as exc:
        logger.exception('Model loading failed')
        raise EngineError('Failed to load ML models') from exc


def initialize_system() -> tuple[pd.DataFrame, Any]:
    """Load the dataset, prepare paper text, and load or create the FAISS index."""
    global _dataframe, _faiss_index

    logger.info('Initializing research paper intelligence system')
    load_models()

    try:
        dataset = load_dataset('CShorten/ML-ArXiv-Papers', split='train')
        df = dataset.to_pandas()
        df = df[['title', 'abstract']].dropna().reset_index(drop=True)
        df = df.iloc[:15000].reset_index(drop=True)
        df['combined_text'] = df['title'].astype(str) + '. ' + df['abstract'].astype(str)
    except Exception as exc:
        logger.exception('Dataset loading failed')
        raise EngineError('Failed to load research paper dataset') from exc

    _dataframe = df

    try:
        os.makedirs(os.path.dirname(INDEX_PATH), exist_ok=True)

        if os.path.exists(INDEX_PATH):
            logger.info('Loading FAISS index from %s', INDEX_PATH)
            _faiss_index = faiss.read_index(INDEX_PATH)
        else:
            if _embedding_model is None:
                raise EngineError('Embedding model is not loaded')

            logger.info('Creating FAISS index at %s', INDEX_PATH)
            embeddings = _embedding_model.encode(
                df['combined_text'].tolist(),
                show_progress_bar=True,
                convert_to_numpy=True
            ).astype('float32')

            faiss.normalize_L2(embeddings)

            dimension = embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(embeddings)

            faiss.write_index(index, INDEX_PATH)
            _faiss_index = index
    except EngineError:
        raise
    except Exception as exc:
        logger.exception('FAISS index loading failed')
        raise EngineError('Failed to load or create FAISS index') from exc

    logger.info('System initialization complete with %d papers', len(_dataframe))
    return _dataframe, _faiss_index


def summarize_text(text: str, max_length: int = 180, min_length: int = 60) -> str:
    """Generate a BART summary for the provided text."""
    if not text or not text.strip():
        return ''

    load_models()

    if _summarizer_tokenizer is None or _summarizer_model is None:
        raise EngineError('Summarization model is not loaded')

    try:
        inputs = _summarizer_tokenizer(
            text,
            return_tensors='pt',
            max_length=1024,
            truncation=True,
        )
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            summary_ids = _summarizer_model.generate(
                **inputs,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
            )

        return _summarizer_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as exc:
        logger.exception('Summary generation failed')
        raise EngineError('Failed to generate summary') from exc


def extract_entities(text: str) -> list[tuple[str, str]]:
    """Extract named entities from text using the BERT NER model."""
    if not text or not text.strip():
        return []

    load_models()

    if _ner_tokenizer is None or _ner_model is None:
        raise EngineError('NER model is not loaded')

    try:
        inputs = _ner_tokenizer(
            text,
            return_tensors='pt',
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offset_mapping = inputs.pop('offset_mapping')[0].tolist()
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}

        with torch.no_grad():
            logits = _ner_model(**inputs).logits

        predictions = torch.argmax(logits, dim=-1)[0].cpu().tolist()
        id_to_label = _ner_model.config.id2label

        entities: list[dict[str, int | str]] = []
        active_entity: dict[str, int | str] | None = None

        for prediction, offsets in zip(predictions, offset_mapping):
            start, end = offsets
            if start == end:
                continue

            label = id_to_label[prediction]
            if label == 'O':
                if active_entity is not None:
                    entities.append(active_entity)
                    active_entity = None
                continue

            prefix, entity_type = label.split('-', 1)

            if (
                prefix == 'B'
                or active_entity is None
                or active_entity['entity_group'] != entity_type
                or start > int(active_entity['end']) + 1
            ):
                if active_entity is not None:
                    entities.append(active_entity)
                active_entity = {
                    'start': start,
                    'end': end,
                    'entity_group': entity_type,
                }
            else:
                active_entity['end'] = end

        if active_entity is not None:
            entities.append(active_entity)

        return [
            (
                text[int(entity['start']):int(entity['end'])],
                str(entity['entity_group']),
            )
            for entity in entities
        ]
    except Exception as exc:
        logger.exception('NER inference failed')
        raise EngineError('Failed to extract named entities') from exc


def process_query(query: str, k: int = 3) -> QueryResponse:
    """Run semantic search, enrichment, and summarization for a user query."""
    global _dataframe, _faiss_index

    if not query or not query.strip():
        raise EngineError('Query parameter cannot be empty', status_code=400)

    logger.info('Query received: %s', query)

    if _dataframe is None or _faiss_index is None:
        initialize_system()

    if _dataframe is None or _faiss_index is None or _embedding_model is None or _keyword_model is None:
        raise EngineError('System is not initialized')

    try:
        search_started_at = time.perf_counter()

        query_embedding = _embedding_model.encode([query], convert_to_numpy=True).astype('float32')
        faiss.normalize_L2(query_embedding)

        scores, indices = _faiss_index.search(query_embedding, k)
        search_time = time.perf_counter() - search_started_at
        logger.info('Semantic search completed in %.3f seconds', search_time)

        results: list[PaperResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            row = _dataframe.iloc[idx]
            title = str(row['title'])
            abstract = str(row['abstract'])

            keywords = _keyword_model.extract_keywords(
                abstract,
                keyphrase_ngram_range=(1, 3),
                stop_words='english',
                top_n=5
            )
            entities = extract_entities(abstract)

            logger.info('Title: %s', title)
            logger.info('Similarity Score: %.4f', float(score))
            logger.info('KeyBERT Keywords: %s', keywords)
            logger.info('NER Entities: %s', entities)

            results.append({
                'title': title,
                'abstract': abstract,
                'score': float(score),
                'keywords': keywords,
                'entities': entities
            })
    except EngineError:
        raise
    except Exception as exc:
        logger.exception('Query inference failed')
        raise EngineError('Failed to process query') from exc

    top_abstracts = [r['abstract'] for r in results[:3]]
    context = ' '.join(top_abstracts)

    prompt = f'Synthesize a technical summary answering the query [{query}] using these references: [{context}]'

    max_input_chars = 4000
    if len(prompt) > max_input_chars:
        prompt = prompt[:max_input_chars]

    summary_started_at = time.perf_counter()
    final_summary = summarize_text(
        prompt,
        max_length=180,
        min_length=60,
    )
    summary_time = time.perf_counter() - summary_started_at
    logger.info('Summary generation completed in %.3f seconds', summary_time)

    return {
        'query': query,
        'generative_summary': final_summary,
        'top_papers': [
            {
                'title': result['title'],
                'similarity_score': result['score'],
                'keywords': result['keywords'],
                'entities': result['entities'],
            }
            for result in results
        ],
    }
