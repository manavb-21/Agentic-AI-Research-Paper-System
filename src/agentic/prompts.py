"""Prompt templates for research-paper reasoning tasks.

The templates in this module are plain strings and formatting helpers. They do
not call an LLM and do not depend on LangChain, Streamlit, or API code.
"""

from __future__ import annotations

GENERAL_RESEARCH_ASSISTANT_SYSTEM_PROMPT = """
You are a careful research assistant for machine learning papers.
Use retrieved paper metadata and abstracts as the primary evidence.
Be precise, distinguish evidence from inference, and avoid inventing citations.
When evidence is insufficient, say what is missing and suggest a focused next
step.
""".strip()

PAPER_SUMMARIZATION_PROMPT_TEMPLATE = """
Summarize the following research paper for a technical reader.

Title:
{title}

Abstract:
{abstract}

Focus on:
- the core research problem
- the proposed method
- the main contribution
- important limitations or assumptions visible from the abstract
""".strip()

PAPER_COMPARISON_PROMPT_TEMPLATE = """
Compare the following research papers.

Paper A Title:
{title_a}

Paper A Abstract:
{abstract_a}

Paper B Title:
{title_b}

Paper B Abstract:
{abstract_b}

Compare them across objective, method, contribution, likely use cases, and key
differences. Ground the comparison only in the supplied text.
""".strip()

RESEARCH_RECOMMENDATION_PROMPT_TEMPLATE = """
Recommend the most relevant papers for the research query below.

Research query:
{query}

Retrieved papers:
{papers}

Explain why each recommended paper is relevant, rank the recommendations, and
note any missing evidence that would improve the recommendation.
""".strip()

DIFFICULTY_ESTIMATION_PROMPT_TEMPLATE = """
Estimate the difficulty of understanding the following research paper.

Title:
{title}

Abstract:
{abstract}

Assess prerequisite knowledge, mathematical depth, implementation complexity,
and expected reading difficulty. Return a clear difficulty level and rationale.
""".strip()

RESEARCH_GAP_ANALYSIS_PROMPT_TEMPLATE = """
Analyze possible research gaps for the topic below using the retrieved papers.

Research topic:
{topic}

Retrieved papers:
{papers}

Identify underexplored directions, unresolved limitations, possible experiments,
and practical next steps. Clearly separate evidence from hypothesis.
""".strip()


def build_paper_summarization_prompt(title: str, abstract: str) -> str:
    """Create a prompt for summarizing a single paper."""
    return PAPER_SUMMARIZATION_PROMPT_TEMPLATE.format(
        title=title.strip(),
        abstract=abstract.strip(),
    )


def build_paper_comparison_prompt(
    title_a: str,
    abstract_a: str,
    title_b: str,
    abstract_b: str,
) -> str:
    """Create a prompt for comparing two papers."""
    return PAPER_COMPARISON_PROMPT_TEMPLATE.format(
        title_a=title_a.strip(),
        abstract_a=abstract_a.strip(),
        title_b=title_b.strip(),
        abstract_b=abstract_b.strip(),
    )


def build_research_recommendation_prompt(query: str, papers: str) -> str:
    """Create a prompt for ranking retrieved papers against a research query."""
    return RESEARCH_RECOMMENDATION_PROMPT_TEMPLATE.format(
        query=query.strip(),
        papers=papers.strip(),
    )


def build_difficulty_estimation_prompt(title: str, abstract: str) -> str:
    """Create a prompt for estimating paper difficulty."""
    return DIFFICULTY_ESTIMATION_PROMPT_TEMPLATE.format(
        title=title.strip(),
        abstract=abstract.strip(),
    )


def build_research_gap_analysis_prompt(topic: str, papers: str) -> str:
    """Create a prompt for analyzing research gaps from retrieved papers."""
    return RESEARCH_GAP_ANALYSIS_PROMPT_TEMPLATE.format(
        topic=topic.strip(),
        papers=papers.strip(),
    )
