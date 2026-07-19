"""Workflow orchestration for the research-paper agent."""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict


class WorkflowError(RuntimeError):
    """Raised when the research-agent workflow cannot complete."""


class WorkflowResponse(TypedDict):
    """Structured response returned by the research workflow."""

    query: str
    answer: str
    retrieved_papers: list[dict[str, Any]]
    agent_result: Mapping[str, Any]


def validate_user_query(query: str) -> str:
    """Validate and normalize a user query before agent execution."""
    cleaned_query = query.strip() if query else ""
    if not cleaned_query:
        raise ValueError("query cannot be empty.")
    return cleaned_query


def _message_content(message: Any) -> Any:
    """Read content from a LangChain message or message-like dictionary."""
    if isinstance(message, Mapping):
        return message.get("content")
    return getattr(message, "content", None)


def _stringify_content(content: Any) -> str:
    """Convert LangChain message content into a plain response string."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, Sequence) and not isinstance(content, str):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text") or item.get("content")
                if text:
                    parts.append(str(text))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def _extract_final_response(agent_result: Mapping[str, Any]) -> str:
    """Extract the final assistant response from a LangChain agent result."""
    messages = agent_result.get("messages")
    if not isinstance(messages, Sequence) or not messages:
        raise WorkflowError("Agent response did not include any messages.")

    final_content = _message_content(messages[-1])
    final_response = _stringify_content(final_content)
    if not final_response:
        raise WorkflowError("Agent response was empty.")

    return final_response


def _parse_tool_content(content: Any) -> Any:
    """Parse structured tool content from LangChain tool messages."""
    if not isinstance(content, str):
        return content

    cleaned_content = content.strip()
    if not cleaned_content:
        return None

    try:
        return json.loads(cleaned_content)
    except json.JSONDecodeError:
        pass

    try:
        return ast.literal_eval(cleaned_content)
    except (SyntaxError, ValueError):
        return cleaned_content


def _paper_key(paper: Mapping[str, Any]) -> str:
    """Return a stable de-duplication key for a retrieved paper."""
    dataset_index = paper.get("dataset_index")
    if dataset_index is not None:
        return f"dataset_index:{dataset_index}"
    return f"title:{paper.get('title', '')}"


def _extract_retrieved_papers(agent_result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Collect structured paper objects from agent tool messages."""
    messages = agent_result.get("messages")
    if not isinstance(messages, Sequence):
        return []

    papers_by_key: dict[str, dict[str, Any]] = {}
    for message in messages:
        parsed_content = _parse_tool_content(_message_content(message))

        candidate_papers: list[Any] = []
        if isinstance(parsed_content, Mapping):
            results = parsed_content.get("results")
            if isinstance(results, list):
                candidate_papers.extend(results)
            elif {"title", "abstract", "dataset_index"}.issubset(parsed_content):
                candidate_papers.append(parsed_content)
        elif isinstance(parsed_content, list):
            candidate_papers.extend(parsed_content)

        for candidate in candidate_papers:
            if not isinstance(candidate, Mapping):
                continue
            if not {"title", "abstract"}.issubset(candidate):
                continue

            paper = dict(candidate)
            papers_by_key[_paper_key(paper)] = paper

    return list(papers_by_key.values())


def process_user_query_structured(query: str) -> WorkflowResponse:
    """Invoke the research agent and return answer plus retrieved papers."""
    cleaned_query = validate_user_query(query)

    try:
        from src.agentic.agent import create_research_agent

        agent = create_research_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": cleaned_query}]}
        )
        if not isinstance(result, Mapping):
            raise WorkflowError("Agent returned an unexpected response format.")

        return {
            "query": cleaned_query,
            "answer": _extract_final_response(result),
            "retrieved_papers": _extract_retrieved_papers(result),
            "agent_result": result,
        }
    except WorkflowError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise WorkflowError(f"Failed to process user query: {exc}") from exc


def process_user_query(query: str) -> str:
    """Validate a query, invoke the research agent, and return its answer."""
    return process_user_query_structured(query)["answer"]
