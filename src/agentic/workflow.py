"""Workflow orchestration for the research-paper agent."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


class WorkflowError(RuntimeError):
    """Raised when the research-agent workflow cannot complete."""


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


def process_user_query(query: str) -> str:
    """Validate a query, invoke the research agent, and return its response."""
    cleaned_query = validate_user_query(query)

    try:
        from src.agentic.agent import create_research_agent

        agent = create_research_agent()
        result = agent.invoke(
            {"messages": [{"role": "user", "content": cleaned_query}]}
        )
        if not isinstance(result, Mapping):
            raise WorkflowError("Agent returned an unexpected response format.")

        return _extract_final_response(result)
    except WorkflowError:
        raise
    except ValueError:
        raise
    except Exception as exc:
        raise WorkflowError(f"Failed to process user query: {exc}") from exc
