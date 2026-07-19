"""Configuration for the agentic research assistant layer."""

from __future__ import annotations

import os

from dotenv import load_dotenv

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
DEFAULT_TOP_K = 5


class ConfigurationError(RuntimeError):
    """Raised when required application configuration is missing."""


load_dotenv()


def _get_required_env(name: str) -> str:
    """Read and validate a required environment variable."""
    value = os.getenv(name)
    if not value:
        raise ConfigurationError(
            f"{name} is missing. Add {name} to your .env file before "
            "initializing the Groq LLM."
        )
    return value


GROQ_API_KEY = _get_required_env("GROQ_API_KEY")
