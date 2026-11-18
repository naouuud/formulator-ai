"""Basic OpenAI LLM client factory.

Returns an ``openai.OpenAI`` instance, which satisfies the ``LLMClient``
protocol in ``structured_completion`` (it exposes ``chat.completions.create``).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import AsyncOpenAI

from formulator_agent.agent.errors import LLMConfigError

load_dotenv()

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2


def _env(name: str) -> str | None:
    value = os.getenv(name)
    return value.strip() if value and value.strip() else None


def _env_number(name: str, default: float, cast: type) -> float:
    raw = _env(name)
    if raw is None:
        return default
    try:
        return cast(raw)
    except ValueError as exc:
        raise LLMConfigError(
            f"{name} must be a valid {cast.__name__}, got {raw!r}"
        ) from exc


def get_default_model() -> str:
    """Model name from OPENAI_MODEL, falling back to DEFAULT_MODEL."""
    return _env("OPENAI_MODEL") or DEFAULT_MODEL


def create_client(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    organization: str | None = None,
    project: str | None = None,
    timeout: float | None = None,
    max_retries: int | None = None,
) -> AsyncOpenAI:
    """Build a new OpenAI client. Explicit arguments override env variables."""
    api_key = api_key or _env("OPENAI_API_KEY")
    if not api_key:
        raise LLMConfigError("OPENAI_API_KEY is not set.")

    return AsyncOpenAI(
        api_key=api_key,
        base_url=base_url or _env("OPENAI_BASE_URL"),
        organization=organization or _env("OPENAI_ORG_ID"),
        project=project or _env("OPENAI_PROJECT_ID"),
        timeout=timeout
        if timeout is not None
        else _env_number("OPENAI_TIMEOUT", DEFAULT_TIMEOUT, float),
        max_retries=(
            max_retries
            if max_retries is not None
            else int(_env_number("OPENAI_MAX_RETRIES", DEFAULT_MAX_RETRIES, int))
        ),
    )


@lru_cache(maxsize=1)
def get_client() -> AsyncOpenAI:
    """Shared, lazily created client configured from the environment."""
    return create_client()
