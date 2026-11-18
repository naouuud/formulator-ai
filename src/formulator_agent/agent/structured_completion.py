"""Structured completion with a validation/repair loop.

Makes a chat completion call through an injected OpenAI-SDK-compatible client,
parses the reply into a Pydantic model, and, if parsing or validation fails,
asks the model to repair its own output for a configurable number of attempts.

The LLM client and the logger are both injected, so this module has no
dependency on how either is constructed:

"""

import json
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol

import openai
from pydantic import BaseModel, ValidationError

from formulator_agent.agent.errors import (
    BadRequestError,
    LLMConfigError,
    StructuredCompletionError,
)


# Injected Dependencies
class LLMClient(Protocol):
    """Minimal surface of the async OpenAI SDK client that this module uses.

    ``openai.AsyncOpenAI`` satisfies this. A thin wrapper from your separate LLM
    client module will too, as long as it exposes ``chat.completions.create``.
    """

    chat: Any


class Logger(Protocol):
    """
    The future log module only needs to expose these methods. Extra context is
    passed through ``fields=`` as structured fields.
    """

    def log_event(self, *, request_id: str, event: str, **fields: Any) -> None: ...


# Result Types
@dataclass
class AttemptRecord:
    attempt: int
    raw_output: str | None
    error: str | None  # None on success
    tokens_in: int
    tokens_out: int
    cost: float | None


@dataclass
class StructuredResult[T: BaseModel]:
    data: T  # instance of response_model
    attempts: list[AttemptRecord] = field(default_factory=list)

    @property
    def repairs_used(self) -> int:
        return max(len(self.attempts) - 1, 0)


# Internals
_REPAIR_PROMPT = (
    "Your previous response could not be used.\n"
    "Error:\n{error}\n\n"
    "Return a corrected response as a single JSON object that conforms to the "
    "required schema. Output only the JSON, with no commentary or code fences."
)


def _build_response_format(response_model: type[BaseModel], strict: bool) -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": response_model.model_json_schema(),
            "strict": strict,
        },
    }


def _strip_code_fences(text: str) -> str:
    """Models occasionally wrap JSON in ```json fences despite instructions."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        if text.rstrip().endswith("```"):
            text = text.rstrip()[:-3]
    return text.strip()


def _format_validation_error(exc: ValidationError) -> str:
    lines = []
    for err in exc.errors():
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "Schema validation failed:\n" + "\n".join(lines)


# Cost calculator
PRICE_TABLE: dict[str, dict[str, float]] = {
    "gemini-3.5-flash-lite": {"input_price": 0.3, "output_price": 2.5}
}


def _calc_cost(model: str, tokens_in: int, tokens_out: int) -> float | None:
    model_data = PRICE_TABLE.get(model)
    if model_data is None:
        print(f"Model '{model}' not found in price table")
        return None
    return (
        model_data["input_price"] * tokens_in + model_data["output_price"] * tokens_out
    ) / 1_000_000


def calc_total_cost(attempts: list[AttemptRecord]) -> float | None:
    if any(attempt.cost is None for attempt in attempts):
        return None
    else:
        return sum(attempt.cost or 0.0 for attempt in attempts)


def cost_message(val: float | None) -> float | Literal["Not available"]:
    if val is None:
        return "Not available"
    else:
        return round(val, 8)


# Public API
async def structured_completion[T: BaseModel](
    *,
    client: LLMClient,
    model: str,
    request_id: str,
    messages: list[dict[str, Any]],
    response_model: type[T],
    total_attempts: int = 3,
    logger: Logger,
    strict_schema: bool = False,
    **completion_kwargs: Any,
) -> StructuredResult[T]:
    """Request output matching ``response_model``, repairing on failure.

    Args:
        client: OpenAI-SDK-compatible client (provided by the LLM client module).
        model: Model name.
        messages: Chat messages. Not mutated; a working copy is used.
        response_model: Pydantic model describing the desired output.
        max_repair_attempts: Number of *additional* tries after the first call.
            0 means a single attempt with no repair. Total calls = 1 + this.
        logger: Optional logger. Defaults to ``logging.getLogger(__name__)``.
        strict_schema: Pass ``strict=True`` to OpenAI's JSON-schema mode. This
            requires a schema that satisfies OpenAI's strict-mode restrictions
            (all fields required, ``additionalProperties: false``).
        **completion_kwargs: Forwarded to ``chat.completions.create``
            (temperature, max_tokens, timeout, ...).

    Returns:
        StructuredResult with the validated model in ``.data`` and a record of
        every attempt.

    Raises:
        StructuredCompletionError: if all attempts fail validation. SDK/network
            errors from the client are not swallowed; handle retries for those
            in the client layer.
    """
    if total_attempts < 0:
        raise StructuredCompletionError("max_repair_attempts must be >= 0")

    response_format = _build_response_format(response_model, strict_schema)
    records: list[AttemptRecord] = []

    for attempt in range(1, total_attempts + 1):
        logger.log_event(
            request_id=request_id,
            event="calling model",
            attempt=attempt,
            messages_count=len(messages),
        )

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                response_format=response_format,
                **completion_kwargs,
            )
        except openai.AuthenticationError as exc:
            raise LLMConfigError(exc)
        except openai.BadRequestError as exc:
            raise BadRequestError(exc)

        usage = response.usage
        tokens_in = usage.prompt_tokens
        tokens_out = usage.completion_tokens
        attempt_cost = _calc_cost(model, tokens_in, tokens_out)
        choice = response.choices[0]
        message = choice.message
        raw = message.content

        error: str | None = None
        parsed: T | None = None

        if getattr(message, "refusal", None):
            # A refusal won't be fixed by repair; fail fast.
            error = f"Model refused: {message.refusal}"
            records.append(
                AttemptRecord(attempt, raw, error, tokens_in, tokens_out, attempt_cost)
            )
            logger.log_event(
                request_id=request_id,
                event="model refusal",
                attempt=attempt,
                refusal=message.refusal,
                total_cost=cost_message(calc_total_cost(records)),
            )
            raise StructuredCompletionError(error)

        if not raw:
            error = "Empty response content."
            if choice.finish_reason == "length":
                error += " Output was truncated (finish_reason=length)."
        else:
            try:
                parsed = response_model.model_validate(
                    json.loads(_strip_code_fences(raw))
                )
            except json.JSONDecodeError as exc:
                error = f"Response was not valid JSON: {exc}"
            except ValidationError as exc:
                error = _format_validation_error(exc)

        records.append(
            AttemptRecord(attempt, raw, error, tokens_in, tokens_out, attempt_cost)
        )

        if parsed is not None:
            return StructuredResult(data=parsed, attempts=records)

        logger.log_event(
            request_id=request_id,
            event="structured output invalid",
            attempt=attempt,
            max_attempts=total_attempts,
            error=error,
            finish_reason=choice.finish_reason,
            total_cost=cost_message(calc_total_cost(records)),
        )

        if attempt < total_attempts:
            messages.append({"role": "assistant", "content": raw or ""})
            messages.append(
                {"role": "user", "content": _REPAIR_PROMPT.format(error=error)}
            )

    logger.log_event(
        request_id=request_id,
        event="structured_completion exhausted attempts",
        attempts=total_attempts,
        last_error=records[-1].error,
        total_cost=cost_message(calc_total_cost(records)),
    )
    raise StructuredCompletionError(
        f"No valid {response_model.__name__} after {total_attempts} attempt(s). "
        f"Last error: {records[-1].error}"
    )
