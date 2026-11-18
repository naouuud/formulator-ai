import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from dotenv import load_dotenv

from formulator_agent.agent.errors import AgentError
from formulator_agent.agent.llm import create_client
from formulator_agent.agent.logger import ObsLogger
from formulator_agent.agent.models.output.form_response import FormResponse
from formulator_agent.agent.models.output.output_container import OutputContainer
from formulator_agent.agent.models.output.query_response import QueryResponse
from formulator_agent.agent.structured_completion import (
    LLMClient,
    calc_total_cost,
    cost_message,
    structured_completion,
)

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
with open(FIXTURES_DIR / "example_0.json", "r", encoding="utf-8") as f:
    example_0 = json.load(f)

load_dotenv()

DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_QUERY = "Candidate screening form for PhD history program. Ask about areas of expertise and include short, multiple-choice questions and long-form essay questions."

SYSTEM_PROMPT = """
You are a form generation assistant. Create a useful form based on the user's query. 
If the user's request is too unclear or generic, DO NOT output a form. Instead, ask the user helpful clarification questions. 
Do not exceed 3 rounds of questions (check 'question_round' in the messages).

General guidelines:  
- Use a variety of question types. Whenever possible, use options-based questions, and provide an generous list of possible options.
- If applicable, provide an 'Other' or 'I don't know' option. 
- Fields like First Name, Last Name and contact information should be required unless there is a good reason.
- To make a field required, the 'required' validator should be set to true.
- Not every question has to be required.
- For checkbox questions, setting 'required' to true means that at least one option must be checked. Otherwise, the responder may leave all options unchecked.

Critical output rules:
- You must produce valid JSON matching the provided schema exactly.
- Every object must include the required discriminator field `output_type`.
- Use `output_type: "form"` for form payloads and `output_type: "query"` for clarification payloads.
- Do not use any other output_type values such as "success", "complete", "error", or "done".
- Do not add extra top-level keys beyond the schema.
- Omit any commentary or markdown fences.
- Every html_type must be exactly one of: text, radio, checkbox, select.
- Never invent HTML input types not listed above.
"""

Message = dict[str, Any]
QAPair = dict[Literal["question", "answer"], str]


def _build_messages(query: str, qa_pairs: list[QAPair] | None = None) -> list[Message]:
    system_msg = {"role": "system", "content": SYSTEM_PROMPT}
    query_msg = {"role": "user", "content": query or DEFAULT_QUERY}
    if qa_pairs is None:
        return [system_msg, query_msg]
    else:
        qa_msgs: list[dict[str, str]] = []
        for item in qa_pairs:
            assistant_msg = {"role": "assistant", "content": item["question"]}
            user_msg = {"role": "user", "content": item["answer"]}
            qa_msgs += [assistant_msg, user_msg]
        return [system_msg, query_msg, *qa_msgs]


COMPLETION_KWARGS = {"temperature": 0.2}


async def run_agent(
    query: str,
    qa_pairs: list[QAPair] | None = None,
    provider: str = "gemini",
    max_repair_attempts: int = 2,
    response_model: type = OutputContainer,
):
    if provider == "openai":
        client: LLMClient = create_client()
        model = os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    else:
        client: LLMClient = create_client(
            base_url=os.getenv("GEMINI_BASE_URL"),
            api_key=os.getenv("GEMINI_API_KEY"),
            organization="not-applicable",
            project="not-applicable",
        )
        model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL)

    total_attempts = max_repair_attempts + 1
    logger = ObsLogger()
    request_id = str(uuid4())

    logger.log_event(
        request_id=request_id,
        event="structured_completion started",
        model=model,
        response_model=response_model.__name__,
        query=query,
        max_attempts=total_attempts,
    )

    try:
        result = await structured_completion(
            client=client,
            model=model,
            request_id=request_id,
            messages=_build_messages(query=query, qa_pairs=qa_pairs),
            response_model=OutputContainer,
            total_attempts=total_attempts,
            logger=logger,
            strict_schema=False,
            **COMPLETION_KWARGS,
        )
    except AgentError as exc:
        logger.log_event(
            request_id=request_id,
            event="structured_completion failed",
            error_code=exc.error_code,
            detail=exc.detail,
        )
        print(f"Generation failed: {exc.error_code}: {exc.detail}")
        raise

    container = result.data
    output = container.output

    attempts = result.attempts
    last_attempt = attempts[-1].attempt
    total_tokens_in = sum(attempt.tokens_in for attempt in attempts)
    total_tokens_out = sum(attempt.tokens_out for attempt in attempts)
    total_cost = calc_total_cost(attempts)

    if isinstance(output, (FormResponse, QueryResponse)):
        logger.log_event(
            request_id=request_id,
            event="structured_completion succeeded",
            attempt_number=last_attempt,
            repairs_used=last_attempt - 1,
            last_reasoning=container.reasoning,
            output_type=output.output_type,
            total_tokens_in=total_tokens_in,
            total_tokens_out=total_tokens_out,
            total_cost=cost_message(total_cost),
        )
        print(f"""{60 * "="}
Generation Complete
{60 * "="}
Query: {query}
Model: {model}
Repairs used: {last_attempt - 1} 
Tokens in: {total_tokens_in}
Tokens out: {total_tokens_out}
Cost (USD): {total_cost}
Last reasoning: {container.reasoning}
Output: {output.model_dump_json(indent=2)}""")
        return output
    else:
        logger.log_event(
            request_id=request_id,
            event="structured_completion failed",
            attempt_number=last_attempt,
            repairs_used=last_attempt - 1,
            total_cost=total_cost,
        )
        print("Generation failed, please try again.")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the structured agent.")
    parser.add_argument(
        "--query",
        type=str,
        default=DEFAULT_QUERY,
        help="The user query to process",
    )
    parser.add_argument(
        "--provider",
        type=str,
        default="gemini",
        help="The LLM provider to use",
        choices=["gemini", "openai"],
    )
    args = parser.parse_args()
    asyncio.run(run_agent(**vars(args)))
