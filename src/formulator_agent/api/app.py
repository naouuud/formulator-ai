from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from formulator_agent.agent.errors import AgentError
from formulator_agent.agent.run_agent import QAPair, run_agent
from formulator_agent.api.errors import BAD_REQUEST, INTERNAL_SERVER_ERROR, APIError

app = FastAPI(title="Formulator Agent API")


class GenerateRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=1,
        description="Query for the agent to generate a form or clarification prompt.",
    )
    qa_pairs: list[QAPair] | None = Field(
        None,
        description="Clarifications to help the agent refine its result.",
    )


async def _generate_from_query(query: str, qa_pairs: list[QAPair] | None) -> Any:
    cleaned_query = (query or "").strip()
    if not cleaned_query:
        raise APIError(
            internal_error_code=BAD_REQUEST,
            title="Bad request body",
            status=400,
            detail="query should have at least 1 character.",
        )

    try:
        result = await run_agent(query=cleaned_query, qa_pairs=qa_pairs)
    except AgentError as exc:
        raise APIError(
            internal_error_code=INTERNAL_SERVER_ERROR,
            title="Server error",
            status=500,
            detail="Unexpected error occurred. Please try again.",
        ) from exc

    return result


@app.get("/")
async def root() -> dict[str, str]:
    return {"message": "Formulator Agent API is running"}


@app.post("/generate")
async def generate_post(payload: GenerateRequest) -> Any:
    return await _generate_from_query(payload.query, payload.qa_pairs)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "status": 400,
            "title": "Invalid request payload",
            "detail": exc.errors(),
        },
    )


@app.exception_handler(APIError)
async def http_exception_handler(request: Request, exc: APIError):
    return JSONResponse(
        status_code=exc.status,
        content={"status": exc.status, "title": exc.title, "detail": exc.detail},
    )
