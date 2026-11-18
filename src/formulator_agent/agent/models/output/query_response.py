from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Query(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(
        description="Question for the user to clarify their original request or help you choose between different approaches."
    )
    options: list[str] = Field(
        default_factory=list,
        description="Options provided to the user to help clarify their original request. Includes 'Other' for a free form answer.",
    )


class QueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_type: Literal["query"] = Field(
        ...,
        description="Required discriminator. MUST be exactly 'query' for this payload.",
    )
    queries: list[Query] = Field(
        description="All queries in this round of clarification, for a maximum of 3.",
        default_factory=list,
        min_length=1,
        max_length=3,
    )
