from typing import Annotated

from pydantic import BaseModel, ConfigDict, Discriminator, Field

from .form_response import FormResponse
from .query_response import QueryResponse


class OutputContainer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reasoning: str = Field(
        description="Step-by-step reasoning before choosing the response type."
    )
    output: Annotated[QueryResponse | FormResponse, Discriminator("output_type")]
