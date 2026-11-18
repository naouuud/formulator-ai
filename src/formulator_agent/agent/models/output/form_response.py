from typing import Annotated, Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, computed_field, model_validator

from formulator_agent.agent.errors import StructuredCompletionError


class Validators(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required: bool = Field(
        description="Indicates whether field is required.",
    )


class Option(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="User facing option label, including capitalization, spacing and any other formatting."
    )

    @computed_field
    @property
    def value(self) -> str:
        return self.label

    @computed_field
    @property
    def id(self) -> str:
        return str(uuid4())


class Note(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="Note content, e.g. 'The following questions will focus on your past experience. Please try to answer them to the best of your ability.'"
    )


Non_Option_Types = Literal[
    "text",
]
Option_Types = Literal["radio", "checkbox", "select"]


class BaseQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str = Field(
        description="Question label or text, e.g. 'First Name' or 'How did you hear about us?' or 'Which of these best describes your situation:'"
    )
    validators: Validators = Field(
        description="Validation rules to be checked when user submits their answers.",
    )


class NonOptionsQuestion(BaseQuestion):
    html_type: Non_Option_Types = Field(
        description=f"Allowed values: {','.join(value for value in get_args(Non_Option_Types))}. Do not use any other value.",
        serialization_alias="htmlType",
    )


class OptionsQuestion(BaseQuestion):
    model_config = ConfigDict(extra="forbid")

    html_type: Option_Types = Field(
        description=f"Allowed values: {','.join(value for value in get_args(Option_Types))}. Do not use any other value.",
        serialization_alias="htmlType",
    )
    options: list[Option] = Field(
        min_length=1,
        description="Available options for option type questions, at least one option required.",
    )

    @model_validator(mode="after")
    def validate_question(self) -> "OptionsQuestion":
        if self.html_type in Option_Types and not self.options:
            raise StructuredCompletionError(
                f"Question of type {self.html_type} must include at least one option.",
            )
        return self


Question = Annotated[
    NonOptionsQuestion | OptionsQuestion, Field(discriminator="html_type")
]


class NoteElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["note"] = Field(
        ...,
        description="Required discriminator. MUST be exactly 'note' for this payload",
    )
    el: Note

    @computed_field
    @property
    def id(self) -> str:
        return str(uuid4())


class QuestionElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["question"] = Field(
        ...,
        description="Required discriminator. MUST be exactly 'question' for this payload",
    )
    el: Question

    @computed_field
    @property
    def id(self) -> str:
        return str(uuid4())


Element = Annotated[NoteElement | QuestionElement, Field(discriminator="type")]


class Page(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        default="", description="Optional page title; can be left empty."
    )
    elements: list[Element] = Field(
        default_factory=list,
        description="Elements may be questions or notes designed to provide more information to the responder.",
    )

    @computed_field
    @property
    def id(self) -> str:
        return str(uuid4())


class FormResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    output_type: Literal["form"] = Field(
        ...,
        description="Required discriminator. MUST be exactly 'form' for this payload.",
    )
    draft_title: str = Field(
        description="Working draft title, not responder-facing. Lives in form builder's workspace only.",
    )
    pages: list[Page] = Field(description="Surveys may be single- or multi-paged.")
