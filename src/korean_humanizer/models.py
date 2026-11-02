"""Explicit records for provenance, generated pairs and review outcomes."""

from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_serializer,
    model_validator,
)


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Rights(Record):
    training: Literal["approved", "pending", "denied"] = "pending"
    third_party_transfer: Literal["approved", "pending", "denied"] = "pending"
    derivative_creation: Literal["approved", "pending", "denied"] = "pending"
    redistribution: Literal["approved", "pending", "denied"] = "pending"
    evidence_ref: str = ""


class Source(Record):
    source_doc_id: str = Field(min_length=1)
    author_group: str = Field(min_length=1)
    source_group: str = Field(min_length=1)
    genre: str = Field(min_length=1)
    text: str = Field(min_length=1)
    provenance: str = Field(min_length=1)
    origin: Literal["human", "synthetic_fixture"]
    source_year: int | None = Field(default=None, ge=1000, le=2100)
    rights: Rights
    protected_spans: list[str] = Field(default_factory=list)

    @field_validator("protected_spans")
    @classmethod
    def nonempty_spans(cls, spans: list[str]) -> list[str]:
        if any(not span.strip() for span in spans):
            raise ValueError("protected spans must not be empty")
        return spans


Split = Literal["train", "dev", "test"]


class AssignedSource(Record):
    source: Source
    split: Split
    group_id: str


class Generation(Record):
    model: str
    prompt_version: str
    seed: int
    mock: bool


class Pair(Record):
    pair_id: str
    source: Source
    synthetic_ai_text: str = Field(min_length=1)
    split: Split
    generation: Generation
    review_status: Literal["pending", "accepted", "rejected", "mock_only"] = "pending"
    review_ref: str = ""
    input_kind: Literal["ai_draft", "natural_control"] = "ai_draft"

    @model_validator(mode="after")
    def control_is_not_ai_generation(self):
        if self.input_kind == "natural_control":
            if (
                self.synthetic_ai_text != self.source.text
                or self.generation.model != "identity-control"
                or self.generation.prompt_version != "natural-control-v1"
                or self.generation.mock
            ):
                raise ValueError(
                    "natural controls require identical text and explicit non-AI metadata"
                )
        elif self.generation.model == "identity-control":
            raise ValueError("identity controls must be labeled natural_control")
        return self

    @model_serializer(mode="wrap")
    def serialize_with_legacy_compatibility(self, handler):
        value = handler(self)
        if self.input_kind == "ai_draft":
            value.pop("input_kind", None)
        return value


class TextResult(Record):
    item_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
