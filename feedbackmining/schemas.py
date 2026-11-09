"""Pydantic schemas for both pipeline stages (inputs, LLM outputs, stored records)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Note(BaseModel):
    id: str = Field(min_length=1)
    text: str


class Stage1Result(BaseModel):
    """Cheap screening: is a missing product capability part of why the deal was lost?"""

    model_config = ConfigDict(extra="ignore")

    id: str
    has_feature_issue: bool
    signal_strength: int = Field(ge=0, le=3, description=(
        "0 = no product gap; 1 = vague product-fit complaint; "
        "2 = specific gap mentioned as a secondary reason; 3 = specific gap is the main reason"))

    @model_validator(mode="after")
    def _consistent(self):
        if self.has_feature_issue != (self.signal_strength >= 1):
            raise ValueError("has_feature_issue must be True iff signal_strength >= 1")
        return self


class Stage1Batch(BaseModel):
    results: list[Stage1Result]


class Stage2Result(BaseModel):
    """Stronger extraction: which specific capabilities were missing."""

    model_config = ConfigDict(extra="ignore")

    id: str
    missing_features: list[str] = Field(default_factory=list, description="short phrases, as written")
    evidence: str | None = Field(default=None, description="shortest quote supporting the extraction")
    canonical_features: list[str] = Field(default_factory=list, description="taxonomy ids (filled by us)")

    @field_validator("missing_features")
    @classmethod
    def _strip(cls, v: list[str]) -> list[str]:
        out = []
        for s in v:
            s = s.strip()
            if s and s not in out:
                out.append(s)
        return out


class Stage2Batch(BaseModel):
    results: list[Stage2Result]
