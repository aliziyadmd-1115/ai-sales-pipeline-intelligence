"""Shared validation and normalization for training, API, and UI inputs."""
from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

CATEGORIES = {
    "region": ("Northeast", "South", "Midwest", "West"),
    "industry": ("technology", "financial_services", "healthcare", "retail", "manufacturing"),
    "customer_segment": ("SMB", "Mid-Market", "Enterprise"),
    "product_line": ("analytics", "automation", "data_platform", "security"),
    "sales_stage": ("qualification", "discovery", "proposal", "negotiation"),
}
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")


def redact_pii(text: str) -> str:
    """Redact email addresses only; this is not a general PII detector."""
    return EMAIL_PATTERN.sub("[REDACTED_EMAIL]", str(text))


class RequestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True, allow_inf_nan=False)


class OpportunityRequest(RequestModel):
    notes: str = Field(..., min_length=20, max_length=5000)
    region: str = "Northeast"
    industry: str = "technology"
    customer_segment: str = "Mid-Market"
    product_line: str = "analytics"
    sales_stage: str = "discovery"
    estimated_value: float = Field(50000, gt=0)
    days_in_pipeline: int = Field(45, ge=0, le=1000)
    engagement_score: int = Field(60, ge=0, le=100)
    meetings_count: int = Field(3, ge=0, le=100)
    competitor_present: bool = False
    discount_pct: float = Field(0.10, ge=0, le=1)
    proposal_sent: bool = False
    decision_threshold: float = Field(0.50, ge=0, le=1)

    @field_validator(*CATEGORIES, mode="before")
    @classmethod
    def normalize_category(cls, value, info: ValidationInfo):
        choices = {choice.casefold(): choice for choice in CATEGORIES[info.field_name]}
        if not isinstance(value, str) or value.strip().casefold() not in choices:
            raise ValueError(f"must be one of {', '.join(choices.values())}")
        return choices[value.strip().casefold()]

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, value):
        if not isinstance(value, str):
            raise ValueError("notes must be text")
        return redact_pii(" ".join(value.split()))

    @field_validator("competitor_present", "proposal_sent", mode="before")
    @classmethod
    def normalize_boolean(cls, value):
        values = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}
        key = str(value).strip().lower()
        if key not in values:
            raise ValueError("must be a boolean, 0/1, or yes/no")
        return values[key]


class SearchRequest(RequestModel):
    query: str = Field(..., min_length=10, max_length=2000)
    top_k: int = Field(3, ge=1, le=10)

    @field_validator("query", mode="before")
    @classmethod
    def normalize_query(cls, value):
        if not isinstance(value, str):
            raise ValueError("query must be text")
        return redact_pii(" ".join(value.split()))


class AnswerRequest(SearchRequest):
    use_llm: bool = False
