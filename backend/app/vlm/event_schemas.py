"""
Structured schemas for VLM event output.
Machine-readable for pipelines, RAG, and routing.
"""

from enum import Enum
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator

EnumType = TypeVar("EnumType", bound=Enum)

def normalize_enum(value: Any, enum_cls: type[EnumType], default: EnumType) -> EnumType:
    """Normalize string values to enum members (case-insensitive, strips whitespace)."""
    if isinstance(value, enum_cls):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        try:
            return enum_cls(normalized)
        except ValueError:
            return default
    return default

class EventCategory(str, Enum):
    incident = "incident"
    traffic_condition = "traffic_condition"
    non_event = "non_event"
    unknown = "unknown"


class Severity(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ImpactLevel(str, Enum):
    none = "none"
    low = "low"
    medium = "medium"
    high = "high"


# ---------------------------------------------------------------------------
# Event schemas
# ---------------------------------------------------------------------------

class VLMBase(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

class EventImpact(VLMBase):
    safety_risk: ImpactLevel = ImpactLevel.none
    traffic_disruption: ImpactLevel = ImpactLevel.none

    @field_validator("safety_risk", "traffic_disruption", mode="before")
    @classmethod
    def normalize_impact_level(cls, v: Any) -> ImpactLevel:
        return normalize_enum(v, ImpactLevel, ImpactLevel.none)


class EventInfo(VLMBase):
    category: EventCategory = EventCategory.unknown
    type: str = "unknown"
    severity: Severity = Severity.none
    confidence: float = Field(ge=0, le=1, default=0)
    impact: EventImpact = Field(default_factory=EventImpact)

    @field_validator("category", mode="before")
    @classmethod
    def normalize_category(cls, v: Any) -> EventCategory:
        return normalize_enum(v, EventCategory, EventCategory.unknown)

    @field_validator("severity", mode="before")
    @classmethod
    def normalize_severity(cls, v: Any) -> Severity:
        return normalize_enum(v, Severity, Severity.none)

    @field_validator("confidence", mode="before")
    @classmethod
    def normalize_confidence(cls, v: Any) -> float:
        try:
            f = float(v)
            return max(0, min(1, f))
        except (TypeError, ValueError):
            return 0.0


class RagInfo(VLMBase):
    tags: list[str] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)


class VLMEventOutput(VLMBase):
    event_id: str = ""
    event: EventInfo = Field(default_factory=EventInfo)
    actors: list[dict[str, Any]] = Field(default_factory=list)
    summary: dict[str, Any] = Field(default_factory=dict)
    timeline: dict[str, Any] | None = None
    evidence: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty: dict[str, Any] | None = None
    recommended_actions: list[dict[str, Any]] = Field(default_factory=list)
    rag: RagInfo = Field(default_factory=RagInfo)
    artifacts: dict[str, Any] = Field(default_factory=dict)

    @field_validator("summary", "artifacts", "timeline", "uncertainty", mode="before")
    @classmethod
    def normalize_dict(cls, v: Any) -> dict | None:
        if v is None:
            return None
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("event", mode="before")
    @classmethod
    def normalize_event(cls, v: Any) -> EventInfo | dict[str, Any]:
        """Normalize event field: return empty dict if None, otherwise let Pydantic parse into EventInfo."""
        if v is None:
            return {}
        if isinstance(v, EventInfo):
            return v
        if isinstance(v, dict):
            return v
        return {}

    @field_validator("actors", "evidence", "recommended_actions", mode="before")
    @classmethod
    def normalize_list(cls, v: Any) -> list:
        if v is None:
            return []
        if isinstance(v, list):
            return v
        return []

    @field_validator("rag", mode="before")
    @classmethod
    def normalize_rag(cls, v: Any) -> RagInfo | dict[str, Any]:
        """Normalize rag field: return empty dict if None, otherwise let Pydantic parse into RagInfo."""
        if v is None:
            return {}
        if isinstance(v, RagInfo):
            return v
        if isinstance(v, dict):
            return v
        return {}
