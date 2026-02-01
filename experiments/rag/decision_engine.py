"""
Decision engine for traffic incident pipeline.
Consumes event_type_candidates, signals, city; returns structured decision + policy excerpts.
LLM-only; requires OPENAI_API_KEY.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from utils import parse_json_from_llm
from .retriever import PolicyRetriever
from .schemas import RetrievedExcerpt


class DecisionInput(BaseModel):
    """Input to the decision engine."""

    model_config = ConfigDict(extra="forbid")

    event_type_candidates: list[str] = Field(default_factory=list)
    signals: list[str] = Field(default_factory=list)
    city: str = "Providence"


class SupportingExcerpt(BaseModel):
    """A policy excerpt supporting the decision."""

    model_config = ConfigDict(extra="forbid")

    text: str
    document_id: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class DecisionOutput(BaseModel):
    """Output from the decision engine."""

    model_config = ConfigDict(extra="forbid")

    decision: dict[str, Any] = Field(default_factory=dict)
    explanation: str = ""
    supporting_excerpts: list[SupportingExcerpt] = Field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


DECISION_SCHEMA = """
{
  "event_type": "primary event type from candidates (e.g. lane_blockage, multi_vehicle_incident)",
  "recommended_actions": ["ACTION_CODE", "..."],
  "severity": "none|low|medium|high|critical",
  "explanation": "2-4 sentence human-readable explanation grounding the decision in the policy excerpts"
}
"""


class DecisionEngine:
    """
    Consumes event context, retrieves relevant policies, produces structured decision.
    LLM-only; requires OPENAI_API_KEY.
    """

    def __init__(
        self,
        retriever: PolicyRetriever,
        top_k: int = 5,
        *,
        llm_model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ):
        self._retriever = retriever
        self._top_k = top_k
        self._llm_model = llm_model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")

    def decide(self, inp: DecisionInput) -> DecisionOutput:
        """
        Produce a decision from event_type_candidates, signals, and city.
        Returns structured JSON decision, human-readable explanation, and policy excerpts.
        """
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY required for RAG decision engine")

        query_parts = inp.event_type_candidates + inp.signals
        query = " ".join(query_parts) if query_parts else "traffic incident policy"

        excerpts: list[RetrievedExcerpt] = self._retriever.retrieve(
            query=query,
            top_k=self._top_k,
            city=inp.city,
            doc_type=None,
        )

        if not excerpts and inp.city:
            excerpts = self._retriever.retrieve(
                query=query,
                top_k=self._top_k,
                city=None,
                doc_type=None,
            )

        supporting = [
            SupportingExcerpt(
                text=e.text,
                document_id=e.document_id,
                score=e.score,
                metadata=e.metadata,
            )
            for e in excerpts
        ]

        return self._decide_with_llm(inp, excerpts, supporting)

    def _decide_with_llm(
        self,
        inp: DecisionInput,
        excerpts: list[RetrievedExcerpt],
        supporting: list[SupportingExcerpt],
    ) -> DecisionOutput:
        """Use LLM to derive decision from event context and policy excerpts."""
        from openai import OpenAI

        policy_text = "\n\n---\n\n".join(
            f"[{e.document_id}] {e.text}" for e in excerpts
        ) or "(No policy excerpts retrieved)"

        prompt = f"""You are a traffic incident decision assistant for {inp.city}.

Given the event context and relevant policy excerpts, produce a structured decision.

## Event context
- Event type candidates: {inp.event_type_candidates}
- Signals: {inp.signals}
- City: {inp.city}

## Relevant policy excerpts
{policy_text}

## Task
Output exactly one JSON object (no markdown, no extra text) following this schema:
{DECISION_SCHEMA}

Action codes: DISPATCH_REVIEW, SAVE_EVIDENCE_BUNDLE, TRAFFIC_ALERT, ANNOTATE_MAP, LOG_EVENT, ESCALATE_INCIDENT, NOTIFY_FLEET_MANAGER
Ground your decision in the policy excerpts. Pick event_type from the candidates. Use severity: critical for life-threatening, high for emergency response or multi-vehicle, medium for lane blockage or stalled vehicle, low for minor, none for non-events.
"""

        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.choices[0].message.content
        data = parse_json_from_llm(text)

        primary_type = data.get("event_type") or (inp.event_type_candidates[0] if inp.event_type_candidates else "unknown")
        actions = data.get("recommended_actions") or ["LOG_EVENT"]
        if not isinstance(actions, list):
            actions = [actions] if actions else ["LOG_EVENT"]
        severity = data.get("severity") or "low"
        explanation = data.get("explanation") or ""

        decision: dict[str, Any] = {
            "event_type": primary_type,
            "event_type_candidates": inp.event_type_candidates,
            "signals": inp.signals,
            "city": inp.city,
            "recommended_actions": actions,
            "severity": severity,
        }

        return DecisionOutput(
            decision=decision,
            explanation=explanation,
            supporting_excerpts=supporting,
        )
