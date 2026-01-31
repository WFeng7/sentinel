"""
Decision engine for traffic incident pipeline.
Consumes event_type_candidates, signals, city; returns structured decision + policy excerpts.
Uses LLM when API key available; falls back to rule-based otherwise.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .retriever import PolicyRetriever
from .schemas import RetrievedExcerpt


@dataclass
class DecisionInput:
    """Input to the decision engine."""

    event_type_candidates: list[str]
    signals: list[str]
    city: str = "Providence"


@dataclass
class SupportingExcerpt:
    """A policy excerpt supporting the decision."""

    text: str
    document_id: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DecisionOutput:
    """Output from the decision engine."""

    decision: dict[str, Any]
    explanation: str
    supporting_excerpts: list[SupportingExcerpt]

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision": self.decision,
            "explanation": self.explanation,
            "supporting_excerpts": [
                {
                    "text": e.text,
                    "document_id": e.document_id,
                    "score": e.score,
                    "metadata": e.metadata,
                }
                for e in self.supporting_excerpts
            ],
        }


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
    Uses LLM when OPENAI_API_KEY is set; otherwise rule-based fallback.
    """

    def __init__(
        self,
        retriever: PolicyRetriever,
        top_k: int = 5,
        *,
        llm_model: str = "gpt-5.1",
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

        if self._api_key:
            try:
                return self._decide_with_llm(inp, excerpts, supporting)
            except Exception:
                pass

        return self._decide_rule_based(inp, supporting)

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
            max_tokens=512,
            temperature=0.2,
        )
        text = resp.choices[0].message.content
        data = self._parse_json_response(text)

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

    def _parse_json_response(self, text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```\s*$", "", text)
        return json.loads(text)

    def _decide_rule_based(self, inp: DecisionInput, supporting: list[SupportingExcerpt]) -> DecisionOutput:
        """Fallback when no LLM available."""
        primary_type = inp.event_type_candidates[0] if inp.event_type_candidates else "unknown"
        actions = self._derive_actions(primary_type, inp.signals)
        severity = self._derive_severity(primary_type, inp.signals)

        decision: dict[str, Any] = {
            "event_type": primary_type,
            "event_type_candidates": inp.event_type_candidates,
            "signals": inp.signals,
            "city": inp.city,
            "recommended_actions": actions,
            "severity": severity,
        }

        explanation = self._build_explanation(
            primary_type=primary_type,
            signals=inp.signals,
            actions=actions,
            excerpt_count=len(supporting),
        )

        return DecisionOutput(
            decision=decision,
            explanation=explanation,
            supporting_excerpts=supporting,
        )

    def _derive_actions(self, primary_type: str, signals: list[str]) -> list[str]:
        actions: list[str] = []
        if "lane_blocked" in signals or "lane_blockage" in primary_type:
            actions.extend(["TRAFFIC_ALERT", "ANNOTATE_MAP"])
        if "multi_vehicle" in signals or "multi_vehicle" in primary_type:
            actions.extend(["DISPATCH_REVIEW", "SAVE_EVIDENCE_BUNDLE"])
        if "emergency_vehicle" in signals:
            actions.extend(["ESCALATE_INCIDENT", "DISPATCH_REVIEW"])
        if "incident" in primary_type or "collision" in primary_type:
            actions.extend(["DISPATCH_REVIEW", "SAVE_EVIDENCE_BUNDLE"])
        if not actions:
            actions.append("LOG_EVENT")
        return list(dict.fromkeys(actions))

    def _derive_severity(self, primary_type: str, signals: list[str]) -> str:
        if "emergency_vehicle" in signals:
            return "high"
        if "multi_vehicle" in signals or "collision" in primary_type:
            return "high"
        if "lane_blocked" in signals:
            return "medium"
        return "low"

    def _build_explanation(
        self,
        primary_type: str,
        signals: list[str],
        actions: list[str],
        excerpt_count: int,
    ) -> str:
        lines = [
            f"Event classified as {primary_type} based on signals: {', '.join(signals) or 'none'}.",
            f"Recommended actions: {', '.join(actions)}.",
            f"Decision supported by {excerpt_count} relevant policy excerpt(s).",
        ]
        return " ".join(lines)
