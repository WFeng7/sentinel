"""
Decision engine for traffic incident pipeline.
Consumes event_type_candidates, signals, city; returns structured decision + policy excerpts.
Uses LLM when API key available; falls back to rule-based otherwise.
"""

import os
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.utils import parse_json_from_llm
from .retriever import PolicyRetriever
from .schemas import RetrievedExcerpt, DecisionInput, DecisionOutput, SupportingExcerpt
from .prompt import build_decision_prompt

RAG_STATS = {"requests": 0, "doc_words": 0}


def _count_words(text: str) -> int:
    return len(str(text).split()) if text else 0


def _record_rag_stats(supporting: list[SupportingExcerpt]) -> None:
    RAG_STATS["requests"] += 1
    doc_words = 0
    for ex in supporting or []:
        doc_words += _count_words(getattr(ex, "text", ""))
    RAG_STATS["doc_words"] += doc_words


def get_rag_stats() -> dict[str, float]:
    reqs = RAG_STATS["requests"]
    doc_words = RAG_STATS["doc_words"]
    avg_doc_words = (doc_words / reqs) if reqs else 0
    return {
        "requests": reqs,
        "doc_words": doc_words,
        "avg_doc_words_per_request": avg_doc_words,
    }

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

        mode = os.environ.get("RAG_MODE", "llm").lower()
        if mode != "llm":
            explanation = f"Retrieved {len(supporting)} policy excerpts for context."
            output = DecisionOutput(
                decision={
                    "event_type_candidates": inp.event_type_candidates,
                    "signals": inp.signals,
                    "city": inp.city,
                },
                explanation=explanation,
                supporting_excerpts=supporting,
            )
            _record_rag_stats(supporting)
            return output

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY required for RAG decision engine")

        output = self._decide_with_llm(inp, excerpts, supporting)
        _record_rag_stats(supporting)
        return output

    def _decide_with_llm(
        self,
        inp: DecisionInput,
        excerpts: list[RetrievedExcerpt],
        supporting: list[SupportingExcerpt],
    ) -> DecisionOutput:
        """Use LLM to derive decision from event context and policy excerpts."""
        from openai import OpenAI
        max_excerpts = int(os.environ.get("RAG_MAX_EXCERPTS", str(self._top_k)))
        max_chars = int(os.environ.get("RAG_MAX_EXCERPT_CHARS", "1200"))

        def _truncate(text: str) -> str:
            if len(text) <= max_chars:
                return text
            return f"{text[:max_chars]}…"

        trimmed = excerpts[:max_excerpts]
        policy_text = "\n\n---\n\n".join(
            f"[{e.document_id}] {_truncate(e.text)}" for e in trimmed
        ) or "(No policy excerpts retrieved)"

        prompt = build_decision_prompt(
            city=inp.city,
            event_type_candidates=inp.event_type_candidates,
            signals=inp.signals,
            policy_text=policy_text,
        )

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
