"""
Decision engine for traffic incident pipeline.
Consumes event_type_candidates, signals, city; returns structured decision + policy excerpts.
Uses LLM when API key available; falls back to rule-based otherwise.
"""

from __future__ import annotations

import os
import re
from typing import Any

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


_GENERIC_PATTERNS = [
    r"\brequires?\s+monitoring\b",
    r"\bshould\s+be\s+monitored\b",
    r"\bneeds?\s+monitoring\b",
    r"\bmonitor(?:ing)?\s+is\s+(?:required|needed|recommended)\b",
    r"\bcontinue(?:d)?\s+monitoring\b",
    r"\bongoing\s+monitoring\b",
    r"\bas\s+a\s+precaution\b",
    r"\bout\s+of\s+an\s+abundance\s+of\s+caution\b",
]


def _looks_generic(text: str) -> bool:
    t = " ".join((text or "").split()).strip().lower()
    if not t:
        return True
    return any(re.search(p, t) for p in _GENERIC_PATTERNS)


def _mentions_concrete_policy_detail(text: str) -> bool:
    t = text or ""
    if re.search(r"\b(shall|must|required|prohibited)\b", t, re.I):
        return True
    if re.search(r"\b(minutes?|hours?|mph|km/h|feet|meters?)\b", t, re.I):
        return True
    if re.search(r"\b(level\s*[1-5]|priority)\b", t, re.I):
        return True
    if re.search(r"\b(lane|shoulder|closure|detour|tow|dispatch|incident commander)\b", t, re.I):
        return True
    return False


def _fallback_excerpt_summary(supporting: list[SupportingExcerpt]) -> str:
    if not supporting:
        return "No relevant policy excerpt was retrieved for this event."
    best = max(supporting, key=lambda x: (x.score or 0))
    return (best.text or "").strip()[:400]


def _normalize_retrieval_query(inp: DecisionInput) -> str:
    # Convert internal labels into a more natural retrieval query for embeddings.
    candidates = ", ".join(inp.event_type_candidates or []) or "unknown"
    signals = ", ".join(inp.signals or []) or "none"
    city = inp.city or "any"
    return (
        f"Traffic incident response policy. City: {city}. "
        f"Event candidates: {candidates}. Signals observed: {signals}. "
        "Find policy rules that map these signals to actions, severity, dispatch, alerts, or escalation."
    )


class DecisionEngine:
    """
    Consumes event context, retrieves relevant policies, produces structured decision.
    Uses LLM when available; otherwise returns rule-based output with excerpts.
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
        query = _normalize_retrieval_query(inp)

        excerpts: list[RetrievedExcerpt] = self._retriever.retrieve(
            query=query,
            top_k=self._top_k,
            city=inp.city,
            doc_type=None,
        )

        # The retriever already falls back internally, but keep this extra safety:
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
            output = DecisionOutput(
                decision={
                    "event_type": inp.event_type_candidates[0] if inp.event_type_candidates else "unknown",
                    "event_type_candidates": inp.event_type_candidates,
                    "signals": inp.signals,
                    "city": inp.city,
                    "recommended_actions": ["LOG_EVENT"],
                    "severity": "low",
                },
                explanation=_fallback_excerpt_summary(supporting) if supporting else "",
                supporting_excerpts=supporting,
            )
            output.explanation = self._sanitize_explanation(output.explanation, supporting)
            _record_rag_stats(supporting)
            return output

        # If we have no excerpts, do NOT ask the LLM to hallucinate a policy-grounded summary.
        if not excerpts:
            output = DecisionOutput(
                decision={
                    "event_type": inp.event_type_candidates[0] if inp.event_type_candidates else "unknown",
                    "event_type_candidates": inp.event_type_candidates,
                    "signals": inp.signals,
                    "city": inp.city,
                    "recommended_actions": ["LOG_EVENT"],
                    "severity": "low",
                },
                explanation="No relevant policy excerpt was retrieved for this event.",
                supporting_excerpts=[],
            )
            _record_rag_stats([])
            return output

        if not self._api_key:
            raise ValueError("OPENAI_API_KEY required for RAG decision engine")

        output = self._decide_with_llm(inp, excerpts, supporting)

        # Optional: allow a second synthesis pass ONLY if you validate grounding.
        # Default off to avoid overwriting good grounded explanations.
        if os.environ.get("RAG_ENABLE_SYNTH", "0").strip() == "1":
            synthesized = self._synthesize_explanation(inp, supporting, allowed_ids={e.document_id for e in excerpts})
            if synthesized:
                output.explanation = synthesized

        output.explanation = self._sanitize_explanation(output.explanation, supporting)
        _record_rag_stats(supporting)
        return output

    def _decide_with_llm(
        self,
        inp: DecisionInput,
        excerpts: list[RetrievedExcerpt],
        supporting: list[SupportingExcerpt],
    ) -> DecisionOutput:
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
            city=inp.city or "unknown",
            event_type_candidates=inp.event_type_candidates or [],
            signals=inp.signals or [],
            policy_text=policy_text,
        )
        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model=self._llm_model,
            messages=[{"role": "user", "content": prompt}],
        )

        text = resp.choices[0].message.content or ""
        data = parse_json_from_llm(text) or {}

        primary_type = data.get("event_type") or (
            inp.event_type_candidates[0] if inp.event_type_candidates else "unknown"
        )

        actions = data.get("recommended_actions") or ["LOG_EVENT"]
        if not isinstance(actions, list):
            actions = [actions] if actions else ["LOG_EVENT"]

        severity = data.get("severity") or "low"
        explanation = data.get("explanation") or ""

        # Grounding fields
        citations = data.get("citations") or []
        if isinstance(citations, str):
            citations = [citations]
        policy_facts = data.get("policy_facts") or []
        if isinstance(policy_facts, str):
            policy_facts = [policy_facts]

        allowed_ids = {e.document_id for e in trimmed}
        citations = [c for c in citations if c in allowed_ids]

        # Enforce grounding: must cite at least one allowed doc and include at least one policy fact.
        grounded = bool(citations) and bool(policy_facts)

        decision: dict[str, Any] = {
            "event_type": primary_type,
            "event_type_candidates": inp.event_type_candidates,
            "signals": inp.signals,
            "city": inp.city,
            "recommended_actions": actions,
            "severity": severity,
            "citations": citations,
            "policy_facts": policy_facts,
        }

        if not grounded:
            explanation = _fallback_excerpt_summary(supporting)

        explanation = self._sanitize_explanation(explanation, supporting)

        return DecisionOutput(
            decision=decision,
            explanation=explanation,
            supporting_excerpts=supporting,
        )

    def _sanitize_explanation(
        self,
        explanation: str,
        supporting: list[SupportingExcerpt],
    ) -> str:
        cleaned = " ".join((explanation or "").split()).strip()
        if not cleaned:
            return _fallback_excerpt_summary(supporting)

        # Require at least one doc-id citation token when we have supporting excerpts.
        if supporting:
            has_any_citation = any(f"[{ex.document_id}]" in cleaned for ex in supporting[:10])
            if not has_any_citation:
                return _fallback_excerpt_summary(supporting)

        if _looks_generic(cleaned) or not _mentions_concrete_policy_detail(cleaned):
            return _fallback_excerpt_summary(supporting)

        return cleaned

    def _synthesize_explanation(
        self,
        inp: DecisionInput,
        supporting: list[SupportingExcerpt],
        *,
        allowed_ids: set[str],
    ) -> str:
        # Safe synth: must cite doc ids; validated like everything else.
        if not self._api_key or not supporting:
            return ""

        max_chars = int(os.environ.get("RAG_SYNTH_EXCERPT_CHARS", "500"))
        max_excerpts = int(os.environ.get("RAG_MAX_EXCERPTS", str(self._top_k)))
        excerpts_text = "\n\n".join(
            f"[{ex.document_id}] {ex.text[:max_chars]}" for ex in supporting[:max_excerpts]
        )
        if not excerpts_text:
            return ""

        prompt = (
            "Write a concise 1-2 sentence explanation grounded in the policy excerpts.\n"
            "Hard requirements:\n"
            "- MUST include citations like [DOC123] referencing the excerpt IDs provided.\n"
            "- MUST mention at least one concrete policy detail from a cited excerpt.\n"
            "Return only the explanation text.\n\n"
            f"Event type candidates: {', '.join(inp.event_type_candidates) or 'unknown'}\n"
            f"Signals: {', '.join(inp.signals) or 'none'}\n\n"
            f"Policy excerpts:\n{excerpts_text}\n"
        )
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        text = (resp.choices[0].message.content or "").strip()

        # Validate synth output: must cite at least one allowed id and not be generic.
        if not text or _looks_generic(text):
            return ""

        if not any(f"[{doc_id}]" in text for doc_id in allowed_ids):
            return ""

        if not _mentions_concrete_policy_detail(text):
            return ""

        return text
