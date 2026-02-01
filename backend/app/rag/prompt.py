DECISION_SCHEMA = """
{
  "event_type": "primary event type from candidates (e.g. lane_blockage, multi_vehicle_incident)",
  "recommended_actions": ["ACTION_CODE", "..."],
  "severity": "none|low|medium|high|critical",
  "explanation": "2-4 sentences. MUST cite excerpt IDs like [DOC123]. MUST mention at least 1 concrete policy detail.",
  "citations": ["DOC123", "DOC9"],
  "policy_facts": ["<verbatim or near-verbatim policy fragment used>"]
}
"""

ACTION_CODES = (
    "DISPATCH_REVIEW, SAVE_EVIDENCE_BUNDLE, TRAFFIC_ALERT, ANNOTATE_MAP, "
    "LOG_EVENT, ESCALATE_INCIDENT, NOTIFY_FLEET_MANAGER"
)

DECISION_PROMPT_TEMPLATE = """You are a traffic incident decision assistant for {city}.

Given the event context and relevant policy excerpts, produce a structured decision.

## Event context
- Event type candidates: {event_type_candidates}
- Signals: {signals}
- City: {city}

## Relevant policy excerpts
{policy_text}

## Task
Output exactly one JSON object (no markdown, no extra text) following this schema:
{schema}

Action codes: {action_codes}

Hard requirements:
- Pick event_type from the candidates exactly.
- In "explanation", include citations in brackets to the document IDs you used, e.g. [DOC123].
- "citations" MUST be a list of document IDs that appear in the provided excerpts (subset of them).
- "policy_facts" MUST include at least one verbatim or near-verbatim policy fragment drawn from the cited excerpt(s).
- Mention at least one concrete policy detail (threshold, required action, role, timeframe, lane closure rule, dispatch rule, etc.).
- If policy excerpts are empty or insufficient to justify a decision, set:
  - severity="low"
  - recommended_actions=["LOG_EVENT"]
  - citations=[]
  - policy_facts=[]
  - explanation="No relevant policy excerpt was retrieved for this event."

Severity guidance:
- critical for life-threatening
- high for emergency response or multi-vehicle
- medium for lane blockage or stalled vehicle
- low for minor
- none for non-events
"""


def build_decision_prompt(
    *,
    city: str,
    event_type_candidates: list[str],
    signals: list[str],
    policy_text: str,
) -> str:
    return DECISION_PROMPT_TEMPLATE.format(
        city=city or "unknown",
        event_type_candidates=event_type_candidates or [],
        signals=signals or [],
        policy_text=policy_text or "(No policy excerpts retrieved)",
        schema=DECISION_SCHEMA,
        action_codes=ACTION_CODES,
    )
