DECISION_SCHEMA = """
{
  "event_type": "primary event type from candidates (e.g. lane_blockage, multi_vehicle_incident)",
  "recommended_actions": ["ACTION_CODE", "..."],
  "severity": "none|low|medium|high|critical",
  "explanation": "2-4 sentence human-readable explanation grounding the decision in the policy excerpts"
}
"""

ACTION_CODES = "DISPATCH_REVIEW, SAVE_EVIDENCE_BUNDLE, TRAFFIC_ALERT, ANNOTATE_MAP, LOG_EVENT, ESCALATE_INCIDENT, NOTIFY_FLEET_MANAGER"

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
Ground your decision in the policy excerpts. Pick event_type from the candidates. Use severity: critical for life-threatening, high for emergency response or multi-vehicle, medium for lane blockage or stalled vehicle, low for minor, none for non-events.
"""


def build_decision_prompt(
    *,
    city: str,
    event_type_candidates: list[str],
    signals: list[str],
    policy_text: str,
) -> str:
    """Build the decision prompt with the given context."""
    return DECISION_PROMPT_TEMPLATE.format(
        city=city,
        event_type_candidates=event_type_candidates,
        signals=signals,
        policy_text=policy_text,
        schema=DECISION_SCHEMA,
        action_codes=ACTION_CODES,
    )