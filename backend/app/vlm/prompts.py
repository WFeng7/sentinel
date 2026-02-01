SYSTEM_PROMPT = """You are an event-understanding system for traffic camera footage.

Your job is to classify and describe what happened in a short window of video.

You must output exactly one JSON object following the schema below.
No markdown fences. No explanations. No extra text before or after.

==============================
CRITICAL INSTRUCTIONS
==============================

1) PRIORITIZE SAFETY-CRITICAL SIGNALS FIRST.
Before choosing a category, you MUST explicitly check for:

- Flashing red and/or blue lights
- Emergency vehicle livery or light bars
- Vehicles using shoulder or abnormal lane traversal
- Police, fire, ambulance presence (If you're unsure, just generalize as "emergency vehicle")
- Fire trucks, ambulances, police cars, etc (may be hidden behind objects or vehicles)
- Sudden stoppage around a single focal vehicle

If any emergency vehicle with active lights is visible,
the event category MUST be "incident" and type must be
"emergency_vehicle_response" — even if congestion is also present.

Congestion secondary to an emergency response is NOT the primary label.

2) DO NOT default to "traffic_condition" unless you are confident
there are no emergency signals, collisions, debris, or abnormal behavior.

3) If an emergency vehicle is visible but the cause is unclear,
classify as:
  category: "incident"
  type: "emergency_vehicle_response"

4) Only classify as "traffic_condition" if:
- No flashing lights
- No abnormal vehicle positioning
- No emergency responders visible
- No signs of collision or hazard

5) If unsure whether lights are flashing:
- Look for red/blue reflections on nearby vehicles
- Look for asymmetric light intensity patterns
- Look for shoulder lane usage by a marked vehicle

==============================
STEP ORDER (MANDATORY)
==============================

Step A: Check for emergency vehicles or flashing lights.
Step B: Check for collision indicators.
Step C: Check for stalled vehicles or lane blockage.
Step D: If none of the above, classify traffic flow state.
Step E: If insufficient visual evidence, use "unknown".

==============================
CATEGORY DEFINITIONS
==============================

incident:
- collision
- near_miss
- pedestrian_conflict
- debris_strike
- emergency_vehicle_response
- vehicle_fire
- hazardous_material
- rollover
- disabled_vehicle_with_response

traffic_condition:
- congestion
- stalled_vehicle
- lane_blockage
- construction_zone
- wrong_way
- merge_bottleneck

non_event:
- shadow_false_positive
- detector_glitch
- normal_braking
- routine_lane_change

unknown:
- insufficient_visibility
- occlusion
- ambiguous_motion

==============================
SEVERITY GUIDANCE
==============================

critical: life-threatening collision, rollover, fire
high: active emergency response, major lane blockage
medium: stalled vehicle, heavy congestion
low: minor slowdown
none: non_event

==============================
CONFIDENCE GUIDANCE
==============================

Confidence must reflect:
- visibility quality
- motion consistency
- clarity of emergency signals
"""


# ---------------------------------------------------------------------------
# Schema (reference)
# ---------------------------------------------------------------------------

SCHEMA_JSON = """
{
  "event_id": "evt_YYYY-MM-DDTHH:MM:SSZ_camXX",
  "event": {
    "category": "incident|traffic_condition|non_event|unknown",
    "type": "<specific type from taxonomy>",
    "severity": "none|low|medium|high|critical",
    "confidence": 0.0,
    "impact": {
      "safety_risk": "none|low|medium|high",
      "traffic_disruption": "none|low|medium|high"
    }
  },
  "actors": [
    {
      "track_id": "t12",
      "role": "ego|other",
      "class": "car|truck|bus|motorcycle|pedestrian|cyclist|unknown",
      "relative_position": "front_left|front|front_right|left|right|rear|unknown",
      "lane_relation": "same_lane|adjacent_lane|crossing|shoulder|unknown"
    }
  ],
  "summary": {
    "one_liner": "Brief sentence.",
    "narrative": "2–4 sentences describing what changed and why it matters. Note any emergency vehicles you see. Look for flashing red and blue lights."
  },
  "timeline": {
    "t_minus": {"ts": 0.0, "state": "description"},
    "peak": {"ts": 3.2, "state": "description"},
    "t_plus": {"ts": 7.5, "state": "description"}
  },
  "evidence": [
    {
      "claim": "Human-readable claim",
      "signals": ["observable cue 1", "observable cue 2"],
      "supporting_frames": [{"ts": 3.1, "frame_id": "kf_peak_overlap"}]
    }
  ],
  "uncertainty": {
    "alternative_explanations": ["could be X", "could be Y"],
    "notes": "Optional clarification."
  },
  "recommended_actions": [
    {"code": "ACTION_CODE", "priority": "low|medium|high|critical"}
  ],
  "rag": {
    "tags": ["tag1", "tag2"],
    "queries": ["query for SOP retrieval 1", "query 2"]
  },
  "artifacts": {
    "clip_uri": "s3://.../evt...mp4",
    "keyframes": ["kf_pre.jpg", "kf_peak.jpg", "kf_post.jpg"],
    "bboxes_uri": "s3://.../bboxes.json"
  }
}
"""

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .analyzer import EventContext


# ---------------------------------------------------------------------------
# Prompt helpers
# ---------------------------------------------------------------------------


def build_user_prompt(ctx: "EventContext") -> str:
    """Build the user prompt from event context."""
    parts = [
        "## Context",
        f"- Event ID: {ctx.event_id}",
        f"- Camera: {ctx.camera_id}",
        f"- Window: {ctx.window_seconds}s @ {ctx.fps} fps",
        "",
        "## Tracking data (last N frames)",
    ]
    if ctx.track_history:
        parts.append(json.dumps(ctx.track_history[:20], indent=2))
    else:
        parts.append("(no track history provided)")
    parts.extend([
        "",
        "## Bboxes snapshot (peak frame)",
        json.dumps(ctx.bboxes_snapshot[:30], indent=2) if ctx.bboxes_snapshot else "(none)",
        "",
        "## Speeds (px/s) at peak",
        json.dumps(ctx.speeds_px_s, indent=2) if ctx.speeds_px_s else "(none)",
        "",
    ])
    if ctx.overlap_pairs:
        parts.extend([
            "## Overlapping track pairs (track_id_a, track_id_b, iou)",
            json.dumps(ctx.overlap_pairs[:10], indent=2),
            "",
        ])
    if ctx.cv_notes:
        parts.extend(["## CV pipeline notes", ctx.cv_notes, ""])
    parts.extend([
        "## Task",
        "Classify this event. Output JSON only, following the schema:",
        SCHEMA_JSON,
    ])
    return "\n".join(parts)