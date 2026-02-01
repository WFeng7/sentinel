"""
VLM stage: semantic incident classifier + narrator.
Turns a suspicious CV event into structured output for pipelines and human dispatch.
"""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from event_schemas import validate_event_output

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

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


@dataclass
class EventContext:
    """Context passed from CV pipeline to VLM."""

    event_id: str
    camera_id: str = "cam00"
    fps: float = 30.0
    window_seconds: float = 10.0
    track_history: list[dict[str, Any]] = field(default_factory=list)
    keyframe_ts: list[float] = field(default_factory=list)
    bboxes_snapshot: list[dict[str, Any]] = field(default_factory=list)
    speeds_px_s: dict[str, float] = field(default_factory=dict)
    overlap_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    cv_notes: str = ""


def _build_user_prompt(ctx: EventContext) -> str:
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
        parts.extend(["## Overlapping track pairs (track_id_a, track_id_b, iou)", json.dumps(ctx.overlap_pairs[:10], indent=2), ""])
    if ctx.cv_notes:
        parts.extend(["## CV pipeline notes", ctx.cv_notes, ""])
    parts.extend([
        "## Task",
        "Classify this event. Output JSON only, following the schema:",
        SCHEMA_JSON,
    ])
    return "\n".join(parts)


def _encode_frame_b64(frame_bytes: bytes) -> str:
    return base64.standard_b64encode(frame_bytes).decode("ascii")


def _parse_json_from_response(text: str) -> dict[str, Any]:
    text = text.strip()
    # Remove markdown code fences if present
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


def render_human_narrative(data: dict[str, Any]) -> str:
    """Produce a crisp dispatch note from the structured output."""
    ev = data.get("event", {})
    summary = data.get("summary", {})
    actors = data.get("actors", [])
    actions = data.get("recommended_actions", [])

    cat = ev.get("category", "unknown")
    typ = ev.get("type", "unknown")
    sev = ev.get("severity", "none")
    conf = ev.get("confidence", 0.0)
    one_liner = summary.get("one_liner", "(no summary)")
    narrative = summary.get("narrative", "")

    lines = [
        f"**Incident:** {typ} (confidence {conf:.2f}), severity: {sev}",
        "",
        "**Involved:** " + "; ".join(
            f"Vehicle {a.get('track_id', '?')} ({a.get('class', '?')}, {a.get('lane_relation', '?')})"
            for a in actors[:5]
        ) or "None identified",
        "",
        "**What happened:** " + one_liner,
        "",
        "**Why we think so:** " + narrative,
        "",
        "**Next steps:** " + ", ".join(
            f"{a.get('code', '?')} ({a.get('priority', '?')})"
            for a in actions[:5]
        ) or "None",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# VLM Analyzer
# ---------------------------------------------------------------------------


class EventAnalyzer:
    """Analyzes flagged CV events via a vision-language model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.2",
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. pip install openai")
        return OpenAI(api_key=self._api_key or None, base_url=self._base_url)

    def analyze(
        self,
        ctx: EventContext,
        keyframes: list[tuple[float, bytes]] | None = None,
        clip_uri: str = "",
        bboxes_uri: str = "",
    ) -> dict[str, Any]:
        """
        Run VLM analysis on the event context and optional keyframe images.
        Returns validated structured output.
        """
        client = self._get_client()
        user_content: list[Any] = [{"type": "text", "text": _build_user_prompt(ctx)}]

        if keyframes:
            for ts, img_bytes in keyframes[:6]:  # Limit to 6 images for token budget
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{_encode_frame_b64(img_bytes)}"},
                })

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content
        data = _parse_json_from_response(text)

        # Fill in artifacts if provided
        if "artifacts" not in data:
            data["artifacts"] = {}
        if clip_uri:
            data["artifacts"]["clip_uri"] = clip_uri
        if bboxes_uri:
            data["artifacts"]["bboxes_uri"] = bboxes_uri
        if keyframes:
            data["artifacts"]["keyframes"] = [f"kf_{i}.jpg" for i in range(min(len(keyframes), 6))]

        return validate_event_output(data)

    def analyze_from_dict(
        self,
        context_dict: dict[str, Any],
        keyframes: list[tuple[float, bytes]] | None = None,
    ) -> dict[str, Any]:
        """Convenience: build EventContext from dict and analyze."""
        ctx = EventContext(
            event_id=context_dict.get("event_id", ""),
            camera_id=context_dict.get("camera_id", "cam00"),
            fps=float(context_dict.get("fps", 30)),
            window_seconds=float(context_dict.get("window_seconds", 10)),
            track_history=context_dict.get("track_history", []),
            keyframe_ts=context_dict.get("keyframe_ts", []),
            bboxes_snapshot=context_dict.get("bboxes_snapshot", []),
            speeds_px_s=context_dict.get("speeds_px_s", {}),
            overlap_pairs=context_dict.get("overlap_pairs", []),
            cv_notes=context_dict.get("cv_notes", ""),
        )
        return self.analyze(
            ctx,
            keyframes=keyframes,
            clip_uri=context_dict.get("clip_uri", ""),
            bboxes_uri=context_dict.get("bboxes_uri", ""),
        )


def generate_event_id(camera_id: str = "cam00") -> str:
    return f"evt_{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}_{camera_id}"


