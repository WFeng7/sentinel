#TODO: Consider doing some sort of normalization before passing to VLM to better see emergency vehicles
"""
VLM analyzer: semantic incident classifier + narrator.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.utils import parse_json_from_llm
from .event_schemas import VLMEventOutput
from .prompts import SYSTEM_PROMPT, SCHEMA_JSON, build_user_prompt
from app.utils import encode_frame_b64

# from stage 1
@dataclass
class EventContext:
    event_id: str = ""
    camera_id: str = "cam00"
    fps: float = 30.0
    window_seconds: float = 10.0
    track_history: list[dict[str, Any]] = field(default_factory=list)
    keyframe_ts: list[float] = field(default_factory=list)
    bboxes_snapshot: list[dict[str, Any]] = field(default_factory=list)
    speeds_px_s: dict[str, float] = field(default_factory=dict)
    overlap_pairs: list[tuple[str, str, float]] = field(default_factory=list)
    cv_notes: str = ""


def render_human_narrative(data: VLMEventOutput) -> str:
    """Produce a crisp dispatch note from the structured output."""
    ev = data.event
    summary = data.summary
    actors = data.actors
    actions = data.recommended_actions

    typ = ev.type
    sev = ev.severity.value
    conf = ev.confidence
    one_liner = summary.get("one_liner", "(no summary)")
    narrative = summary.get("narrative", "")

    # Handle involved actors - show "None" if no actors
    involved_text = "None identified"
    if actors:
        actor_list = []
        for a in actors[:5]:
            actor_desc = f"Vehicle {a.get('track_id', '?')} ({a.get('class', '?')}, {a.get('lane_relation', '?')})"
            actor_list.append(actor_desc)
        if actor_list:
            involved_text = "; ".join(actor_list)

    lines = [
        f"**Incident:** {typ} (confidence {conf:.2f}), severity: {sev}",
        "",
        f"**Involved:** {involved_text}",
        "",
        f"**What happened:** {one_liner}",
        "",
        f"**Why we think so:** {narrative}",
        "",
        "**Next steps:** " + ", ".join(
            f"{a.get('code', '?')} ({a.get('priority', '?')})"
            for a in actions[:5]
        ) or "None",
    ]
    return "\n".join(lines)


class EventAnalyzer:
    """Analyzes flagged CV events via a vision-language model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gpt-5.1",
        base_url: str | None = None,
    ):
        self.model = model
        self._api_key = api_key
        self._base_url = base_url
        print(f"[OpenAI VLM] Initialized with model: {model}")
        print(f"[OpenAI VLM] Using API key: {self._api_key}")
        print(f"[OpenAI VLM] Base URL: {self._base_url or 'default'}")

    def _get_client(self):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("openai package required. pip install openai")
        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def analyze(
        self,
        ctx: EventContext,
        keyframes: list[tuple[float, bytes]] | None = None,
        clip_uri: str = "",
        bboxes_uri: str = "",
    ) -> VLMEventOutput:
        """
        Run VLM analysis on the event context and optional keyframe images.
        Returns validated Pydantic model.
        """
        if not self._api_key:
            raise ValueError("OPENAI_API_KEY required for VLM analysis")
        client = self._get_client()
        user_content: list[Any] = [{"type": "text", "text": build_user_prompt(ctx)}]

        if keyframes:
            for ts, img_bytes in keyframes[:6]:
                user_content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encode_frame_b64(img_bytes)}"},
                })

        resp = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
        )
        text = resp.choices[0].message.content
        data = parse_json_from_llm(text)

        if "artifacts" not in data:
            data["artifacts"] = {}
        if clip_uri:
            data["artifacts"]["clip_uri"] = clip_uri
        if bboxes_uri:
            data["artifacts"]["bboxes_uri"] = bboxes_uri
        if keyframes:
            data["artifacts"]["keyframes"] = [f"kf_{i}.jpg" for i in range(min(len(keyframes), 6))]

        return VLMEventOutput.model_validate(data)
