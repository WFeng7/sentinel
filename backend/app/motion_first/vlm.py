"""
VLM analyzer: semantic incident classifier + narrator.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .event_schemas import VLMEventOutput
from .prompts import SYSTEM_PROMPT, SCHEMA_JSON, build_user_prompt, encode_frame_b64
import cv2
import numpy as np

from event_schemas import VLMEventOutput
from prompts import SYSTEM_PROMPT, SCHEMA_JSON, build_user_prompt, encode_frame_b64

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


def _parse_json_from_response(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)


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


def _normalize_frame_bytes(
    img_bytes: bytes,
    *,
    max_dim: int = 960,
    jpeg_quality: int = 85,
) -> bytes:
    """Light normalization before VLM to stabilize exposure and color."""
    buf = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        return img_bytes

    h, w = img.shape[:2]
    scale = min(1.0, max_dim / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    img = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    s_ch = np.clip(s_ch.astype(np.float32) * 1.15, 0, 255).astype(np.uint8)
    img = cv2.cvtColor(cv2.merge([h_ch, s_ch, v_ch]), cv2.COLOR_HSV2BGR)

    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    return enc.tobytes() if ok else img_bytes


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
                img_bytes = _normalize_frame_bytes(img_bytes)
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
        data = _parse_json_from_response(text)

        if "artifacts" not in data:
            data["artifacts"] = {}
        if clip_uri:
            data["artifacts"]["clip_uri"] = clip_uri
        if bboxes_uri:
            data["artifacts"]["bboxes_uri"] = bboxes_uri
        if keyframes:
            data["artifacts"]["keyframes"] = [f"kf_{i}.jpg" for i in range(min(len(keyframes), 6))]

        return VLMEventOutput.model_validate(data)

    def analyze_from_dict(
        self,
        context_dict: dict[str, Any],
        keyframes: list[tuple[float, bytes]] | None = None,
    ) -> VLMEventOutput:
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
