"""
Gemini VLM analyzer: semantic incident classifier + narrator.
Uses Google Gemini API instead of OpenAI.
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

@dataclass
class EventContext:
    """Event context for Gemini analysis."""
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

class GeminiEventAnalyzer:
    """Analyzes flagged CV events via Google Gemini vision-language model."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
    ):
        self.model = model
        self._api_key = api_key or os.environ.get('GEMINI_API_KEY')
        print(f"[Gemini VLM] Initialized with model: {model}")
        print(f"[Gemini VLM] Using API key: {self._api_key[:20]}..." if self._api_key else "[Gemini VLM] No API key found")

    def _get_client(self):
        try:
            import google.generativeai as genai
        except ImportError:
            raise ImportError("google-generativeai package required. pip install google-generativeai")
        
        genai.configure(api_key=self._api_key)
        return genai.GenerativeModel(self.model)

    def analyze(
        self,
        ctx: EventContext,
        keyframes: list[tuple[float, bytes]] | None = None,
        clip_uri: str = "",
        bboxes_uri: str = "",
    ) -> VLMEventOutput:
        """Analyze event using Gemini."""
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY required for VLM analysis")
        
        client = self._get_client()
        
        # Prepare content
        user_content: list[Any] = [{"type": "text", "text": build_user_prompt(ctx)}]
        
        # Add keyframes if provided
        if keyframes:
            for ts, frame_bytes in keyframes[:3]:  # Limit to 3 keyframes
                import base64
                b64_data = base64.b64encode(frame_bytes).decode()
                user_content.append({
                    "type": "image_url", 
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_data}"
                    }
                })
        
        # Build the prompt
        prompt_parts = []
        for content in user_content:
            if content["type"] == "text":
                prompt_parts.append(content["text"])
            elif content["type"] == "image_url":
                # For Gemini, we need to handle images differently
                import base64
                b64_data = content["image_url"]["url"].split(",")[1]
                import PIL.Image
                import io
                image_data = base64.b64decode(b64_data)
                image = PIL.Image.open(io.BytesIO(image_data))
                prompt_parts.append(image)
        
        try:
            # Generate response
            response = client.generate_content(prompt_parts)
            response_text = response.text
            
            # Parse the response
            result = parse_json_from_llm(response_text, SCHEMA_JSON)
            
            # Convert to VLMEventOutput format
            return VLMEventOutput.model_validate(result)
            
        except Exception as e:
            print(f"[Gemini] Analysis failed: {e}")
            # Return a default response
            return VLMEventOutput(
                event={
                    "type": "unknown",
                    "severity": "low",
                    "confidence": 0.1,
                    "duration_seconds": ctx.window_seconds
                },
                summary={
                    "one_liner": f"Analysis failed for {ctx.camera_id}",
                    "narrative": f"Unable to analyze due to error: {str(e)}"
                },
                actors=[],
                recommended_actions=[],
                rag={
                    "tags": ["analysis_failed"],
                    "queries": []
                }
            )

# Alias for compatibility
EventAnalyzer = GeminiEventAnalyzer
