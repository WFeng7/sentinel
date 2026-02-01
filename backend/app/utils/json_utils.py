"""JSON parsing utilities for LLM responses."""

import json
import re
from typing import Any


def parse_json_from_llm(text: str) -> dict[str, Any]:
    """
    Parse JSON from LLM output, stripping markdown code fences if present.
    """
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    return json.loads(text)
