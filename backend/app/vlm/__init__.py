"""
VLM stage: semantic incident classifier + narrator.
Turns CV events into structured output for pipelines and human dispatch.
"""

from .analyzer import EventAnalyzer
from .event_schemas import validate_event_output
from .analyzer import render_human_narrative, EventContext

__all__ = [
    "EventAnalyzer",
    "EventContext",
    "render_human_narrative",
    "validate_event_output",
]
