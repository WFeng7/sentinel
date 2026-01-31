"""
Structured schemas for VLM event output.
Machine-readable for pipelines, RAG, and routing.
"""

from typing import Any

# Type taxonomies (for validation and RAG)
EVENT_CATEGORIES = ("incident", "traffic_condition", "non_event", "unknown")

INCIDENT_TYPES = (
    "rear_end_collision",
    "side_swipe_collision",
    "pedestrian_conflict",
    "near_miss",
    "debris_strike",
    "multi_vehicle_pileup",
    "rollover",
    "unknown_incident",
)

TRAFFIC_CONDITION_TYPES = (
    "congestion_stop_and_go",
    "lane_blockage",
    "stalled_vehicle",
    "construction_zone",
    "lane_closure_cones",
    "unprotected_left_turn_queue",
    "wrong_way_vehicle",
    "aggressive_merging_pattern",
    "unknown_traffic",
)

NON_EVENT_TYPES = (
    "shadow_false_positive",
    "occlusion_false_positive",
    "camera_artifact_glare",
    "detector_id_switch",
    "normal_braking",
    "unknown_non_event",
)

ALL_EVENT_TYPES = INCIDENT_TYPES + TRAFFIC_CONDITION_TYPES + NON_EVENT_TYPES

SEVERITY_LEVELS = ("none", "low", "medium", "high", "critical")

IMPACT_LEVELS = ("none", "low", "medium", "high")

ACTOR_ROLES = ("ego", "other")
ACTOR_CLASSES = ("car", "truck", "bus", "motorcycle", "pedestrian", "cyclist", "unknown")
RELATIVE_POSITIONS = (
    "front_left",
    "front",
    "front_right",
    "left",
    "right",
    "rear",
    "unknown",
)
LANE_RELATIONS = ("same_lane", "adjacent_lane", "crossing", "shoulder", "unknown")

# Action codes for playbooks
ACTION_CODES = (
    "DISPATCH_REVIEW",
    "SAVE_EVIDENCE_BUNDLE",
    "NOTIFY_FLEET_MANAGER",
    "TRAFFIC_ALERT",
    "LOG_EVENT",
    "ESCALATE_INCIDENT",
    "SUPPRESS_FALSE_POSITIVE",
    "ANNOTATE_MAP",
    "unknown",
)

PRIORITY_LEVELS = ("low", "medium", "high", "critical")


def validate_event_output(data: dict[str, Any]) -> dict[str, Any]:
    """
    Lightweight validation: ensure required fields exist and enums are valid.
    Returns the data with any fixes; raises ValueError on critical issues.
    """
    required_top = ("event_id", "event", "actors", "summary", "evidence", "recommended_actions", "rag")
    for k in required_top:
        if k not in data:
            raise ValueError(f"Missing required field: {k}")

    ev = data["event"]
    if "category" not in ev:
        ev["category"] = "unknown"
    if ev["category"] not in EVENT_CATEGORIES:
        ev["category"] = "unknown"
    if "type" not in ev:
        ev["type"] = "unknown_incident" if ev["category"] == "incident" else "unknown_non_event"
    if "severity" not in ev:
        ev["severity"] = "none"
    if ev["severity"] not in SEVERITY_LEVELS:
        ev["severity"] = "none"
    if "confidence" not in ev:
        ev["confidence"] = 0.0
    ev["confidence"] = max(0.0, min(1.0, float(ev["confidence"])))
    if "impact" not in ev:
        ev["impact"] = {"safety_risk": "none", "traffic_disruption": "none"}
    imp = ev["impact"]
    if "safety_risk" not in imp:
        imp["safety_risk"] = "none"
    if "traffic_disruption" not in imp:
        imp["traffic_disruption"] = "none"

    if not isinstance(data["actors"], list):
        data["actors"] = []
    if not isinstance(data["evidence"], list):
        data["evidence"] = []
    if not isinstance(data["recommended_actions"], list):
        data["recommended_actions"] = []
    if "rag" in data and not isinstance(data["rag"], dict):
        data["rag"] = {"tags": [], "queries": []}
    rag = data["rag"]
    if "tags" not in rag:
        rag["tags"] = []
    if "queries" not in rag:
        rag["queries"] = []

    return data
