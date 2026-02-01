"""
Policy document source abstraction.
Switching from MockPolicyProvider to S3PolicyProvider requires no refactor outside this module.
"""

from abc import ABC, abstractmethod

from .schemas import PolicyDocument


class PolicyProvider(ABC):
    """Abstract interface for document ingestion sources."""

    @abstractmethod
    def fetch_documents(self) -> list[PolicyDocument]:
        """Fetch policy documents from the source. Called by the ingestion pipeline."""
        ...


class MockPolicyProvider(PolicyProvider):
    """Returns sample Providence policy text for development and testing."""

    def fetch_documents(self) -> list[PolicyDocument]:
        return [
            PolicyDocument(
                id="prov_incident_response_001",
                text=(
                    "When a traffic incident occurs on a state highway within Providence city limits, "
                    "the Rhode Island Department of Transportation (RIDOT) Traffic Management Center "
                    "shall be notified within 15 minutes. Incidents involving injuries, fatalities, "
                    "or hazmat require immediate escalation to RIDOT and Providence Emergency Management."
                ),
                metadata={
                    "city": "Providence",
                    "doc_type": "incident_response",
                    "source": "RIDOT_TMC_Procedures",
                    "section": "5.2",
                },
            ),
            PolicyDocument(
                id="prov_lane_blockage_001",
                text=(
                    "Lane blockage events—including stalled vehicles, debris, and construction—"
                    "require deployment of variable message signs (VMS) within 20 minutes when "
                    "traffic flow is reduced by more than one lane. Providence Public Works "
                    "coordinates with RIDOT for state highway segments."
                ),
                metadata={
                    "city": "Providence",
                    "doc_type": "lane_blockage",
                    "source": "Providence_Traffic_Ops",
                    "section": "3.1",
                },
            ),
            PolicyDocument(
                id="prov_multi_vehicle_001",
                text=(
                    "Multi-vehicle incidents involving three or more vehicles are classified "
                    "as high-priority. Providence PD and RIDOT shall establish a unified command. "
                    "Evidence preservation protocols apply; video footage from traffic cameras "
                    "shall be retained for a minimum of 72 hours."
                ),
                metadata={
                    "city": "Providence",
                    "doc_type": "multi_vehicle_incident",
                    "source": "Providence_PD_Traffic",
                    "section": "8.4",
                },
            ),
            PolicyDocument(
                id="prov_congestion_001",
                text=(
                    "Congestion and stop-and-go traffic on Providence arterials require "
                    "signal timing adjustments when average speed falls below 15 mph for "
                    "10 consecutive minutes. Non-emergency; routine TMC procedures apply."
                ),
                metadata={
                    "city": "Providence",
                    "doc_type": "congestion",
                    "source": "RIDOT_Signal_Ops",
                    "section": "2.3",
                },
            ),
            PolicyDocument(
                id="prov_emergency_vehicle_001",
                text=(
                    "Emergency vehicle response—flashing lights, ambulance, fire, or police—"
                    "takes precedence over all other traffic events. Operators shall flag "
                    "for human review and annotate the event. Do not suppress as false positive."
                ),
                metadata={
                    "city": "Providence",
                    "doc_type": "emergency_response",
                    "source": "RIDOT_TMC_Procedures",
                    "section": "5.1",
                },
            ),
        ]
