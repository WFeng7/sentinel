"""
Policy document source abstraction.
- MockPolicyProvider: hardcoded dev data
- LocalDataProvider: local ./data/ folder (PDFs, etc.)
- S3PolicyProvider: future AWS S3
"""

from abc import ABC, abstractmethod
from pathlib import Path

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


class LocalDataProvider(PolicyProvider):
    """
    Loads policy documents from a local directory (e.g. ./data/).
    Uses LlamaIndex SimpleDirectoryReader for PDF/text. Fallback to MockPolicyProvider if empty.
    """

    def __init__(self, data_dir: str | Path | None = None):
        self._data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"

    def fetch_documents(self) -> list[PolicyDocument]:
        """Load documents from data_dir. Returns MockPolicyProvider data if empty or load fails."""
        try:
            from llama_index.core import SimpleDirectoryReader
        except ImportError:
            return MockPolicyProvider().fetch_documents()

        if not self._data_dir.exists():
            return MockPolicyProvider().fetch_documents()

        try:
            reader = SimpleDirectoryReader(
                input_dir=str(self._data_dir),
                required_exts=[".pdf", ".txt", ".md"],
                recursive=False,
            )
            raw_docs = reader.load_data()
        except Exception:
            return MockPolicyProvider().fetch_documents()

        if not raw_docs:
            return MockPolicyProvider().fetch_documents()

        policy_docs = []
        for i, doc in enumerate(raw_docs):
            text = getattr(doc, "text", str(doc))
            metadata = getattr(doc, "metadata", {}) or {}
            fpath = metadata.get("file_path") or metadata.get("filename") or metadata.get("file_name")
            stem = Path(fpath).stem if fpath else f"doc_{i}"
            doc_id = f"local_{i}_{stem}"[:80]
            policy_docs.append(
                PolicyDocument(
                    id=doc_id,
                    text=text,
                    metadata={
                        "source": "local",
                        "filename": Path(fpath).name if fpath else doc_id,
                        **{k: v for k, v in metadata.items() if k not in ("file_path", "file_name")},
                    },
                )
            )
        return policy_docs
