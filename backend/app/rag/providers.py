#TODO: fill in correct S3 bucket and prefix

"""
Policy document source abstraction.
- S3PolicyProvider: AWS S3 (dummy placeholder)
- MockPolicyProvider: minimal dev fallback
"""

from abc import ABC, abstractmethod

from .schemas import PolicyDocument


class S3PolicyProvider(PolicyProvider):
    """
    Fetch policy documents from S3.
    Dummy implementation: placeholder config, returns empty until wired to boto3.
    """

    def __init__(
        self,
        *,
        bucket: str = "sentinel-policy-docs",
        prefix: str = "policy/",
        region: str = "us-east-1",
    ):
        self._bucket = bucket
        self._prefix = prefix
        self._region = region
        # Placeholder: s3://bucket/prefix
        self._uri = f"s3://{bucket}/{prefix}"

    def fetch_documents(self) -> list[PolicyDocument]:
        """TODO: boto3 list_objects_v2, get_object, parse PDFs. Returns empty for now."""
        # Dummy: would use boto3.client("s3").list_objects_v2(Bucket=..., Prefix=...)
        return []


class MockPolicyProvider(PolicyProvider):
    """Minimal hardcoded docs for dev/testing."""

    def fetch_documents(self) -> list[PolicyDocument]:
        return [
            PolicyDocument(
                id="mock_001",
                text="Traffic incidents on state highways require RIDOT notification within 15 minutes.",
                metadata={"city": "Providence", "doc_type": "incident_response"},
            ),
            PolicyDocument(
                id="mock_002",
                text="Lane blockages require VMS deployment within 20 minutes when flow is reduced.",
                metadata={"city": "Providence", "doc_type": "lane_blockage"},
            ),
        ]
