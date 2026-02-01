#TODO: fill in correct S3 bucket and prefix

"""
- PolicyProvider: ABC for document sources
- S3PolicyProvider: AWS S3
- MockPolicyProvider: minimal dev fallback
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path

from .schemas import PolicyDocument


class PolicyProvider(ABC):
    """Abstract base for policy document sources."""

    @abstractmethod
    def fetch_documents(self) -> list[PolicyDocument]:
        ...


class S3PolicyProvider(PolicyProvider):
    """Fetch policy documents from S3. Supports PDF, txt, md."""

    def __init__(
        self,
        *,
        bucket: str = "sentinel-policy-docs",
        prefix: str = "policy/",
        region: str = "us-east-1",
    ):
        self._bucket = bucket
        self._prefix = (prefix.rstrip("/") + "/") if prefix else ""
        self._region = region

    def fetch_documents(self) -> list[PolicyDocument]:
        """List S3 objects, download, parse (PDF/txt/md), return PolicyDocuments."""
        import boto3

        client = boto3.client("s3", region_name=self._region)
        paginator = client.get_paginator("list_objects_v2")
        docs: list[PolicyDocument] = []
        seen: set[str] = set()

        for page in paginator.paginate(Bucket=self._bucket, Prefix=self._prefix):
            for obj in page.get("Contents") or []:
                key = obj["Key"]
                if key.endswith("/"):
                    continue
                ext = Path(key).suffix.lower()
                if ext not in (".pdf", ".txt", ".md"):
                    continue
                doc_id = key.replace("/", "_").replace(" ", "_")[:80]
                if doc_id in seen:
                    continue
                seen.add(doc_id)

                try:
                    resp = client.get_object(Bucket=self._bucket, Key=key)
                    body = resp["Body"].read()
                except Exception:
                    continue

                text = self._parse_content(body, ext, key)
                if not text:
                    continue

                docs.append(
                    PolicyDocument(
                        id=doc_id,
                        text=text,
                        metadata={"source": "s3", "key": key, "bucket": self._bucket},
                    )
                )

        return docs

    def _parse_content(self, body: bytes, ext: str, key: str) -> str:
        if ext == ".txt" or ext == ".md":
            return body.decode("utf-8", errors="replace")
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                from io import BytesIO

                reader = PdfReader(BytesIO(body))
                return "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception:
                return ""
        return ""


class MockPolicyProvider(PolicyProvider):
    """Minimal hardcoded docs for dev/testing."""

    def fetch_documents(self) -> list[PolicyDocument]:
        base_dir = Path(__file__).resolve().parents[2] / "data"
        local_path = Path(os.environ.get("RAG_LOCAL_PATH", str(base_dir)))
        docs: list[PolicyDocument] = []

        if local_path.exists() and local_path.is_dir():
            for path in sorted(local_path.iterdir()):
                if not path.is_file():
                    continue
                ext = path.suffix.lower()
                if ext not in (".pdf", ".txt", ".md"):
                    continue
                try:
                    body = path.read_bytes()
                except Exception:
                    continue
                text = self._parse_content(body, ext, path)
                if not text:
                    continue
                doc_id = path.name.replace(" ", "_")[:80]
                docs.append(
                    PolicyDocument(
                        id=doc_id,
                        text=text,
                        metadata={"source": "local", "path": str(path)},
                    )
                )

        if docs:
            return docs

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

    def _parse_content(self, body: bytes, ext: str, path: Path) -> str:
        if ext == ".txt" or ext == ".md":
            return body.decode("utf-8", errors="replace")
        if ext == ".pdf":
            try:
                from pypdf import PdfReader
                from io import BytesIO

                reader = PdfReader(BytesIO(body))
                return "\n\n".join(p.extract_text() or "" for p in reader.pages)
            except Exception as exc:
                print(f"[rag] Failed to parse PDF: {path.name} ({exc})")
                return ""
        return ""
