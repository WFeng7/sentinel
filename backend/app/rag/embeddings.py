"""
Embedding utilities. Swappable for OpenAI/sentence-transformers later.
For mock documents, uses a simple deterministic text-to-vector.
"""

import hashlib
import re


DEFAULT_EMBEDDING_DIM = 384  # Common dimension for many models; OpenSearch-compatible


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


def simple_embed(text: str, dim: int = DEFAULT_EMBEDDING_DIM) -> list[float]:
    """
    Deterministic pseudo-embedding from text. For mock/dev only.
    Real deployments should use OpenAI, sentence-transformers, or OpenSearch ingest pipeline.
    """
    tokens = _tokenize(text)
    vec = [0.0] * dim
    for i, tok in enumerate(tokens):
        h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
        idx = h % dim
        vec[idx] += 1.0 / (i + 1)
    # L2 normalize so cosine similarity behaves
    norm = sum(x * x for x in vec) ** 0.5
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))
