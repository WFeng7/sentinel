"""Shared utilities for backend modules."""

from .embeddings import DEFAULT_EMBEDDING_DIM, cosine_similarity, simple_embed
from .encoding import encode_frame_b64
from .json_utils import parse_json_from_llm

__all__ = [
    "DEFAULT_EMBEDDING_DIM",
    "cosine_similarity",
    "encode_frame_b64",
    "parse_json_from_llm",
    "simple_embed",
]
