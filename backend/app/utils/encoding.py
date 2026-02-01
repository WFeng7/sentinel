"""Encoding utilities."""

import base64


def encode_frame_b64(frame_bytes: bytes) -> str:
    """Encode frame bytes to base64 for use in image URLs."""
    return base64.standard_b64encode(frame_bytes).decode("ascii")
