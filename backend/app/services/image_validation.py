"""Shared image validation (Phase 1 rules, reused by Phase 2).

Moved verbatim from the case-photos router so sighting-photo uploads
enforce byte-identical rules: size cap, magic-byte detection, and
Pillow verification with a content/type cross-check. The client's
filename and declared Content-Type are never trusted.
"""

import io

from fastapi import HTTPException, status
from PIL import Image, UnidentifiedImageError

# Detected from magic bytes — the client's filename and declared
# Content-Type are never trusted.
ALLOWED_TYPES = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
}

PILLOW_FORMATS = {
    "JPEG": ("image/jpeg", "jpg"),
    "PNG": ("image/png", "png"),
    "WEBP": ("image/webp", "webp"),
}


def detect_mime(data: bytes) -> str | None:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_image(data: bytes) -> tuple[str, str, int, int]:
    """Return (mime_type, ext, width, height) or raise HTTPException."""
    from app.core.config import settings

    if not data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty",
        )
    if len(data) > settings.max_photo_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds the maximum allowed size",
        )
    mime_type = detect_mime(data)
    if mime_type is None:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Only JPEG, PNG, and WebP images are accepted",
        )
    try:
        with Image.open(io.BytesIO(data)) as probe:
            probe.verify()
        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            pillow_mime, pillow_ext = PILLOW_FORMATS[probe.format]
    except (UnidentifiedImageError, KeyError, OSError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is not a readable image",
        )
    if (mime_type, ALLOWED_TYPES[mime_type]) != (pillow_mime, pillow_ext):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image content does not match its detected type",
        )
    return mime_type, ALLOWED_TYPES[mime_type], width, height
