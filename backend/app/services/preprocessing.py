"""Shared deterministic image preprocessing (Phase 3).

Pure byte transformation shared by CasePhoto and SightingPhoto. This
module deliberately knows nothing about the database, authorization,
storage keys, users, cases, or sightings: it maps validated original
image bytes to one normalized derived rendition.

Normalization contract (phase3-v1):
    - EXIF orientation corrected (Pillow transpose operation)
    - normalized to RGB, alpha flattened onto a WHITE background
    - resized only when the long edge exceeds 1024 px (aspect kept,
      never upscaled) using Lanczos resampling
    - metadata/EXIF stripped from the output
    - encoded as JPEG quality 85 with pinned encoder flags

Determinism: same original bytes + same processor version + same
parameters produce bit-identical derived bytes and therefore the same
derived SHA-256. No timestamps, random values, or UUIDs are generated.
"""

import hashlib
import io

from PIL import Image, ImageOps

# Processor version pinned into every derived artifact's metadata. Bump
# this (e.g. phase3-v2) whenever the contract above changes so a new
# derived SHA/key is produced instead of silently mixing renditions.
PROCESSOR_VERSION = "phase3-v1"

# Output contract constants (single source of truth for Phase 3).
DERIVED_MIME_TYPE = "image/jpeg"
DERIVED_EXTENSION = "jpg"
DERIVED_JPEG_QUALITY = 85
DERIVED_MAX_LONG_EDGE = 1024

# Recovery threshold for an interrupted retry: a photo left in
# PROCESSING longer than this is considered stale and may be
# reprocessed. Short enough for operator recovery, long enough that a
# normal synchronous attempt (milliseconds to seconds) never trips it.
STALE_PROCESSING_THRESHOLD_SECONDS = 600


class PreprocessingError(Exception):
    """Raised when original bytes cannot be normalized.

    Carries only a safe, user-presentable message: routers persist
    ``str(exc)`` as ``processing_error``, so internals (paths, SQL,
    tracebacks) must never be embedded here.
    """


def preprocess(image_bytes: bytes) -> tuple[bytes, str, int, int, str, str]:
    """Normalize original bytes into the Phase 3 derived rendition.

    Returns (derived_bytes, derived_sha256, width, height, mime_type,
    processor_version). Raises PreprocessingError on any failure.
    """
    if not image_bytes:
        raise PreprocessingError("Image preprocessing failed: empty file")
    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            image = ImageOps.exif_transpose(src)
            image.load()
            if image.mode in ("RGBA", "LA"):
                background = Image.new("RGB", image.size, (255, 255, 255))
                background.paste(image, mask=image.split()[-1])
                image = background
            elif image.mode == "P" and "transparency" in image.info:
                background = Image.new(
                    "RGB", image.size, (255, 255, 255)
                )
                background.paste(image, mask=image.convert("RGBA").split()[-1])
                image = background
            else:
                image = image.convert("RGB")
            width, height = image.size
            long_edge = max(width, height)
            if long_edge > DERIVED_MAX_LONG_EDGE:
                scale = DERIVED_MAX_LONG_EDGE / long_edge
                image = image.resize(
                    (round(width * scale), round(height * scale)),
                    Image.Resampling.LANCZOS,
                )
                width, height = image.size
            out = io.BytesIO()
            image.save(
                out,
                format="JPEG",
                quality=DERIVED_JPEG_QUALITY,
                optimize=False,
                progressive=False,
            )
            derived = out.getvalue()
    except PreprocessingError:
        raise
    except Exception:
        raise PreprocessingError(
            "Image preprocessing failed: unreadable image"
        )
    digest = hashlib.sha256(derived).hexdigest()
    return (
        derived,
        digest,
        width,
        height,
        DERIVED_MIME_TYPE,
        PROCESSOR_VERSION,
    )
