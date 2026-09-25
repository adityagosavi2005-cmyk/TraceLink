"""Deterministic face input preparation (Phase 8).

Pure byte transformation shared by case and sighting restoration
flows. This module deliberately knows nothing about the database,
authorization, storage keys, users, cases, or sightings: it maps
Phase 3 derived image bytes plus one selected YuNet FaceDetection
geometry into the exact prepared face crop a FaceRestorer consumes.

Preparation contract (phase8-v1):
    - bbox clamped to the source frame, expanded to a centered
      square, shifted to fit, black-padded only when the square
      still overflows (edge faces)
    - square resized to GFPGAN_INPUT_SIZE x GFPGAN_INPUT_SIZE
      with Lanczos resampling (never upscaled beyond the guard;
      the service rejects oversized faces before calling here)
    - metadata/EXIF stripped from the output
    - encoded as JPEG quality 95 with pinned encoder flags

Determinism: same source bytes + same face geometry + same
prep version produce bit-identical prepared bytes and therefore
the same prepared-input SHA-256. No timestamps, random values,
or UUIDs are generated.

The prepared crop is EPHEMERAL: callers persist the SHA and the
transform record on the restoration run, never the bytes.
"""

import hashlib
import io
import math

from PIL import Image

from app.services.face_representation import (
    REPRESENTATION_LANDMARK_NAMES,
)

# Preparation pipeline version pinned into every restoration run.
# Bump this (e.g. phase8-v2) whenever the contract above changes
# so a new prepared SHA is produced instead of silently mixing
# renditions prepared under different rules.
FACE_PREP_VERSION = "phase8-v1"

# Edge length of the square prepared face crop. Matches the
# canonical GFPGAN aligned-face frame; the adapter asserts the
# relationship between prepared input and restored output.
GFPGAN_INPUT_SIZE = 512

# Output contract constants (single source of truth for Phase 8).
PREP_MIME_TYPE = "image/jpeg"
PREP_JPEG_QUALITY = 95

# Version of the canonical (synthetic) restored-frame landmark
# set, used only when mapped landmarks are invalid because the
# restorer realigned the face internally.
CANONICAL_GEOMETRY_VERSION = "phase8-canon-v1"

# Canonical landmark positions as fractions of the restored frame
# (width, height). Explicitly synthetic: they describe "a
# plausible frontal face filling the restored frame", not an
# observation. Order matches REPRESENTATION_LANDMARK_NAMES.
_CANONICAL_FRACTIONS = {
    "right_eye": (0.35, 0.38),
    "left_eye": (0.65, 0.38),
    "nose_tip": (0.50, 0.55),
    "mouth_right": (0.38, 0.72),
    "mouth_left": (0.62, 0.72),
}


class PreparationError(Exception):
    """Raised when a face cannot be prepared.

    Carries only a safe, user-presentable message: callers persist
    ``str(exc)`` as the run error, so internals (paths, SQL,
    tracebacks) must never be embedded here.
    """


def _validate_landmarks(landmarks) -> dict:
    """Return landmarks as {name: [x, y]} floats, or raise."""
    if not isinstance(landmarks, dict):
        raise PreparationError(
            "Face preparation failed: missing face landmarks"
        )
    cleaned: dict = {}
    for name in REPRESENTATION_LANDMARK_NAMES:
        point = landmarks.get(name)
        if (
            not isinstance(point, (list, tuple))
            or len(point) != 2
            or not all(
                isinstance(v, (int, float)) and math.isfinite(v)
                for v in point
            )
        ):
            raise PreparationError(
                "Face preparation failed: malformed face landmarks"
            )
        cleaned[name] = [float(point[0]), float(point[1])]
    return cleaned


def prepare_face(
    image_bytes: bytes,
    src_width: int,
    src_height: int,
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    landmarks,
) -> tuple:
    """Prepare one selected face for restoration.

    Returns (prepared_bytes, prepared_sha256, width, height,
    mime_type, transform). ``transform`` is a JSON-serializable
    dict recording the exact crop/pad/scale applied. Raises
    PreparationError on any failure.
    """
    if not image_bytes:
        raise PreparationError("Face preparation failed: empty image")
    if src_width <= 0 or src_height <= 0:
        raise PreparationError("Face preparation failed: invalid source")
    if x_max <= x_min or y_max <= y_min:
        raise PreparationError(
            "Face preparation failed: degenerate face box"
        )
    landmark_snapshot = _validate_landmarks(landmarks)

    # Clamp the bbox to the source frame first; the snapshot above
    # keeps the original detector values for provenance.
    cx0 = max(0, min(int(x_min), src_width))
    cy0 = max(0, min(int(y_min), src_height))
    cx1 = max(0, min(int(x_max), src_width))
    cy1 = max(0, min(int(y_max), src_height))
    if cx1 <= cx0 or cy1 <= cy0:
        raise PreparationError(
            "Face preparation failed: face is outside the image"
        )

    # Centered square around the clamped box.
    side = max(cx1 - cx0, cy1 - cy0)
    center_x = (cx0 + cx1) / 2.0
    center_y = (cy0 + cy1) / 2.0
    sq_x0 = int(round(center_x - side / 2.0))
    sq_y0 = int(round(center_y - side / 2.0))

    # Shift the square to fit inside the frame when possible.
    if side <= src_width:
        sq_x0 = max(0, min(sq_x0, src_width - side))
    if side <= src_height:
        sq_y0 = max(0, min(sq_y0, src_height - side))
    sq_x1 = sq_x0 + side
    sq_y1 = sq_y0 + side

    # Pad only the overflow (edge faces on small frames).
    pad_left = max(0, -sq_x0)
    pad_top = max(0, -sq_y0)
    pad_right = max(0, sq_x1 - src_width)
    pad_bottom = max(0, sq_y1 - src_height)

    try:
        with Image.open(io.BytesIO(image_bytes)) as src:
            image = src.convert("RGB")
            if image.size != (src_width, src_height):
                raise PreparationError(
                    "Face preparation failed: unreadable image"
                )
            crop = image.crop(
                (
                    max(0, sq_x0),
                    max(0, sq_y0),
                    min(src_width, sq_x1),
                    min(src_height, sq_y1),
                )
            )
            if pad_left or pad_top or pad_right or pad_bottom:
                padded = Image.new(
                    "RGB",
                    (crop.size[0] + pad_left + pad_right,
                     crop.size[1] + pad_top + pad_bottom),
                    (0, 0, 0),
                )
                padded.paste(crop, (pad_left, pad_top))
                crop = padded
            crop = crop.resize(
                (GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE),
                Image.Resampling.LANCZOS,
            )
            out = io.BytesIO()
            crop.save(
                out,
                format="JPEG",
                quality=PREP_JPEG_QUALITY,
                optimize=False,
                progressive=False,
            )
            prepared = out.getvalue()
    except PreparationError:
        raise
    except Exception:
        raise PreparationError(
            "Face preparation failed: unreadable image"
        )

    scale = GFPGAN_INPUT_SIZE / float(side)
    transform = {
        "prep_version": FACE_PREP_VERSION,
        "crop_x0": sq_x0,
        "crop_y0": sq_y0,
        "crop_side": side,
        "pad_left": pad_left,
        "pad_top": pad_top,
        "pad_right": pad_right,
        "pad_bottom": pad_bottom,
        "scale": scale,
        "alignment_method": "square-crop-pad-resize",
        "output_width": GFPGAN_INPUT_SIZE,
        "output_height": GFPGAN_INPUT_SIZE,
    }
    digest = hashlib.sha256(prepared).hexdigest()
    return (
        prepared,
        digest,
        GFPGAN_INPUT_SIZE,
        GFPGAN_INPUT_SIZE,
        PREP_MIME_TYPE,
        transform,
    )


def map_landmarks_to_prepared(landmarks: dict, transform: dict) -> dict:
    """Map source-frame landmarks into the prepared crop frame.

    Uses the ACTUAL recorded transform (crop origin, padding,
    scale). Raises PreparationError for malformed input. Only
    valid when preparation performed no geometry-changing warp
    (contract: square-crop-pad-resize).
    """
    cleaned = _validate_landmarks(landmarks)
    try:
        crop_x0 = float(transform["crop_x0"])
        crop_y0 = float(transform["crop_y0"])
        pad_left = float(transform.get("pad_left", 0))
        pad_top = float(transform.get("pad_top", 0))
        scale = float(transform["scale"])
    except (KeyError, TypeError, ValueError):
        raise PreparationError(
            "Face preparation failed: invalid transform record"
        )
    if not math.isfinite(scale) or scale <= 0:
        raise PreparationError(
            "Face preparation failed: invalid transform record"
        )
    return {
        name: [
            (pt[0] - crop_x0 + pad_left) * scale,
            (pt[1] - crop_y0 + pad_top) * scale,
        ]
        for name, pt in cleaned.items()
    }


def canonical_restored_landmarks(width: int, height: int) -> dict:
    """Versioned synthetic landmarks for a restored frame.

    Only used when the restorer realigned the face internally so
    mapped source landmarks are no longer valid. Explicitly
    marked synthetic via CANONICAL_GEOMETRY_VERSION at the call
    site; never silently substituted.
    """
    if width <= 0 or height <= 0:
        raise PreparationError(
            "Face preparation failed: invalid restored frame"
        )
    return {
        name: [fx * float(width), fy * float(height)]
        for name, (fx, fy) in _CANONICAL_FRACTIONS.items()
    }
