"""Face-representation abstraction (Phase 5).

The representation service depends on the FaceRepresentation
protocol, never on a concrete model implementation. Production
wires SFaceRepresentation (see sface_representation.py); automated
tests inject FakeRepresentation (tests/fake_representation.py,
test-only). Swapping the representation engine later means adding
one implementation, not redesigning services, persistence,
authorization, or the database.

This module is dependency-free on purpose (stdlib only): geometry
reconstruction and vector validation are pure functions so they
can be unit-tested without OpenCV, NumPy, or the database.

SFace input contract (OpenCV Zoo YuNet -> SFace workflow): the
face passed to ``FaceRecognizerSF.alignCrop`` is the YuNet-style
row WITHOUT the confidence score::

    [x, y, width, height,
     right_eye_x, right_eye_y,
     left_eye_x, left_eye_y,
     nose_tip_x, nose_tip_y,
     mouth_right_x, mouth_right_y,
     mouth_left_x, mouth_left_y]

The landmark order MUST match the order YuNet emits
(see yunet_detector._YUNET_LANDMARK_NAMES); do not reorder it.
"""

import math
from dataclasses import dataclass, field
from typing import Protocol

# Dimensionality of one SFace feature vector. Fixed by the model,
# NOT configurable: the face_embeddings schema stores VECTOR(128)
# and this constant is the single source of truth for validation.
REPRESENTATION_DIMENSION = 128

# Landmark names in the exact order YuNet emits them and SFace
# expects them. Shared here so the adapter and tests cannot drift
# from the Phase 4 representation.
REPRESENTATION_LANDMARK_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_right",
    "mouth_left",
)


class RepresentationError(Exception):
    """Raised when a representation cannot be produced.

    Carries only a safe, user-presentable message: callers surface
    ``str(exc)`` directly, so internals (paths, tracebacks, model
    dumps) must never be embedded here.
    """


@dataclass(frozen=True)
class FaceGeometry:
    """One Phase 4 face in source-image pixel coordinates.

    Mirrors the persisted FaceDetection bbox + landmarks columns so
    the adapter never touches ORM rows directly.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    # Named YuNet landmarks ({name: [x, y]}). Required: SFace
    # alignment consumes them, so None/malformed is an error here,
    # not a silent pass-through.
    landmarks: dict | None = field(default=None)


def build_sface_face_row(geometry: FaceGeometry) -> list[float]:
    """Reconstruct the 14-value SFace face row from Phase 4 data.

    Converts corner bbox (x_min, y_min, x_max, y_max) to
    (x, y, width, height) and appends the five landmarks in
    REPRESENTATION_LANDMARK_NAMES order. Raises
    RepresentationError for degenerate boxes or missing /
    malformed landmarks. Landmark coordinates are passed through
    unchanged (never clamped or fabricated): they are
    detector-native values in derived-image pixels, exactly as
    Phase 4 stores them.
    """
    width = geometry.x_max - geometry.x_min
    height = geometry.y_max - geometry.y_min
    if width <= 0 or height <= 0:
        raise RepresentationError(
            "Face representation failed: degenerate face box"
        )
    landmarks = geometry.landmarks
    if not isinstance(landmarks, dict):
        raise RepresentationError(
            "Face representation failed: missing face landmarks"
        )
    coords: list[float] = []
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
            raise RepresentationError(
                "Face representation failed: malformed face landmarks"
            )
        coords.extend([float(point[0]), float(point[1])])
    return [
        float(geometry.x_min),
        float(geometry.y_min),
        float(width),
        float(height),
        *coords,
    ]


def validate_vector(values) -> list[float]:
    """Validate a raw representation vector.

    Requires exactly REPRESENTATION_DIMENSION finite values (no
    NaN, no infinities) and returns them as plain floats. Raises
    RepresentationError otherwise. Enforced by both the SFace
    adapter and the persistence service so a faulty engine can
    never store an invalid row through either path.
    """
    try:
        items = [float(v) for v in values]
    except (TypeError, ValueError) as exc:
        raise RepresentationError(
            "Face representation failed: invalid feature vector"
        ) from exc
    if len(items) != REPRESENTATION_DIMENSION:
        raise RepresentationError(
            "Face representation failed: invalid feature vector"
        )
    if not all(math.isfinite(v) for v in items):
        raise RepresentationError(
            "Face representation failed: invalid feature vector"
        )
    return items


class FaceRepresentation(Protocol):
    """Produce a numerical representation for one detected face."""

    representation_name: str
    representation_version: str
    model_name: str
    model_version: str
    model_sha256: str
    dimension: int

    def represent(
        self, image_bytes: bytes, geometry: FaceGeometry
    ) -> list[float]:
        """Align and featurize one face from derived image bytes.

        Returns exactly ``dimension`` finite floats. Raises
        RepresentationError when representation cannot run. Never
        persists the aligned crop: alignment exists only in memory.
        """
        ...
