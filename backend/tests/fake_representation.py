"""Deterministic test-only face representation (Phase 5).

FakeRepresentation implements the FaceRepresentation protocol
with scripted vectors so unit/service tests can exercise
idempotency, provenance, validation, and persistence WITHOUT
requiring SFace weights or inference.

It must NEVER be used by the production application: the service
defaults to SFaceRepresentation via get_face_representation().
"""

from app.services.face_representation import (
    REPRESENTATION_DIMENSION,
    RepresentationError,
    build_sface_face_row,
)


def default_vector() -> list[float]:
    """Deterministic valid 128-dim vector for scripted results."""
    return [float(i) / REPRESENTATION_DIMENSION for i in range(
        REPRESENTATION_DIMENSION
    )]


def yunet_landmarks(
    x: float = 10.0, y: float = 10.0
) -> dict:
    """YuNet-style landmark dict in SFace-expected key order."""
    return {
        "right_eye": [x, y],
        "left_eye": [x + 10.0, y],
        "nose_tip": [x + 5.0, y + 5.0],
        "mouth_right": [x, y + 10.0],
        "mouth_left": [x + 10.0, y + 10.0],
    }


class FakeRepresentation:
    """Scripted FaceRepresentation for tests."""

    def __init__(
        self,
        vector: list[float] | None = None,
        fail_with: str | None = None,
        representation_name: str = "fake-representation",
        representation_version: str = "test-v1",
        model_name: str = "fake-model",
        model_version: str = "test-mv1",
        model_sha256: str = "0" * 64,
    ) -> None:
        self._vector = (
            list(vector) if vector is not None else default_vector()
        )
        self.fail_with = fail_with
        self.representation_name = representation_name
        self.representation_version = representation_version
        self.model_name = model_name
        self.model_version = model_version
        self.model_sha256 = model_sha256
        self.dimension = REPRESENTATION_DIMENSION
        self.calls: list = []

    def represent(self, image_bytes: bytes, geometry):
        self.calls.append((image_bytes, geometry))
        if self.fail_with is not None:
            raise RepresentationError(self.fail_with)
        # Enforce the same geometry contract as the production
        # SFace adapter (which validates via build_sface_face_row
        # before inference): degenerate boxes and missing/malformed
        # landmarks must raise here too, not slip through on the
        # scripted path. The row itself is discarded; only the
        # scripted vector below is returned.
        build_sface_face_row(geometry)
        return list(self._vector)
