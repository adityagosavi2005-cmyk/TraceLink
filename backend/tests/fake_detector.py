"""Deterministic test-only face detector (Phase 4).

FakeDetector implements the FaceDetector protocol with scripted
results so unit/integration tests can exercise service logic,
persistence, state transitions, error handling, authorization, and
retries WITHOUT requiring YuNet weights or inference.

It must NEVER be used by the production application: the service
defaults to YuNetDetector via get_face_detector().
"""

from app.services.face_detector import DetectedFace, DetectorError


class FakeDetector:
    """Scripted FaceDetector for tests."""

    name = "fake"
    version = "test-v1"

    def __init__(
        self,
        faces: list[DetectedFace] | None = None,
        fail_with: str | None = None,
    ) -> None:
        self._faces = list(faces) if faces else []
        self.fail_with = fail_with
        self.calls: list[tuple[bytes, int, int]] = []

    def detect(
        self, image_bytes: bytes, width: int, height: int
    ) -> list[DetectedFace]:
        self.calls.append((image_bytes, width, height))
        if self.fail_with is not None:
            raise DetectorError(self.fail_with)
        return list(self._faces)


def box(
    x_min: int,
    y_min: int,
    x_max: int,
    y_max: int,
    confidence: float = 0.9,
    landmarks: dict | None = None,
) -> DetectedFace:
    """Convenience constructor for scripted faces."""
    return DetectedFace(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        confidence=confidence,
        landmarks=landmarks,
    )
