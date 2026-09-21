"""Face-detector abstraction (Phase 4).

The application service depends on the FaceDetector protocol, never
on a concrete model implementation. Production wires YuNetDetector
(see yunet_detector.py); automated tests inject FakeDetector
(tests/fake_detector.py, test-only). Swapping the detector later
means adding one implementation, not redesigning services, API,
authorization, or the database.
"""

from dataclasses import dataclass, field
from typing import Protocol


class DetectorError(Exception):
    """Raised when the detector cannot produce a result.

    Carries only a safe, user-presentable message: the service
    persists ``str(exc)`` as the run error, so internals (paths,
    tracebacks) must never be embedded here.
    """


@dataclass(frozen=True)
class DetectedFace:
    """One face hypothesis in derived-image pixel coordinates."""

    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    # Optional detector-native landmarks (e.g. YuNet eye/nose/mouth
    # points) kept opaquely for future alignment. None when the
    # detector provides none.
    landmarks: dict | None = field(default=None)


class FaceDetector(Protocol):
    """Detect faces in Phase 3 derived image bytes."""

    name: str
    version: str

    def detect(
        self, image_bytes: bytes, width: int, height: int
    ) -> list[DetectedFace]:
        """Return faces in derived-pixel coordinates.

        An empty list is a valid result (zero faces). Raises
        DetectorError when detection cannot run.
        """
        ...
