"""Face-restorer abstraction (Phase 8).

The restoration service depends on the FaceRestorer protocol, never
on a concrete model implementation. Production wires
GFPGANAdapter (see gfpgan_adapter.py); automated tests inject
FakeRestorer (tests/fake_restorer.py, test-only). Swapping the
restorer later means adding one implementation, not redesigning
services, API, authorization, or the database.

This is intentionally separate from the Phase 7 ImageEnhancer
protocol: ImageEnhancer maps a whole derived image to a whole
enhanced image, while FaceRestorer maps one prepared face crop
to one restored face artifact.
"""

from dataclasses import dataclass
from typing import Protocol


class RestorationError(Exception):
    """Raised when restoration cannot produce a result.

    Carries only a safe, user-presentable message: the service
    persists ``str(exc)`` as the run error, so internals (paths,
    tracebacks, model dumps) must never be embedded here.
    """


@dataclass(frozen=True)
class RestoredFace:
    """One restored face artifact in output-face pixel coordinates."""

    image_bytes: bytes
    width: int
    height: int
    mime_type: str
    # True when the model performed its own internal detection /
    # alignment, so source-frame landmarks mapped through the
    # preparation transform are no longer valid and the service
    # must use versioned canonical restored-frame geometry.
    realigned: bool


class FaceRestorer(Protocol):
    """Produce a restored face from prepared face bytes."""

    name: str
    version: str
    model_name: str
    model_version: str
    model_sha256: str

    def restore(self, prepared_bytes: bytes) -> RestoredFace:
        """Return the restored face for one prepared face crop.

        Input is always the prepared face bytes produced by
        face_preparation (never a whole photo, never the original
        evidence). Raises RestorationError when restoration cannot
        run. Never persists anything: storage is the service's job.
        """
        ...
