"""Image-enhancer abstraction (Phase 7).

The enhancement service depends on the ImageEnhancer protocol, never
on a concrete model implementation. Production wires
RealESRGANAdapter (see realesrgan_adapter.py); automated tests inject
FakeEnhancer (tests/fake_enhancer.py, test-only). Swapping the
enhancer later means adding one implementation, not redesigning
services, API, authorization, or the database.
"""

from dataclasses import dataclass
from typing import Protocol


class EnhancementError(Exception):
    """Raised when enhancement cannot produce a result.

    Carries only a safe, user-presentable message: the service
    persists ``str(exc)`` as the run error, so internals (paths,
    tracebacks, model dumps) must never be embedded here.
    """


@dataclass(frozen=True)
class EnhancedImage:
    """One enhanced rendition in output-image pixel coordinates."""

    image_bytes: bytes
    width: int
    height: int
    mime_type: str


class ImageEnhancer(Protocol):
    """Produce an enhanced rendition of Phase 3 derived bytes."""

    name: str
    version: str
    model_name: str
    model_version: str
    model_sha256: str
    scale: int

    def enhance(self, image_bytes: bytes) -> EnhancedImage:
        """Return the enhanced rendition of one derived image.

        Input is always Phase 3 derived bytes (never the original
        evidence). Raises EnhancementError when enhancement cannot
        run. Never persists anything: storage is the service's job.
        """
        ...
