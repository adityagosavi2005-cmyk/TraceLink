"""Deterministic test-only image enhancer (Phase 7).

FakeEnhancer implements the ImageEnhancer protocol with scripted
results so unit/integration tests can exercise service logic,
persistence, state transitions, error handling, authorization, and
retries WITHOUT requiring torch or Real-ESRGAN weights.

It must NEVER be used by the production application: the service
defaults to RealESRGANAdapter via get_image_enhancer().
"""

import io

from PIL import Image

from app.services.enhancement import EnhancedImage, EnhancementError


def default_output(size=(32, 24), color="red"):
    """Deterministic JPEG bytes used as the scripted output."""
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=90,
        optimize=False,
        progressive=False,
    )
    return buf.getvalue()


class FakeEnhancer:
    """Scripted ImageEnhancer for tests."""

    name = "fake-enhancer"
    version = "test-v1"
    model_name = "fake-enhancer-model"
    model_version = "test-mv1"
    scale = 2

    def __init__(
        self,
        output_bytes=None,
        width=32,
        height=24,
        mime_type="image/jpeg",
        fail_with=None,
        bad_model=False,
        model_sha256="1" * 64,
    ):
        self._output = (
            output_bytes if output_bytes is not None else default_output()
        )
        self._width = width
        self._height = height
        self._mime_type = mime_type
        self.fail_with = fail_with
        self.bad_model = bad_model
        self._model_sha256 = model_sha256
        self.calls = []

    @property
    def model_sha256(self):
        if self.bad_model:
            raise EnhancementError("Fake model is unavailable")
        return self._model_sha256

    def enhance(self, image_bytes):
        self.calls.append(image_bytes)
        if self.fail_with is not None:
            raise EnhancementError(self.fail_with)
        return EnhancedImage(
            image_bytes=self._output,
            width=self._width,
            height=self._height,
            mime_type=self._mime_type,
        )
