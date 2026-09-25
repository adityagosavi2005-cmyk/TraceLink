"""Deterministic test-only face restorer (Phase 8).

FakeRestorer implements the FaceRestorer protocol with scripted
results so unit/integration tests can exercise preparation,
service logic, persistence, state transitions, error handling,
authorization, retries, and multi-face flows WITHOUT requiring
the gfpgan package or GFPGAN weights.

It must NEVER be used by the production application: the service
defaults to GFPGANAdapter via get_face_restorer().
"""

import io

from PIL import Image

from app.services.face_preparation import GFPGAN_INPUT_SIZE
from app.services.face_restoration import (
    RestoredFace,
    RestorationError,
)


def default_output(size=None, color="green"):
    """Deterministic JPEG bytes used as the scripted output."""
    size = size or (GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE)
    img = Image.new("RGB", size, color)
    buf = io.BytesIO()
    img.save(
        buf,
        format="JPEG",
        quality=95,
        optimize=False,
        progressive=False,
    )
    return buf.getvalue()


class FakeRestorer:
    """Scripted FaceRestorer for tests."""

    name = "fake-restorer"
    version = "test-v1"
    model_name = "fake-restorer-model"
    model_version = "test-mv1"

    def __init__(
        self,
        output_bytes=None,
        width=GFPGAN_INPUT_SIZE,
        height=GFPGAN_INPUT_SIZE,
        mime_type="image/jpeg",
        realigned=False,
        fail_with=None,
        bad_model=False,
        model_sha256="2" * 64,
    ):
        self._output = (
            output_bytes
            if output_bytes is not None
            else default_output((width, height))
        )
        self._width = width
        self._height = height
        self._mime_type = mime_type
        self._realigned = realigned
        self.fail_with = fail_with
        self.bad_model = bad_model
        self._model_sha256 = model_sha256
        self.calls = []

    @property
    def model_sha256(self):
        if self.bad_model:
            raise RestorationError("Fake model is unavailable")
        return self._model_sha256

    def restore(self, prepared_bytes):
        self.calls.append(prepared_bytes)
        if self.fail_with is not None:
            raise RestorationError(self.fail_with)
        return RestoredFace(
            image_bytes=self._output,
            width=self._width,
            height=self._height,
            mime_type=self._mime_type,
            realigned=self._realigned,
        )
