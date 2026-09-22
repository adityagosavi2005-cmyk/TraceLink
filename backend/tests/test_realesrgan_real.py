"""Phase 7 real-model test: Phase 3 derived image -> Real-ESRGAN.

Isolated from the deterministic suite: skipped automatically when
the torch runtime or the Real-ESRGAN weights are absent, and never
required for the normal test run (see tests/fake_enhancer.py).

When enabled (weights at ENHANCER_MODEL_PATH or
backend/app/assets/RealESRGAN_x4plus.pth), it proves the
production path end to end on a locally generated derived image:
output dimensions must equal input x scale, output must be
deterministic JPEG bytes, and unreadable bytes must raise
EnhancementError.

Run from backend/:  python -m pytest tests/test_realesrgan_real.py -v
"""

import hashlib
import io

import pytest
from PIL import Image

from app.services.enhancement import EnhancementError
from app.services.preprocessing import preprocess
from app.services.realesrgan_adapter import (
    RealESRGANAdapter,
    default_model_path,
)

torch = pytest.importorskip("torch")


def _model_available() -> bool:
    import os

    return os.path.exists(default_model_path())


@pytest.mark.skipif(
    not _model_available(),
    reason="Real-ESRGAN weights absent (see backend/app/assets/README.md)",
)
def test_realesrgan_upscales_derived_image():
    raw = io.BytesIO()
    Image.new("RGB", (32, 24), "white").save(raw, format="PNG")
    derived, _, width, height, _, _ = preprocess(raw.getvalue())

    adapter = RealESRGANAdapter()
    first = adapter.enhance(derived)
    assert first.width == width * adapter.scale
    assert first.height == height * adapter.scale
    assert first.mime_type == "image/jpeg"
    assert len(first.image_bytes) > 0
    # Deterministic: same input + weights give identical bytes.
    second = adapter.enhance(derived)
    assert hashlib.sha256(first.image_bytes).hexdigest() == (
        hashlib.sha256(second.image_bytes).hexdigest()
    )


@pytest.mark.skipif(
    not _model_available(),
    reason="Real-ESRGAN weights absent (see backend/app/assets/README.md)",
)
def test_realesrgan_unreadable_bytes_raise_enhancement_error():
    adapter = RealESRGANAdapter()
    with pytest.raises(EnhancementError):
        adapter.enhance(b"definitely-not-an-image")
