"""Phase 4 real-model test: Phase 3 derived image -> YuNet.

Isolated from the deterministic suite: skipped automatically when
the OpenCV runtime or the YuNet weights are absent, and never
required for the normal test run (see tests/fake_detector.py).

When enabled (weights at YUNET_MODEL_PATH or
backend/app/assets/face_detection_yunet_2023mar.onnx), it proves the
production path end to end on a locally generated derived image:
blank input must complete with zero faces, and unreadable bytes
must raise DetectorError.

Run from backend/:  python -m pytest tests/test_yunet_real.py -v
"""

import io

import pytest
from PIL import Image

from app.services.face_detector import DetectorError
from app.services.preprocessing import preprocess
from app.services.yunet_detector import YuNetDetector, default_model_path

cv2 = pytest.importorskip("cv2")


def _model_available() -> bool:
    import os

    return os.path.exists(default_model_path())


@pytest.mark.skipif(
    not _model_available(),
    reason="YuNet weights absent (see backend/app/assets/README.md)",
)
def test_yunet_blank_derived_image_yields_zero_faces():
    raw = io.BytesIO()
    Image.new("RGB", (320, 320), "white").save(raw, format="PNG")
    derived, _, width, height, _, _ = preprocess(raw.getvalue())

    detector = YuNetDetector()
    faces = detector.detect(derived, width, height)
    assert isinstance(faces, list)
    assert faces == []


@pytest.mark.skipif(
    not _model_available(),
    reason="YuNet weights absent (see backend/app/assets/README.md)",
)
def test_yunet_unreadable_bytes_raise_detector_error():
    detector = YuNetDetector()
    with pytest.raises(DetectorError):
        detector.detect(b"definitely-not-an-image", 64, 64)
