"""Phase 5 real-model test: SFace adapter against real weights.

Isolated from the deterministic suite: skipped automatically when
the OpenCV runtime or the SFace weights are absent, and never
required for the normal test run (see
tests/fake_representation.py).

When enabled (weights at SFACE_MODEL_PATH or
backend/app/assets/face_recognition_sface_2021dec.onnx), it proves
the production path on a locally generated image: a synthetic
face geometry yields a 128-dim finite vector, model provenance is
captured from the exact file, and bad inputs raise
RepresentationError (never raw OpenCV exceptions).

Run from backend/:  python -m pytest tests/test_sface_real.py -v
"""

import hashlib
import io
import os

import pytest
from PIL import Image

from app.services.face_representation import (
    REPRESENTATION_DIMENSION,
    FaceGeometry,
    RepresentationError,
)
from app.services.sface_representation import (
    SFaceRepresentation,
    default_model_path,
    get_face_representation,
    hash_model_file,
)
from tests.fake_representation import yunet_landmarks

cv2 = pytest.importorskip("cv2")


def _model_available() -> bool:
    return os.path.exists(default_model_path())


def _image_bytes(size=(256, 256)) -> bytes:
    img = Image.new("RGB", size, "white")
    img.paste(Image.new("RGB", (96, 128), "gray"), (80, 64))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _geometry() -> FaceGeometry:
    return FaceGeometry(
        x_min=80,
        y_min=64,
        x_max=176,
        y_max=192,
        landmarks={
            "right_eye": [105.0, 105.0],
            "left_eye": [150.0, 105.0],
            "nose_tip": [128.0, 135.0],
            "mouth_right": [108.0, 160.0],
            "mouth_left": [148.0, 160.0],
        },
    )


def test_sface_model_path_points_at_bundled_weights():
    assert default_model_path().endswith(
        "face_recognition_sface_2021dec.onnx"
    )


def test_sface_missing_model_file_raises_domain_error(tmp_path):
    adapter = SFaceRepresentation(
        model_path=str(tmp_path / "absent.onnx")
    )
    with pytest.raises(RepresentationError):
        adapter.represent(_image_bytes(), _geometry())


def test_sface_empty_bytes_raise_domain_error(tmp_path):
    adapter = SFaceRepresentation(
        model_path=str(tmp_path / "absent.onnx")
    )
    with pytest.raises(RepresentationError):
        adapter.represent(b"", _geometry())


def test_sface_factory_returns_configured_adapter():
    adapter = get_face_representation()
    assert adapter.representation_name == "sface"
    assert adapter.model_name == "sface"
    assert adapter.dimension == REPRESENTATION_DIMENSION == 128


@pytest.mark.skipif(
    not _model_available(),
    reason="SFace weights absent (see backend/app/assets/README.md)",
)
def test_sface_model_sha_matches_file():
    with open(default_model_path(), "rb") as handle:
        expected = hashlib.sha256(handle.read()).hexdigest()
    assert hash_model_file(default_model_path()) == expected
    assert SFaceRepresentation().model_sha256 == expected
    assert len(expected) == 64


@pytest.mark.skipif(
    not _model_available(),
    reason="SFace weights absent (see backend/app/assets/README.md)",
)
def test_sface_synthetic_face_yields_128_finite_vector():
    import math

    vector = SFaceRepresentation().represent(
        _image_bytes(), _geometry()
    )
    assert len(vector) == 128
    assert all(
        isinstance(v, float) and math.isfinite(v) for v in vector
    )


@pytest.mark.skipif(
    not _model_available(),
    reason="SFace weights absent (see backend/app/assets/README.md)",
)
def test_sface_unreadable_bytes_raise_domain_error():
    adapter = SFaceRepresentation()
    with pytest.raises(RepresentationError):
        adapter.represent(b"definitely-not-an-image", _geometry())


@pytest.mark.skipif(
    not _model_available(),
    reason="SFace weights absent (see backend/app/assets/README.md)",
)
def test_sface_degenerate_geometry_raises_before_inference():
    adapter = SFaceRepresentation()
    bad = FaceGeometry(
        x_min=10,
        y_min=10,
        x_max=10,
        y_max=20,
        landmarks=yunet_landmarks(),
    )
    with pytest.raises(RepresentationError):
        adapter.represent(_image_bytes(), bad)
