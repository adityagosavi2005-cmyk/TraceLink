"""Phase 5 unit tests: geometry reconstruction + vector validation.

Pure-function tests only (app.services.face_representation). No
database, no S3, no model weights, no OpenCV: these tests pin the
SFace input contract (bbox conversion, landmark ordering) and the
validation rules (128 dims, finite values) that both the SFace
adapter and the persistence service enforce.
"""

import pytest

from app.services import face_representation as fr
from app.services.face_representation import (
    FaceGeometry,
    REPRESENTATION_DIMENSION,
    REPRESENTATION_LANDMARK_NAMES,
    RepresentationError,
    build_sface_face_row,
    validate_vector,
)
from app.services.sface_representation import (
    default_model_path,
    hash_model_file,
)
from tests.fake_representation import (
    FakeRepresentation,
    default_vector,
    yunet_landmarks,
)


def _geometry(**overrides) -> FaceGeometry:
    params = {
        "x_min": 10,
        "y_min": 20,
        "x_max": 50,
        "y_max": 80,
        "landmarks": yunet_landmarks(),
    }
    params.update(overrides)
    return FaceGeometry(**params)


def test_dimension_constant_is_128():
    assert REPRESENTATION_DIMENSION == 128


def test_landmark_order_matches_yunet_contract():
    assert REPRESENTATION_LANDMARK_NAMES == (
        "right_eye",
        "left_eye",
        "nose_tip",
        "mouth_right",
        "mouth_left",
    )


def test_build_row_converts_corners_and_orders_landmarks():
    row = build_sface_face_row(_geometry())
    assert row[:4] == [10.0, 20.0, 40.0, 60.0]
    assert row[4:] == [
        10.0, 10.0,  # right_eye
        20.0, 10.0,  # left_eye
        15.0, 15.0,  # nose_tip
        10.0, 20.0,  # mouth_right
        20.0, 20.0,  # mouth_left
    ]
    assert len(row) == 14


def test_build_row_rejects_degenerate_boxes():
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(x_max=10))  # zero width
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(y_max=20))  # zero height
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(x_min=60, x_max=50))  # inverted


def test_build_row_rejects_missing_landmarks():
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(landmarks=None))


def test_build_row_rejects_malformed_landmarks():
    bad = yunet_landmarks()
    del bad["nose_tip"]
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(landmarks=bad))
    with pytest.raises(RepresentationError):
        build_sface_face_row(
            _geometry(landmarks={"right_eye": [1.0]})
        )
    nan = yunet_landmarks()
    nan["left_eye"] = [float("nan"), 1.0]
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(landmarks=nan))
    with pytest.raises(RepresentationError):
        build_sface_face_row(_geometry(landmarks=["not", "a dict"]))


def test_build_row_passes_landmarks_through_unclamped():
    # Out-of-frame detector values are preserved, never fabricated:
    # Phase 4 stores landmarks opaquely and Phase 5 must not invent
    # coordinates.
    outside = {
        name: [-5.0, 5000.0] for name in REPRESENTATION_LANDMARK_NAMES
    }
    row = build_sface_face_row(_geometry(landmarks=outside))
    assert row[4:] == [-5.0, 5000.0] * 5


def test_validate_vector_accepts_128_finite():
    assert validate_vector(default_vector()) == default_vector()


def test_validate_vector_rejects_wrong_dimension():
    with pytest.raises(RepresentationError):
        validate_vector([0.0] * 127)
    with pytest.raises(RepresentationError):
        validate_vector([0.0] * 129)
    with pytest.raises(RepresentationError):
        validate_vector([])


def test_validate_vector_rejects_non_finite():
    with pytest.raises(RepresentationError):
        validate_vector([float("nan")] * 128)
    with pytest.raises(RepresentationError):
        validate_vector([float("inf")] * 128)
    with pytest.raises(RepresentationError):
        validate_vector([float("-inf")] * 128)
    mixed = default_vector()
    mixed[7] = float("nan")
    with pytest.raises(RepresentationError):
        validate_vector(mixed)


def test_validate_vector_rejects_non_numeric():
    with pytest.raises(RepresentationError):
        validate_vector(None)
    with pytest.raises(RepresentationError):
        validate_vector(["a"] * 128)


def test_fake_representation_identity_and_calls():
    fake = FakeRepresentation()
    assert (
        fake.representation_name,
        fake.representation_version,
        fake.model_name,
        fake.model_version,
    ) == (
        "fake-representation",
        "test-v1",
        "fake-model",
        "test-mv1",
    )
    assert fake.dimension == 128
    assert len(fake.model_sha256) == 64
    out = fake.represent(b"bytes", _geometry())
    assert out == default_vector()
    assert len(fake.calls) == 1


def test_fake_representation_failure_raises_domain_error():
    fake = FakeRepresentation(fail_with="boom")
    with pytest.raises(RepresentationError, match="boom"):
        fake.represent(b"b", _geometry())


def test_sface_default_model_path_respects_env(monkeypatch):
    monkeypatch.setenv("SFACE_MODEL_PATH", "/tmp/custom.onnx")
    assert default_model_path() == "/tmp/custom.onnx"


def test_sface_default_model_path_falls_back_to_assets(monkeypatch):
    monkeypatch.delenv("SFACE_MODEL_PATH", raising=False)
    path = default_model_path()
    assert path.endswith("face_recognition_sface_2021dec.onnx")
    assert "assets" in path


def test_hash_model_file_missing_raises_domain_error(tmp_path):
    with pytest.raises(RepresentationError):
        hash_model_file(str(tmp_path / "absent.onnx"))


def test_hash_model_file_matches_hashlib(tmp_path):
    import hashlib

    target = tmp_path / "model.onnx"
    target.write_bytes(b"fake-model-bytes")
    assert hash_model_file(str(target)) == hashlib.sha256(
        b"fake-model-bytes"
    ).hexdigest()


def test_module_has_no_heavy_imports():
    # face_representation must stay importable without OpenCV,
    # NumPy, SQLAlchemy, or pgvector (it is stdlib-only): assert
    # on the source so the check is independent of test ordering
    # and sys.modules state.
    import inspect

    source = inspect.getsource(fr)
    for heavy in ("cv2", "numpy", "sqlalchemy", "pgvector", "fastapi"):
        assert "import %s" % heavy not in source
