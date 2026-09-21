"""Phase 4 unit tests: detector protocol + output normalization.

Uses FakeDetector only (tests/fake_detector.py). No database, no S3,
no model weights: these tests pin the FaceDetector contract and the
service's normalize_faces behavior (clamping, invalid-box rejection,
deterministic ordering, landmarks passthrough).
"""

import pytest

from app.services import face_detection_service
from app.services.face_detector import DetectedFace, DetectorError
from tests.fake_detector import FakeDetector, box


def test_fake_detector_records_calls_and_returns_scripted():
    fake = FakeDetector(faces=[box(1, 2, 10, 20, 0.7)])
    out = fake.detect(b"bytes", 100, 100)
    assert len(out) == 1
    assert out[0].x_max == 10
    assert fake.calls == [(b"bytes", 100, 100)]


def test_fake_detector_zero_faces_is_empty_list():
    assert FakeDetector().detect(b"b", 50, 50) == []


def test_fake_detector_failure_raises_detector_error():
    fake = FakeDetector(fail_with="boom")
    with pytest.raises(DetectorError, match="boom"):
        fake.detect(b"b", 10, 10)


def test_normalize_clamps_to_frame():
    faces = face_detection_service.normalize_faces(
        [DetectedFace(-5, -5, 500, 500, 0.9, None)], 100, 80
    )
    assert len(faces) == 1
    assert (faces[0].x_min, faces[0].y_min) == (0, 0)
    assert (faces[0].x_max, faces[0].y_max) == (100, 80)


def test_normalize_drops_zero_area_and_inverted_boxes():
    raw = [
        DetectedFace(10, 10, 10, 20, 0.9, None),  # zero width
        DetectedFace(5, 5, 3, 9, 0.9, None),  # inverted
        DetectedFace(200, 200, 300, 300, 0.9, None),  # outside frame
        DetectedFace(1, 1, 5, 5, 0.9, None),  # valid
    ]
    faces = face_detection_service.normalize_faces(raw, 100, 100)
    assert len(faces) == 1
    assert (faces[0].x_min, faces[0].y_min) == (1, 1)


def test_normalize_orders_by_confidence_then_coordinates():
    raw = [
        box(50, 50, 90, 90, 0.5),
        box(10, 10, 30, 30, 0.9),
        box(5, 5, 20, 20, 0.9),
    ]
    faces = face_detection_service.normalize_faces(raw, 100, 100)
    assert [f.confidence for f in faces] == [0.9, 0.9, 0.5]
    assert (faces[0].x_min, faces[1].x_min) == (5, 10)


def test_normalize_preserves_confidence_and_landmarks():
    landmarks = {"nose_tip": [1.0, 2.0]}
    faces = face_detection_service.normalize_faces(
        [box(2, 2, 8, 8, 0.77, landmarks)], 100, 100
    )
    assert faces[0].confidence == 0.77
    assert faces[0].landmarks == landmarks


def test_active_detector_identity_matches_settings():
    from app.core.config import settings

    assert face_detection_service.active_detector_identity() == (
        settings.FACE_DETECTOR_NAME,
        settings.FACE_DETECTOR_VERSION,
        settings.FACE_DETECTION_THRESHOLD,
    )


def test_yunet_default_model_path_respects_env(monkeypatch):
    from app.services import yunet_detector

    monkeypatch.setenv("YUNET_MODEL_PATH", "/tmp/custom.onnx")
    assert yunet_detector.default_model_path() == "/tmp/custom.onnx"


def test_yunet_missing_model_file_raises_detector_error(tmp_path):
    from app.services.yunet_detector import YuNetDetector

    detector = YuNetDetector(
        model_path=str(tmp_path / "absent.onnx"),
        version="test",
        score_threshold=0.5,
    )
    with pytest.raises(DetectorError):
        detector.detect(b"bytes", 100, 100)


def test_yunet_empty_bytes_raise_detector_error(tmp_path):
    from app.services.yunet_detector import YuNetDetector

    detector = YuNetDetector(
        model_path=str(tmp_path / "absent.onnx"),
        version="test",
        score_threshold=0.5,
    )
    with pytest.raises(DetectorError):
        detector.detect(b"", 100, 100)
