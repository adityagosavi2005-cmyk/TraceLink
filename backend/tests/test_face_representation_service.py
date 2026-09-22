"""Phase 5 service tests: representation persistence + idempotency.

SQLite + moto S3, mirroring tests/test_face_detection.py. Phase 4
runs are created deterministically through the real detection
service with FakeDetector (scripted YuNet-style faces WITH
landmarks); the representation engine is FakeRepresentation
(tests/fake_representation.py), so no SFace weights or inference
are needed. The PostgreSQL VECTOR(128) round-trip itself is
covered by tests/test_face_embeddings_postgres.py.

Run from backend/:  python -m pytest tests/test_face_representation_service.py -v
"""

import io

import pytest
from fastapi import HTTPException
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto  # noqa: F401
from app.models.face_detection import (  # noqa: F401
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.face_embedding import FaceEmbedding  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import face_representation_service, storage
from app.services.face_representation import RepresentationError
from tests.conftest import (
    TestSession,
    auth_headers,
    make_user,
)
from tests.fake_detector import FakeDetector, box
from tests.fake_representation import (
    FakeRepresentation,
    default_vector,
    yunet_landmarks,
)

assert CasePhoto is not None and SightingPhoto is not None
assert FaceDetection is not None and FaceDetectionRun is not None
assert FaceEmbedding is not None


@pytest.fixture()
def s3mock(monkeypatch):
    mock = mock_aws()
    mock.start()
    monkeypatch.setattr(settings, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(settings, "S3_ENDPOINT_URL", None)
    monkeypatch.setattr(settings, "PHOTO_MAX_BYTES", 1024 * 1024)
    storage.ensure_bucket()
    yield storage
    mock.stop()


@pytest.fixture()
def fake_factory(monkeypatch):
    """Route the production detector factory to FakeDetector."""
    holder = {"fake": FakeDetector()}
    from app.services import yunet_detector

    monkeypatch.setattr(
        yunet_detector, "get_face_detector", lambda: holder["fake"]
    )
    return holder


def _image_bytes(fmt="PNG", size=(64, 48)):
    img = Image.new("RGB", size, "blue")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_case(client, headers):
    resp = client.post(
        "/cases", json={"title": "T", "description": "D"},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _upload_case_photo(client, headers, case_id):
    resp = client.post(
        "/cases/%d/photos" % case_id,
        files={"file": ("photo.png", _image_bytes(), "image/png")},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["processing_status"] == "READY"
    return resp.json()


def _make_sighting(client, headers, case_id):
    resp = client.post(
        "/cases/%d/sightings" % case_id,
        json={
            "sighting_at": "2026-01-01T10:00:00Z",
            "location_text": "Park",
            "description": "Saw someone",
        },
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _upload_sighting_photo(client, headers, case_id, sighting_id):
    resp = client.post(
        "/cases/%d/sightings/%d/photos" % (case_id, sighting_id),
        files={"file": ("photo.png", _image_bytes(), "image/png")},
        headers=headers,
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["processing_status"] == "READY"
    return resp.json()


def _reviewer_headers(db_session):
    return auth_headers(make_user(db_session, role=UserRole.REVIEWER))


def _photo_row(photo_id):
    db = TestSession()
    try:
        row = db.get(CasePhoto, photo_id)
        db.expunge(row)
        return row
    finally:
        db.close()


def _detect(client, reviewer_headers, url, fake_factory, faces):
    """Run detection with scripted faces; assert COMPLETE."""
    fake_factory["fake"] = FakeDetector(faces=faces)
    resp = client.post(url, headers=reviewer_headers)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETE", body
    return body


def _embedding_count(photo_id, kind="case"):
    # Scoped to the test's own photo: the suite shares one
    # session-scoped database with no per-test truncation (see
    # tests/conftest.py), so a global count would accumulate rows
    # from other tests. This follows the existing convention of
    # asserting counts for the entity created by that test (cf.
    # _run_count(kind, photo_id) in test_face_detection.py).
    db = TestSession()
    try:
        column = (
            FaceDetection.case_photo_id
            if kind == "case"
            else FaceDetection.sighting_photo_id
        )
        return (
            db.query(FaceEmbedding)
            .join(
                FaceDetection,
                FaceEmbedding.face_detection_id == FaceDetection.id,
            )
            .filter(column == photo_id)
            .count()
        )
    finally:
        db.close()


def _embedding_version_count(photo_id, version, kind="case"):
    # Version-history count scoped to the test's own photo, for the
    # same shared-database reason as _embedding_count.
    db = TestSession()
    try:
        column = (
            FaceDetection.case_photo_id
            if kind == "case"
            else FaceDetection.sighting_photo_id
        )
        return (
            db.query(FaceEmbedding)
            .join(
                FaceDetection,
                FaceEmbedding.face_detection_id == FaceDetection.id,
            )
            .filter(
                column == photo_id,
                FaceEmbedding.representation_version == version,
            )
            .count()
        )
    finally:
        db.close()


def _face_count_for_photo(photo_id, kind="case"):
    # Face count scoped to the test's own photo.
    db = TestSession()
    try:
        column = (
            FaceDetection.case_photo_id
            if kind == "case"
            else FaceDetection.sighting_photo_id
        )
        return (
            db.query(FaceDetection).filter(column == photo_id).count()
        )
    finally:
        db.close()


def _embeddings_for_photo(photo_id):
    db = TestSession()
    try:
        rows = (
            db.query(FaceEmbedding)
            .join(
                FaceDetection,
                FaceEmbedding.face_detection_id == FaceDetection.id,
            )
            .filter(FaceDetection.case_photo_id == photo_id)
            .order_by(FaceEmbedding.id.asc())
            .all()
        )
        for row in rows:
            db.expunge(row)
        return rows
    finally:
        db.close()


def _two_faces():
    return [
        box(5, 5, 20, 20, 0.95, yunet_landmarks(6, 6)),
        box(30, 10, 55, 40, 0.8, yunet_landmarks(32, 12)),
    ]


def _setup_detected(client, db, s3mock, fake_factory, faces=None):
    """Upload a case photo and detect scripted faces on it."""
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    url = "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"])
    _detect(
        client, reviewer_headers, url, fake_factory,
        _two_faces() if faces is None else faces,
    )
    return case_id, photo, reviewer_headers


# ---- Happy paths ----

def test_represent_creates_embeddings_with_provenance(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    fake = FakeRepresentation()
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]), fake
    )
    assert len(rows) == 2
    for row in rows:
        assert row.id is not None
        assert row.source_derived_sha256 == photo["derived_sha256"]
        assert row.representation_name == "fake-representation"
        assert row.representation_version == "test-v1"
        assert row.model_name == "fake-model"
        assert row.model_version == "test-mv1"
        assert row.model_sha256 == "0" * 64
        assert row.dimension == 128
        assert list(row.embedding) == pytest.approx(default_vector())
    face_ids = sorted(r.face_detection_id for r in rows)
    db_faces = TestSession()
    try:
        expected = sorted(
            f.id for f in db_faces.query(FaceDetection).filter(
                FaceDetection.case_photo_id == photo["id"]
            ).all()
        )
    finally:
        db_faces.close()
    assert face_ids == expected
    # The adapter saw the real derived bytes for every face.
    assert len(fake.calls) == 2
    assert all(call[0] for call in fake.calls)


def test_represent_reuses_existing_without_recompute(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    fake = FakeRepresentation()
    # Sessions stay bound until the ids are compared: rows created
    # by a committed-then-collected session are detached, and
    # reading their expired attributes raises DetachedInstanceError
    # instead of testing reuse.
    first_session = TestSession()
    try:
        first = face_representation_service.represent_photo_faces(
            first_session, "case", _photo_row(photo["id"]), fake
        )
        calls_after_first = len(fake.calls)
        assert calls_after_first == 2
        second_session = TestSession()
        try:
            second = face_representation_service.represent_photo_faces(
                second_session, "case", _photo_row(photo["id"]), fake
            )
            assert [r.id for r in first] == [r.id for r in second]
            assert len(fake.calls) == calls_after_first
        finally:
            second_session.close()
    finally:
        first_session.close()
    assert _embedding_count(photo["id"]) == 2


def test_representation_version_change_creates_new_rows(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(representation_version="test-v2"),
    )
    assert len(rows) == 2
    assert _embedding_count(photo["id"]) == 4
    # History preserved: old-version rows remain untouched.
    # Scoped to this photo's faces (shared test database).
    old = _embedding_version_count(photo["id"], "test-v1")
    new = _embedding_version_count(photo["id"], "test-v2")
    assert old == 2
    assert new == 2


def test_model_version_change_creates_new_rows(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(model_version="test-mv2"),
    )
    assert _embedding_count(photo["id"]) == 4


# ---- Phase 4 interaction ----

def test_zero_faces_yields_zero_embeddings(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(
        client, db, s3mock, fake_factory, faces=[]
    )
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    assert rows == []
    assert _embedding_count(photo["id"]) == 0


def test_failed_run_yields_no_embeddings(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(
        fail_with="detector exploded"
    )
    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.json()["status"] == "FAILED"
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    assert rows == []
    assert _embedding_count(photo["id"]) == 0


def test_never_detected_photo_yields_empty(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    assert rows == []
    assert _embedding_count(photo["id"]) == 0


def test_superseded_run_faces_are_rejected(
    client, db, s3mock, fake_factory
):
    case_id, photo, reviewer_headers = _setup_detected(
        client, db, s3mock, fake_factory
    )
    url = "/cases/%d/photos/%d/faces" % (case_id, photo["id"])
    _detect(
        client, reviewer_headers, url + "/redetect", fake_factory,
        [box(1, 1, 9, 9, 0.9, yunet_landmarks(2, 2))],
    )
    db_session = TestSession()
    try:
        runs = (
            db_session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.case_photo_id == photo["id"])
            .order_by(FaceDetectionRun.id.asc())
            .all()
        )
        assert len(runs) == 2
        old_face = (
            db_session.query(FaceDetection)
            .filter(FaceDetection.run_id == runs[0].id)
            .first()
        )
        current_run = runs[1]
    finally:
        db_session.close()
    # Only the current run's faces are embedded by the photo entry.
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    assert len(rows) == 1
    assert rows[0].face_detection.run_id == current_run.id
    # A superseded face is rejected even when passed directly.
    with pytest.raises(RepresentationError):
        face_representation_service.represent_face(
            TestSession(), old_face, _photo_row(photo["id"]),
            FakeRepresentation(), current_run,
        )
    assert _embedding_count(photo["id"]) == 1


def test_sighting_photo_symmetry(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    photo = _upload_sighting_photo(
        client, headers, case_id, sighting_id
    )
    reviewer_headers = _reviewer_headers(db)
    base = "/cases/%d/sightings/%d/photos/%d/faces" % (
        case_id, sighting_id, photo["id"]
    )
    _detect(
        client, reviewer_headers, base + "/detect", fake_factory,
        [box(3, 3, 20, 20, 0.88, yunet_landmarks(4, 4))],
    )
    db_session = TestSession()
    try:
        sprow = db_session.get(SightingPhoto, photo["id"])
        db_session.expunge(sprow)
    finally:
        db_session.close()
    rows = face_representation_service.represent_photo_faces(
        TestSession(), "sighting", sprow, FakeRepresentation()
    )
    assert len(rows) == 1
    assert rows[0].source_derived_sha256 == photo["derived_sha256"]
    assert rows[0].face_detection.sighting_photo_id == photo["id"]


# ---- Failure handling: no row without a valid vector ----

def test_sha_mismatch_raises_and_stores_nothing(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        storage.put_derived(
            row.storage_key_derived, b"not the derived bytes",
            "image/jpeg",
        )
        session.commit()
    finally:
        session.close()
    with pytest.raises(RepresentationError):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(),
        )
    assert _embedding_count(photo["id"]) == 0


def test_missing_derived_object_raises(
    client, db, s3mock, fake_factory
):
    case_id, photo, _ = _setup_detected(
        client, db, s3mock, fake_factory
    )
    storage.delete_prefix("derived/%d/%d/" % (case_id, photo["id"]))
    with pytest.raises(RepresentationError):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(),
        )
    assert _embedding_count(photo["id"]) == 0


def test_missing_landmarks_raises_and_stores_nothing(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(
        client, db, s3mock, fake_factory,
        faces=[box(5, 5, 20, 20, 0.9, None)],
    )
    with pytest.raises(RepresentationError):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(),
        )
    assert _embedding_count(photo["id"]) == 0


def test_short_vector_rejected(client, db, s3mock, fake_factory):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    with pytest.raises(RepresentationError):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(vector=[0.0] * 64),
        )
    assert _embedding_count(photo["id"]) == 0


def test_nan_vector_rejected(client, db, s3mock, fake_factory):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    bad = default_vector()
    bad[3] = float("nan")
    with pytest.raises(RepresentationError):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(vector=bad),
        )
    assert _embedding_count(photo["id"]) == 0


def test_adapter_failure_propagates_without_row(
    client, db, s3mock, fake_factory
):
    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    with pytest.raises(RepresentationError, match="engine down"):
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(fail_with="engine down"),
        )
    assert _embedding_count(photo["id"]) == 0


def test_non_ready_photo_conflicts(client, db, s3mock, fake_factory):
    from app.models.case_photo import PhotoStatus

    _, photo, _ = _setup_detected(client, db, s3mock, fake_factory)
    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.processing_status = PhotoStatus.UPLOADED
        session.commit()
    finally:
        session.close()
    with pytest.raises(HTTPException) as exc_info:
        face_representation_service.represent_photo_faces(
            TestSession(), "case", _photo_row(photo["id"]),
            FakeRepresentation(),
        )
    assert exc_info.value.status_code == 409


# ---- Cascade ----

def test_delete_photo_removes_embeddings(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    url = "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"])
    _detect(
        client, reviewer_headers, url, fake_factory, _two_faces()
    )
    face_representation_service.represent_photo_faces(
        TestSession(), "case", _photo_row(photo["id"]),
        FakeRepresentation(),
    )
    assert _embedding_count(photo["id"]) == 2
    resp = client.delete(
        "/cases/%d/photos/%d" % (case_id, photo["id"]), headers=headers
    )
    assert resp.status_code == 200
    assert _embedding_count(photo["id"]) == 0
    # Scoped to the deleted photo: other tests' faces persist in
    # the shared test database.
    assert _face_count_for_photo(photo["id"]) == 0
