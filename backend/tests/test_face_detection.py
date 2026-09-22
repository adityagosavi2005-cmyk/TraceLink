"""Phase 4 integration tests: face detection runs + faces.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

The YuNet production detector is replaced with FakeDetector
(tests/fake_detector.py) via the service factory, so every test is
deterministic. The isolated real-model path lives in
test_yunet_real.py (skipped without weights/runtime).

Covers both photo types symmetrically: READY gating, COMPLETE with
0..N faces, FAILED, current-result reuse, redetect history,
staleness, cascade cleanup, and authorization (incl. the narrow
REVIEWER detect permission with negative edit/delete checks).

Run from backend/:  python -m pytest tests/test_face_detection.py -v
"""

import io

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto, PhotoStatus  # noqa: F401
from app.models.enhancement import ImageSourceType  # noqa: F401
from app.models.face_detection import (  # noqa: F401
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.organization import OrgRole
from app.models.sighting import Sighting  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import face_detection_service, storage
from tests.conftest import (
    TestSession,
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from tests.fake_detector import FakeDetector, box

assert CasePhoto is not None and SightingPhoto is not None
assert FaceDetection is not None and FaceDetectionRun is not None


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
    """Route the production factory to a replaceable FakeDetector."""
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


def _make_case(client, headers, **extra):
    payload = {"title": "T", "description": "D"}
    payload.update(extra)
    resp = client.post("/cases", json=payload, headers=headers)
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
    """Authorized face-detection actor (ADMIN/REVIEWER only)."""
    return auth_headers(make_user(db_session, role=UserRole.REVIEWER))


def _run_count(kind, photo_id):
    db = TestSession()
    try:
        column = (
            FaceDetectionRun.case_photo_id
            if kind == "case"
            else FaceDetectionRun.sighting_photo_id
        )
        return (
            db.query(FaceDetectionRun)
            .filter(column == photo_id)
            .count()
        )
    finally:
        db.close()


# ---- CasePhoto: happy paths ----

def test_case_detect_zero_faces_is_complete(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETE"
    assert body["face_count"] == 0
    assert body["faces"] == []
    assert body["run"]["detector_name"] == settings.FACE_DETECTOR_NAME
    assert (
        body["run"]["source_derived_sha"] == photo["derived_sha256"]
    )


def test_case_detect_persists_faces_with_derived_coords(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(
        faces=[
            box(5, 5, 20, 20, 0.95),
            box(30, 10, 60, 50, 0.8),
        ]
    )

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["face_count"] == 2
    faces = sorted(body["faces"], key=lambda f: f["ordinal"])
    assert [f["confidence"] for f in faces] == [0.95, 0.8]
    assert faces[0]["frame_width"] == photo["derived_width"]
    assert faces[0]["case_photo_id"] == photo["id"]
    assert faces[0]["sighting_photo_id"] is None

    # Photo summary reflects the current run.
    got = client.get(
        "/cases/%d/photos/%d" % (case_id, photo["id"]), headers=headers
    )
    assert got.json()["face_detection_status"] == "COMPLETE"
    assert got.json()["face_count"] == 2


def test_case_detect_reuses_current_result(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(faces=[box(1, 1, 9, 9)])

    first = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    ).json()
    second = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    ).json()
    assert first["run"]["id"] == second["run"]["id"]
    assert _run_count("case", photo["id"]) == 1


def test_case_redetect_creates_new_run_and_keeps_history(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    first = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    ).json()
    second = client.post(
        "/cases/%d/photos/%d/faces/redetect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    ).json()
    assert first["run"]["id"] != second["run"]["id"]
    assert _run_count("case", photo["id"]) == 2


def test_case_get_faces_returns_current_result(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    missing = client.get(
        "/cases/%d/photos/%d/faces" % (case_id, photo["id"]),
        headers=headers,
    )
    assert missing.status_code == 404

    fake_factory["fake"] = FakeDetector(faces=[box(2, 2, 8, 8)])
    client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    got = client.get(
        "/cases/%d/photos/%d/faces" % (case_id, photo["id"]),
        headers=headers,
    )
    assert got.status_code == 200
    assert got.json()["face_count"] == 1


# ---- SightingPhoto: symmetric behavior ----

def test_sighting_detect_and_get(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    photo = _upload_sighting_photo(client, headers, case_id, sighting_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(faces=[box(3, 3, 12, 12, 0.88)])

    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["face_count"] == 1
    assert body["faces"][0]["sighting_photo_id"] == photo["id"]
    assert body["faces"][0]["case_photo_id"] is None
    assert body["run"]["sighting_id"] == sighting_id

    got = client.get(
        "/cases/%d/sightings/%d/photos/%d/faces"
        % (case_id, sighting_id, photo["id"]),
        headers=headers,
    )
    assert got.status_code == 200
    assert got.json()["face_count"] == 1


def test_sighting_redetect_reuses_and_appends(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    photo = _upload_sighting_photo(client, headers, case_id, sighting_id)
    reviewer_headers = _reviewer_headers(db)
    base = "/cases/%d/sightings/%d/photos/%d/faces" % (
        case_id, sighting_id, photo["id"]
    )

    first = client.post(base + "/detect", headers=reviewer_headers).json()
    again = client.post(base + "/detect", headers=reviewer_headers).json()
    assert first["run"]["id"] == again["run"]["id"]
    third = client.post(base + "/redetect", headers=reviewer_headers).json()
    assert third["run"]["id"] != first["run"]["id"]
    assert _run_count("sighting", photo["id"]) == 2


# ---- Failure and gating ----

def test_detect_rejects_non_ready_photo(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.processing_status = PhotoStatus.UPLOADED
        session.commit()
    finally:
        session.close()

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 409


def test_detector_failure_marks_run_failed(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(fail_with="detector exploded")

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "FAILED"
    assert body["face_count"] == 0
    assert "detector exploded" in (body["run"]["error"] or "")


def test_missing_derived_object_marks_run_failed(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    storage.delete_prefix("derived/%d/%d/" % (case_id, photo["id"]))

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "FAILED"


def test_derived_sha_mismatch_marks_run_failed(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    # Overwrite the stored derived object with different bytes.
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

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "FAILED"


def test_active_processing_run_conflicts(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        session.add(
            FaceDetectionRun(
                case_id=case_id,
                case_photo_id=photo["id"],
                sighting_photo_id=None,
                status=FaceDetectionStatus.PROCESSING,
                detector_name=settings.FACE_DETECTOR_NAME,
                detector_version=settings.FACE_DETECTOR_VERSION,
                threshold=settings.FACE_DETECTION_THRESHOLD,
                source_derived_sha=row.derived_sha256,
                source_derived_width=row.derived_width,
                source_derived_height=row.derived_height,
                source_type=ImageSourceType.DERIVED,
                source_sha256=row.derived_sha256,
                source_width=row.derived_width,
                source_height=row.derived_height,
            )
        )
        session.commit()
    finally:
        session.close()

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 409


def test_detector_version_change_stales_previous_run(
    client, db, s3mock, fake_factory, monkeypatch
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    url = "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"])

    first = client.post(url, headers=reviewer_headers).json()
    monkeypatch.setattr(
        settings, "FACE_DETECTOR_VERSION", "next-version"
    )
    second = client.post(url, headers=reviewer_headers).json()
    assert second["run"]["id"] != first["run"]["id"]
    assert second["run"]["detector_version"] == "next-version"
    assert _run_count("case", photo["id"]) == 2


# ---- Authorization ----

def test_outsider_blocked_from_detect_redetect_read(
    client, db, s3mock, fake_factory
):
    owner = make_user(db, role=UserRole.REPORTER)
    outsider = make_user(db, role=UserRole.REPORTER)
    owner_headers = auth_headers(owner)
    outsider_headers = auth_headers(outsider)
    case_id = _make_case(client, owner_headers)
    photo = _upload_case_photo(client, owner_headers, case_id)
    base = "/cases/%d/photos/%d/faces" % (case_id, photo["id"])

    assert (
        client.post(base + "/detect", headers=outsider_headers).status_code
        == 403
    )
    assert (
        client.post(base + "/redetect", headers=outsider_headers).status_code
        == 403
    )
    assert client.get(base, headers=outsider_headers).status_code == 403


def test_wrong_case_returns_404(client, db, s3mock, fake_factory):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    other_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (other_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 404


def test_reviewer_can_detect_but_gains_no_edit_rights(
    client, db, s3mock, fake_factory
):
    owner = make_user(db, role=UserRole.REPORTER)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    owner_headers = auth_headers(owner)
    reviewer_headers = auth_headers(reviewer)
    case_id = _make_case(client, owner_headers)
    photo = _upload_case_photo(client, owner_headers, case_id)

    resp = client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text

    redetect = client.post(
        "/cases/%d/photos/%d/faces/redetect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert redetect.status_code == 200, redetect.text
    assert redetect.json()["run"]["id"] != resp.json()["run"]["id"]

    # Narrow permission: reviewer still cannot edit the case,
    # delete the photo, or retry Phase 3 processing.
    assert (
        client.patch(
            "/cases/%d" % case_id,
            json={"title": "hijacked"},
            headers=reviewer_headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_id, photo["id"]),
            headers=reviewer_headers,
        ).status_code
        == 403
    )
    assert (
        client.post(
            "/cases/%d/photos/%d/retry" % (case_id, photo["id"]),
            headers=reviewer_headers,
        ).status_code
        == 403
    )


def test_sighting_reviewer_can_detect(client, db, s3mock, fake_factory):
    owner = make_user(db, role=UserRole.REPORTER)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    owner_headers = auth_headers(owner)
    reviewer_headers = auth_headers(reviewer)
    case_id = _make_case(client, owner_headers)
    sighting_id = _make_sighting(client, owner_headers, case_id)
    photo = _upload_sighting_photo(
        client, owner_headers, case_id, sighting_id
    )

    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert resp.status_code == 200, resp.text


def test_sighting_reporter_is_denied(client, db, s3mock, fake_factory):
    owner = make_user(db, role=UserRole.REPORTER)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    owner_headers = auth_headers(owner)
    reviewer_headers = auth_headers(reviewer)
    case_id = _make_case(client, owner_headers)
    sighting_id = _make_sighting(client, owner_headers, case_id)
    photo = _upload_sighting_photo(
        client, owner_headers, case_id, sighting_id
    )
    base = "/cases/%d/sightings/%d/photos/%d/faces" % (
        case_id, sighting_id, photo["id"]
    )

    # The sighting reporter gains no detection right from reporting.
    assert (
        client.post(base + "/detect", headers=owner_headers).status_code
        == 403
    )
    assert (
        client.post(base + "/redetect", headers=owner_headers).status_code
        == 403
    )

    # Reads still follow normal case-view authorization.
    assert client.get(base, headers=owner_headers).status_code == 404
    assert (
        client.post(base + "/detect", headers=reviewer_headers).status_code
        == 200
    )
    assert client.get(base, headers=owner_headers).status_code == 200


def test_sighting_outsider_blocked(client, db, s3mock, fake_factory):
    owner = make_user(db, role=UserRole.REPORTER)
    outsider = make_user(db, role=UserRole.REPORTER)
    owner_headers = auth_headers(owner)
    outsider_headers = auth_headers(outsider)
    case_id = _make_case(client, owner_headers)
    sighting_id = _make_sighting(client, owner_headers, case_id)
    photo = _upload_sighting_photo(
        client, owner_headers, case_id, sighting_id
    )
    base = "/cases/%d/sightings/%d/photos/%d/faces" % (
        case_id, sighting_id, photo["id"]
    )

    assert (
        client.post(base + "/detect", headers=outsider_headers).status_code
        == 403
    )
    assert client.get(base, headers=outsider_headers).status_code == 403


# ---- Cascade ----

def test_delete_case_photo_removes_runs_and_faces(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(faces=[box(1, 1, 9, 9)])
    client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    assert _run_count("case", photo["id"]) == 1

    session = TestSession()
    try:
        run_ids = [
            r.id
            for r in session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.case_photo_id == photo["id"])
            .all()
        ]
        assert run_ids
    finally:
        session.close()

    resp = client.delete(
        "/cases/%d/photos/%d" % (case_id, photo["id"]), headers=headers
    )
    assert resp.status_code == 200
    assert _run_count("case", photo["id"]) == 0
    session = TestSession()
    try:
        assert (
            session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.id.in_(run_ids))
            .count()
            == 0
        )
        assert (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id.in_(run_ids))
            .count()
            == 0
        )
    finally:
        session.close()


def test_delete_case_removes_all_face_data(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    sighting_id = _make_sighting(client, headers, case_id)
    sphoto = _upload_sighting_photo(
        client, headers, case_id, sighting_id
    )
    reviewer_headers = _reviewer_headers(db)
    fake_factory["fake"] = FakeDetector(faces=[box(1, 1, 9, 9)])
    client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=reviewer_headers,
    )
    client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, sphoto["id"]),
        headers=reviewer_headers,
    )

    session = TestSession()
    try:
        run_ids = [
            r.id
            for r in session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.case_id == case_id)
            .all()
        ]
        assert run_ids
    finally:
        session.close()

    resp = client.delete("/cases/%d" % case_id, headers=headers)
    assert resp.status_code == 200
    session = TestSession()
    try:
        assert (
            session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.case_id == case_id)
            .count()
            == 0
        )
        assert (
            session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.id.in_(run_ids))
            .count()
            == 0
        )
        assert (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id.in_(run_ids))
            .count()
            == 0
        )
    finally:
        session.close()


def test_delete_sighting_removes_sighting_face_data(
    client, db, s3mock, fake_factory
):
    user = make_user(db, role=UserRole.REPORTER)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    sphoto = _upload_sighting_photo(
        client, headers, case_id, sighting_id
    )
    fake_factory["fake"] = FakeDetector(faces=[box(1, 1, 9, 9)])
    reviewer_headers = _reviewer_headers(db)
    client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, sphoto["id"]),
        headers=reviewer_headers,
    )
    assert _run_count("sighting", sphoto["id"]) == 1

    session = TestSession()
    try:
        run_ids = [
            r.id
            for r in session.query(FaceDetectionRun)
            .filter(FaceDetectionRun.sighting_photo_id == sphoto["id"])
            .all()
        ]
        assert run_ids
    finally:
        session.close()

    resp = client.delete(
        "/cases/%d/sightings/%d" % (case_id, sighting_id), headers=headers
    )
    assert resp.status_code == 200
    assert _run_count("sighting", sphoto["id"]) == 0
    session = TestSession()
    try:
        assert (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id.in_(run_ids))
            .count()
            == 0
        )
    finally:
        session.close()


# ---- Narrow authorization: ADMIN/REVIEWER only ----

def test_admin_can_detect_and_redetect_case(
    client, db, s3mock, fake_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    admin_headers = auth_headers(admin)
    case_id = _make_case(client, admin_headers)
    photo = _upload_case_photo(client, admin_headers, case_id)
    base = "/cases/%d/photos/%d/faces" % (case_id, photo["id"])

    assert (
        client.post(base + "/detect", headers=admin_headers).status_code
        == 200
    )
    assert (
        client.post(base + "/redetect", headers=admin_headers).status_code
        == 200
    )
    assert _run_count("case", photo["id"]) == 2


def test_reporter_owner_denied_case_detect_redetect(
    client, db, s3mock, fake_factory
):
    owner = make_user(db, role=UserRole.REPORTER)
    owner_headers = auth_headers(owner)
    reviewer_headers = _reviewer_headers(db)
    case_id = _make_case(client, owner_headers)
    photo = _upload_case_photo(client, owner_headers, case_id)
    base = "/cases/%d/photos/%d/faces" % (case_id, photo["id"])

    # Case ownership grants upload/edit, but NOT face detection.
    assert (
        client.post(base + "/detect", headers=owner_headers).status_code
        == 403
    )
    assert (
        client.post(base + "/redetect", headers=owner_headers).status_code
        == 403
    )

    # Reads still follow normal case-view authorization.
    assert (
        client.post(base + "/detect", headers=reviewer_headers).status_code
        == 200
    )
    assert client.get(base, headers=owner_headers).status_code == 200


def test_org_investigator_denied_case_detect(
    client, db, s3mock, fake_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    admin_headers = auth_headers(admin)
    org = make_org(db, admin)
    investigator = make_user(db)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    investigator_headers = auth_headers(investigator)
    reviewer_headers = _reviewer_headers(db)

    case_id = _make_case(
        client, admin_headers, organization_id=org.id
    )
    photo = _upload_case_photo(client, admin_headers, case_id)
    base = "/cases/%d/photos/%d/faces" % (case_id, photo["id"])

    # Organization case-edit access does NOT imply detection access.
    assert (
        client.post(base + "/detect", headers=investigator_headers)
        .status_code
        == 403
    )
    assert (
        client.post(base + "/redetect", headers=investigator_headers)
        .status_code
        == 403
    )

    # ...while genuine case-edit rights are untouched.
    assert (
        client.patch(
            "/cases/%d" % case_id,
            json={"title": "investigator edit"},
            headers=investigator_headers,
        ).status_code
        == 200
    )

    # A reviewer can still detect on the same photo.
    assert (
        client.post(base + "/detect", headers=reviewer_headers).status_code
        == 200
    )


def test_sighting_investigator_denied(
    client, db, s3mock, fake_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    admin_headers = auth_headers(admin)
    org = make_org(db, admin)
    investigator = make_user(db)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    investigator_headers = auth_headers(investigator)
    reviewer_headers = _reviewer_headers(db)

    case_id = _make_case(
        client, admin_headers, organization_id=org.id
    )
    sighting_id = _make_sighting(client, admin_headers, case_id)
    photo = _upload_sighting_photo(
        client, admin_headers, case_id, sighting_id
    )
    base = "/cases/%d/sightings/%d/photos/%d/faces" % (
        case_id, sighting_id, photo["id"]
    )

    assert (
        client.post(base + "/detect", headers=investigator_headers)
        .status_code
        == 403
    )
    assert (
        client.post(base + "/redetect", headers=investigator_headers)
        .status_code
        == 403
    )

    # Reads still follow normal case-view authorization.
    assert (
        client.post(base + "/detect", headers=reviewer_headers).status_code
        == 200
    )
    assert (
        client.get(base, headers=investigator_headers).status_code == 200
    )
