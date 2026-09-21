"""Phase 3 integration tests: sync processing, retry, deletion.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

Covers both photo types symmetrically: upload -> READY, failure ->
FAILED, retry, stale/fresh PROCESSING, authorization, derived URL
presence, original immutability, and original+derived deletion.

Run from backend/:  python -m pytest tests/test_photo_processing.py -v
"""

import hashlib
import io
from datetime import datetime, timedelta, timezone

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto, PhotoStatus  # noqa: F401 (registers table)
from app.models.sighting import Sighting  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import preprocessing, storage
from app.services.preprocessing import PROCESSOR_VERSION
from tests.conftest import (
    TestSession,
    auth_headers,
    make_user,
)

assert CasePhoto is not None and SightingPhoto is not None


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


def _image_bytes(fmt="PNG", size=(64, 48)):
    img = Image.new("RGB", size, "red")
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _make_case(client, headers, **extra):
    payload = {"title": "T", "description": "D"}
    payload.update(extra)
    resp = client.post("/cases", json=payload, headers=headers)
    assert resp.status_code == 201
    return resp.json()["id"]


def _make_sighting(client, headers, case_id):
    resp = client.post(
        "/cases/%d/sightings" % case_id,
        json={
            "sighting_at": datetime.now(timezone.utc).isoformat(),
            "location_text": "Market Square",
            "description": "Seen near the fountain.",
        },
        headers=headers,
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def _upload_case(client, headers, case_id, data):
    return client.post(
        "/cases/%d/photos" % case_id,
        files={"file": ("photo.png", data, "image/png")},
        headers=headers,
    )


def _upload_sighting(client, headers, case_id, sighting_id, data):
    return client.post(
        "/cases/%d/sightings/%d/photos" % (case_id, sighting_id),
        files={"file": ("photo.png", data, "image/png")},
        headers=headers,
    )


def _retry_case(client, headers, case_id, photo_id):
    return client.post(
        "/cases/%d/photos/%d/retry" % (case_id, photo_id),
        headers=headers,
    )


def _retry_sighting(client, headers, case_id, sighting_id, photo_id):
    return client.post(
        "/cases/%d/sightings/%d/photos/%d/retry"
        % (case_id, sighting_id, photo_id),
        headers=headers,
    )


def _expected_derived(data):
    return preprocessing.preprocess(data)


# ---------- CasePhoto ----------

def test_case_upload_ready_with_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")

    body = _upload_case(client, headers, case_id, data).json()
    assert body["processing_status"] == "READY"
    assert body["processing_error"] is None
    assert body["processing_version"] == PROCESSOR_VERSION
    assert body["derived_view_url"].startswith("http")
    assert body["derived_expires_in"] > 0
    assert body["view_url"].startswith("http")  # original still served

    derived, digest, width, height, mime, _ = _expected_derived(data)
    assert body["derived_sha256"] == digest
    assert body["derived_width"] == width
    assert body["derived_height"] == height
    assert body["derived_mime_type"] == mime == "image/jpeg"

    # Derived object exists at the deterministic key; original intact.
    stored_derived = s3mock._client().get_object(
        Bucket="test-bucket",
        Key="derived/%d/%d/%s.jpg" % (case_id, body["id"], digest),
    )["Body"].read()
    assert stored_derived == derived
    stored_original = s3mock._client().get_object(
        Bucket="test-bucket", Key="originals/%d/%d/%s.png"
        % (case_id, body["id"], body["sha256"]),
    )["Body"].read()
    assert stored_original == data


def test_case_processing_failure_keeps_original(s3mock, client, db, monkeypatch):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")

    def _boom(payload):
        raise preprocessing.PreprocessingError("Image preprocessing failed")

    monkeypatch.setattr(preprocessing, "preprocess", _boom)
    body = _upload_case(client, headers, case_id, data).json()
    assert body["processing_status"] == "FAILED"
    assert body["processing_error"]
    assert body["derived_view_url"] is None
    assert body["view_url"].startswith("http")

    # Original row + object survive the derived failure.
    photo = db.get(CasePhoto, body["id"])
    assert photo is not None
    stored = s3mock._client().get_object(
        Bucket="test-bucket", Key=photo.storage_key_original
    )["Body"].read()
    assert stored == data


def test_case_retry_failed_to_ready_is_idempotent(
    s3mock, client, db, monkeypatch
):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")

    def _boom(payload):
        raise preprocessing.PreprocessingError("Image preprocessing failed")

    with monkeypatch.context() as m:
        m.setattr(preprocessing, "preprocess", _boom)
        failed = _upload_case(client, headers, case_id, data).json()
        assert failed["processing_status"] == "FAILED"

    first = _retry_case(client, headers, case_id, failed["id"])
    assert first.status_code == 200
    assert first.json()["processing_status"] == "READY"
    assert first.json()["processing_error"] is None
    key_before = first.json()["derived_sha256"]

    second = _retry_case(client, headers, case_id, failed["id"])
    assert second.status_code == 200
    assert second.json()["processing_status"] == "READY"
    assert second.json()["derived_sha256"] == key_before


def test_case_historical_uploaded_retry(s3mock, client, db):
    """Pre-Phase-3 rows (UPLOADED, NULL derived) process via retry."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    body = _upload_case(client, headers, case_id, data).json()

    photo = db.get(CasePhoto, body["id"])
    photo.processing_status = PhotoStatus.UPLOADED
    photo.storage_key_derived = None
    photo.processing_version = None
    photo.derived_sha256 = None
    photo.derived_width = None
    photo.derived_height = None
    photo.derived_mime_type = None
    photo.processing_error = None
    photo.processing_started_at = None
    db.commit()
    s3mock._client().delete_object(
        Bucket="test-bucket",
        Key="derived/%d/%d/%s.jpg" % (case_id, body["id"], body["derived_sha256"]),
    )

    retried = _retry_case(client, headers, case_id, body["id"])
    assert retried.status_code == 200
    assert retried.json()["processing_status"] == "READY"
    assert retried.json()["derived_sha256"] == body["derived_sha256"]


def test_case_retry_marks_processing_and_blocks_concurrent(
    s3mock, client, db, monkeypatch
):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    body = _upload_case(client, headers, case_id, data).json()

    seen = {}

    real_put = storage.put_derived

    def _spy_put(key, payload, content_type):
        # The PROCESSING marker was committed before the pipeline ran:
        # a fresh session observes it mid-attempt.
        fresh = TestSession()
        try:
            row = fresh.get(CasePhoto, body["id"])
            seen["status"] = row.processing_status
            seen["started_at"] = row.processing_started_at
        finally:
            fresh.close()
        return real_put(key, payload, content_type)

    monkeypatch.setattr(storage, "put_derived", _spy_put)
    resp = _retry_case(client, headers, case_id, body["id"])
    assert resp.status_code == 200
    assert seen["status"] == PhotoStatus.PROCESSING
    assert seen["started_at"] is not None

    # Fresh PROCESSING cannot be retried again.
    photo = db.get(CasePhoto, body["id"])
    photo.processing_status = PhotoStatus.PROCESSING
    photo.processing_started_at = datetime.now(timezone.utc)
    db.commit()
    conflict = _retry_case(client, headers, case_id, body["id"])
    assert conflict.status_code == 409

    # Stale PROCESSING recovers through the same pipeline.
    photo.processing_started_at = datetime.now(timezone.utc) - timedelta(
        seconds=601
    )
    db.commit()
    # Targeted restoration: remove only the spy, keep the shared
    # s3mock fixture patches (a bare monkeypatch.undo() would revert
    # those too and redirect storage at the real configuration).
    monkeypatch.setattr(storage, "put_derived", real_put)
    recovered = _retry_case(client, headers, case_id, body["id"])
    assert recovered.status_code == 200
    assert recovered.json()["processing_status"] == "READY"


def test_case_retry_authorization(s3mock, client, db):
    owner = make_user(db)
    outsider = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    photo_id = _upload_case(
        client, headers, case_id, _image_bytes("PNG")
    ).json()["id"]

    assert _retry_case(
        client, auth_headers(outsider), case_id, photo_id
    ).status_code == 403
    assert _retry_case(
        client, auth_headers(reviewer), case_id, photo_id
    ).status_code == 403
    assert _retry_case(client, headers, case_id, photo_id).status_code == 200
    assert _retry_case(
        client, headers, 999999, photo_id
    ).status_code == 404


def test_case_delete_removes_original_and_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    photo_id = _upload_case(client, headers, case_id, data).json()["id"]

    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_id, photo_id), headers=headers
        ).status_code
        == 200
    )
    remaining_originals = s3mock._client().list_objects_v2(
        Bucket="test-bucket", Prefix="originals/%d/" % case_id
    ).get("Contents", [])
    remaining_derived = s3mock._client().list_objects_v2(
        Bucket="test-bucket", Prefix="derived/%d/" % case_id
    ).get("Contents", [])
    assert remaining_originals == []
    assert remaining_derived == []


# ---------- SightingPhoto (symmetry) ----------

def test_sighting_upload_ready_with_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")

    body = _upload_sighting(
        client, headers, case_id, sighting_id, data
    ).json()
    assert body["processing_status"] == "READY"
    assert body["processing_error"] is None
    assert body["processing_version"] == PROCESSOR_VERSION
    assert body["derived_view_url"].startswith("http")
    assert body["view_url"].startswith("http")

    derived, digest, width, height, mime, _ = _expected_derived(data)
    assert body["derived_sha256"] == digest
    assert (body["derived_width"], body["derived_height"]) == (width, height)
    assert body["derived_mime_type"] == mime

    stored_derived = s3mock._client().get_object(
        Bucket="test-bucket",
        Key="derived/sightings/%d/%d/%d/%s.jpg"
        % (case_id, sighting_id, body["id"], digest),
    )["Body"].read()
    assert stored_derived == derived
    stored_original = s3mock._client().get_object(
        Bucket="test-bucket", Key="originals/sightings/%d/%d/%d/%s.png"
        % (case_id, sighting_id, body["id"], body["sha256"]),
    )["Body"].read()
    assert stored_original == data
    assert hashlib.sha256(data).hexdigest() == body["sha256"]


def test_sighting_failure_retry_and_auth(s3mock, client, db, monkeypatch):
    owner = make_user(db)
    outsider = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")

    def _boom(payload):
        raise preprocessing.PreprocessingError("Image preprocessing failed")

    with monkeypatch.context() as m:
        m.setattr(preprocessing, "preprocess", _boom)
        failed = _upload_sighting(
            client, headers, case_id, sighting_id, data
        ).json()
        assert failed["processing_status"] == "FAILED"
        assert failed["processing_error"]
        assert failed["derived_view_url"] is None

    assert _retry_sighting(
        client, auth_headers(outsider), case_id, sighting_id, failed["id"]
    ).status_code == 403
    assert _retry_sighting(
        client, auth_headers(reviewer), case_id, sighting_id, failed["id"]
    ).status_code == 403

    retried = _retry_sighting(
        client, headers, case_id, sighting_id, failed["id"]
    )
    assert retried.status_code == 200
    assert retried.json()["processing_status"] == "READY"

    # Mismatched bindings stay 404 on retry.
    other_case = _make_case(client, headers)
    assert _retry_sighting(
        client, headers, other_case, sighting_id, failed["id"]
    ).status_code == 404


def test_sighting_delete_removes_original_and_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    photo_id = _upload_sighting(
        client, headers, case_id, sighting_id, _image_bytes("PNG")
    ).json()["id"]

    assert (
        client.delete(
            "/cases/%d/sightings/%d/photos/%d" % (case_id, sighting_id, photo_id),
            headers=headers,
        ).status_code
        == 200
    )
    for prefix in (
        "originals/sightings/%d/%d/" % (case_id, sighting_id),
        "derived/sightings/%d/%d/" % (case_id, sighting_id),
    ):
        remaining = s3mock._client().list_objects_v2(
            Bucket="test-bucket", Prefix=prefix
        ).get("Contents", [])
        assert remaining == []


def test_sighting_cascade_cleans_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    _upload_sighting(client, headers, case_id, sighting_id, _image_bytes("PNG"))

    assert (
        client.delete(
            "/cases/%d/sightings/%d" % (case_id, sighting_id), headers=headers
        ).status_code
        == 200
    )
    for prefix in (
        "originals/sightings/%d/%d/" % (case_id, sighting_id),
        "derived/sightings/%d/%d/" % (case_id, sighting_id),
    ):
        remaining = s3mock._client().list_objects_v2(
            Bucket="test-bucket", Prefix=prefix
        ).get("Contents", [])
        assert remaining == []


def test_case_cascade_cleans_all_derived(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")
    _upload_case(client, headers, case_id, data)
    _upload_sighting(client, headers, case_id, sighting_id, data)

    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code
        == 200
    )
    for prefix in ("originals/", "derived/"):
        remaining = s3mock._client().list_objects_v2(
            Bucket="test-bucket", Prefix=prefix
        ).get("Contents", [])
        assert remaining == []
    assert (
        db.query(CasePhoto).filter(CasePhoto.case_id == case_id).count() == 0
    )
    assert (
        db.query(SightingPhoto)
        .filter(SightingPhoto.case_id == case_id)
        .count()
        == 0
    )
