"""Phase 2 tests: SightingPhoto upload/view/delete + cascade semantics.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

Covers the sighting-photo triple binding
(photo.case_id == sighting.case_id == URL case_id), the ordered
storage-first cascade on sighting and case deletion, and the
storage-failure rollback (HTTP 502, rows intact).

Run from backend/:  python -m pytest tests/test_sighting_photos.py -v
"""

import hashlib
import io
from datetime import datetime, timezone

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto  # noqa: F401
from app.models.organization import OrgRole
from app.models.sighting import Sighting  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import storage
from tests.conftest import (
    auth_headers,
    make_membership,
    make_org,
    make_user,
)

assert Sighting is not None and SightingPhoto is not None


@pytest.fixture()
def s3mock(monkeypatch):
    mock = mock_aws()
    mock.start()
    monkeypatch.setattr(settings, "S3_BUCKET", "test-bucket")
    monkeypatch.setattr(settings, "S3_ENDPOINT_URL", None)
    monkeypatch.setattr(settings, "max_photo_size_bytes", 1024 * 1024)
    storage.ensure_bucket()
    yield storage
    mock.stop()


def _image_bytes(fmt="PNG", size=(8, 6)):
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


def _upload(client, headers, case_id, sighting_id, data, filename="photo.png"):
    return client.post(
        "/cases/%d/sightings/%d/photos" % (case_id, sighting_id),
        files={"file": (filename, data, "image/png")},
        headers=headers,
    )


def _upload_case_photo(client, headers, case_id, data):
    return client.post(
        "/cases/%d/photos" % case_id,
        files={"file": ("photo.png", data, "image/png")},
        headers=headers,
    )


def test_sighting_photo_upload_ok(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")

    resp = _upload(client, headers, case_id, sighting_id, data)
    assert resp.status_code == 201
    body = resp.json()
    assert body["sighting_id"] == sighting_id
    assert body["case_id"] == case_id
    assert body["uploaded_by"] == owner.id
    assert body["mime_type"] == "image/png"
    assert body["byte_size"] == len(data)
    assert body["width"] == 8 and body["height"] == 6
    assert body["sha256"] == hashlib.sha256(data).hexdigest()
    # Phase 3: synchronous processing completes inside the upload.
    assert body["processing_status"] == "READY"
    assert body["view_url"].startswith("http")
    assert body["expires_in"] > 0

    # Stored under the sightings layout, readable back via the service.
    stored = s3mock._client().get_object(
        Bucket="test-bucket",
        Key="originals/sightings/%d/%d/%d/%s.png"
        % (case_id, sighting_id, body["id"], body["sha256"]),
    )["Body"].read()
    assert stored == data


def test_sighting_photo_unknown_sighting_404(s3mock, client, db):
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    resp = _upload(
        client, auth_headers(owner), case_id, 999999, _image_bytes("PNG")
    )
    assert resp.status_code == 404


def test_sighting_photo_rejects_non_image(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    resp = _upload(client, headers, case_id, sighting_id, b"hello text")
    assert resp.status_code == 415


def test_sighting_photo_rejects_oversize(s3mock, client, db, monkeypatch):
    monkeypatch.setattr(settings, "max_photo_size_bytes", 10)
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    resp = _upload(client, headers, case_id, sighting_id, _image_bytes("PNG"))
    assert resp.status_code == 413


def test_sighting_photo_authorization(s3mock, client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    owner = make_user(db)
    outsider = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")
    photo_id = _upload(client, headers, case_id, sighting_id, data).json()["id"]
    base = "/cases/%d/sightings/%d/photos" % (case_id, sighting_id)

    # Outsider: no upload, no list, no view, no delete.
    assert _upload(client, auth_headers(outsider), case_id, sighting_id, data).status_code == 403
    assert client.get(base, headers=auth_headers(outsider)).status_code == 403
    assert (
        client.get(
            "%s/%d" % (base, photo_id), headers=auth_headers(outsider)
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "%s/%d" % (base, photo_id), headers=auth_headers(outsider)
        ).status_code
        == 403
    )

    # Reviewer: read-only.
    assert client.get(base, headers=auth_headers(reviewer)).status_code == 200
    assert (
        client.get(
            "%s/%d" % (base, photo_id), headers=auth_headers(reviewer)
        ).status_code
        == 200
    )
    assert (
        _upload(client, auth_headers(reviewer), case_id, sighting_id, data).status_code
        == 403
    )

    # Admin: full access to another user's case.
    assert _upload(client, auth_headers(admin), case_id, sighting_id, data).status_code == 201


def test_sighting_reporter_can_photo_own_sighting(s3mock, client, db):
    """An org VIEWER (reporter-class role) may photograph their own
    sighting but not another member's."""
    admin = make_user(db, role=UserRole.ADMIN)
    investigator = make_user(db)
    viewer = make_user(db)
    org = make_org(db, admin)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    make_membership(db, viewer, org, OrgRole.VIEWER)

    case_id = _make_case(
        client, auth_headers(investigator), organization_id=org.id
    )
    own_sighting = client.post(
        "/cases/%d/sightings" % case_id,
        json={
            "sighting_at": datetime.now(timezone.utc).isoformat(),
            "location_text": "Pier",
            "description": "Seen at dawn.",
        },
        headers=auth_headers(viewer),
    ).json()["id"]
    others_sighting = _make_sighting(
        client, auth_headers(investigator), case_id
    )
    data = _image_bytes("PNG")

    assert (
        _upload(client, auth_headers(viewer), case_id, own_sighting, data).status_code
        == 201
    )
    assert (
        _upload(
            client, auth_headers(viewer), case_id, others_sighting, data
        ).status_code
        == 403
    )


def test_sighting_photo_triple_binding(s3mock, client, db):
    """A photo is reachable only through its own case AND sighting."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_a = _make_case(client, headers)
    case_b = _make_case(client, headers)
    sighting_a1 = _make_sighting(client, headers, case_a)
    sighting_a2 = _make_sighting(client, headers, case_a)
    sighting_b = _make_sighting(client, headers, case_b)
    photo_id = _upload(
        client, headers, case_a, sighting_a1, _image_bytes("PNG")
    ).json()["id"]

    # Wrong sighting, right case.
    assert (
        client.get(
            "/cases/%d/sightings/%d/photos/%d"
            % (case_a, sighting_a2, photo_id),
            headers=headers,
        ).status_code
        == 404
    )
    # Right sighting id, wrong case.
    assert (
        client.get(
            "/cases/%d/sightings/%d/photos/%d"
            % (case_b, sighting_a1, photo_id),
            headers=headers,
        ).status_code
        == 404
    )
    # Photo listed under its own sighting only.
    listed = client.get(
        "/cases/%d/sightings/%d/photos" % (case_a, sighting_a2),
        headers=headers,
    ).json()
    assert listed == []
    # Deletes through mismatched bindings are also 404.
    assert (
        client.delete(
            "/cases/%d/sightings/%d/photos/%d"
            % (case_a, sighting_a2, photo_id),
            headers=headers,
        ).status_code
        == 404
    )
    assert (
        client.delete(
            "/cases/%d/sightings/%d/photos/%d"
            % (case_b, sighting_b, photo_id),
            headers=headers,
        ).status_code
        == 404
    )


def test_delete_sighting_removes_photos_and_objects(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")
    photo_id = _upload(client, headers, case_id, sighting_id, data).json()["id"]

    assert (
        client.delete(
            "/cases/%d/sightings/%d" % (case_id, sighting_id),
            headers=headers,
        ).status_code
        == 200
    )
    # Photo row gone.
    assert (
        client.get(
            "/cases/%d/sightings/%d/photos/%d"
            % (case_id, sighting_id, photo_id),
            headers=headers,
        ).status_code
        == 404
    )
    # No objects remain under the sighting scope.
    remaining = s3mock._client().list_objects_v2(
        Bucket="test-bucket",
        Prefix="originals/sightings/%d/%d/" % (case_id, sighting_id),
    ).get("Contents", [])
    assert remaining == []


def test_delete_case_cleans_all_evidence(s3mock, client, db):
    """Case deletion removes CasePhoto + Sighting + SightingPhoto rows
    and every MinIO object: no orphaned rows, no orphaned objects."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")
    _upload_case_photo(client, headers, case_id, data)
    _upload(client, headers, case_id, sighting_id, data)

    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code
        == 200
    )
    assert (
        client.get("/cases/%d" % case_id, headers=headers).status_code
        == 404
    )
    assert (
        db.query(CasePhoto).filter(CasePhoto.case_id == case_id).count()
        == 0
    )
    assert (
        db.query(Sighting).filter(Sighting.case_id == case_id).count()
        == 0
    )
    assert (
        db.query(SightingPhoto)
        .filter(SightingPhoto.case_id == case_id)
        .count()
        == 0
    )
    remaining = s3mock._client().list_objects_v2(
        Bucket="test-bucket", Prefix="originals/"
    ).get("Contents", [])
    assert remaining == []


def test_delete_case_storage_failure_rolls_back(s3mock, client, db, monkeypatch):
    """When MinIO fails mid-cascade, nothing is deleted (HTTP 502)."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    data = _image_bytes("PNG")
    _upload_case_photo(client, headers, case_id, data)
    _upload(client, headers, case_id, sighting_id, data)

    def _boom(prefix):
        raise RuntimeError("storage down")

    monkeypatch.setattr(storage, "delete_prefix", _boom)
    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code
        == 502
    )
    # Every row survives the failed cascade.
    assert (
        client.get("/cases/%d" % case_id, headers=headers).status_code
        == 200
    )
    assert (
        db.query(CasePhoto).filter(CasePhoto.case_id == case_id).count()
        == 1
    )
    assert (
        db.query(Sighting).filter(Sighting.case_id == case_id).count()
        == 1
    )
    assert (
        db.query(SightingPhoto)
        .filter(SightingPhoto.case_id == case_id)
        .count()
        == 1
    )
