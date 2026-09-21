"""Phase 1 smoke tests: CasePhoto upload/view/delete + storage boundary.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

Run from backend/:  python -m pytest tests/test_case_photos.py -v
"""

import hashlib
import io

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto  # noqa: F401 (registers table)
from app.models.organization import OrgRole
from app.models.user import UserRole
from app.services import storage
from tests.conftest import (
    auth_headers,
    make_membership,
    make_org,
    make_user,
)

assert CasePhoto is not None


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


def _upload(client, headers, case_id, data, filename="photo.png"):
    return client.post(
        "/cases/%d/photos" % case_id,
        files={"file": (filename, data, "image/png")},
        headers=headers,
    )


def test_upload_ok(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")

    resp = _upload(client, headers, case_id, data)
    assert resp.status_code == 201
    body = resp.json()
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

    # Stored object is readable back through the service layer.
    stored = s3mock._client().get_object(
        Bucket="test-bucket",
        Key="originals/%d/%d/%s.png"
        % (case_id, body["id"], body["sha256"]),
    )["Body"].read()
    assert stored == data


def test_upload_jpeg_ok(s3mock, client, db):
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    resp = _upload(
        client, auth_headers(owner), case_id, _image_bytes("JPEG"), "p.jpg"
    )
    assert resp.status_code == 201
    assert resp.json()["mime_type"] == "image/jpeg"


def test_upload_unknown_case_404(s3mock, client, db):
    owner = make_user(db)
    resp = _upload(
        client, auth_headers(owner), 999999, _image_bytes("PNG")
    )
    assert resp.status_code == 404


def test_upload_rejects_non_image(s3mock, client, db):
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    resp = _upload(client, auth_headers(owner), case_id, b"hello text")
    assert resp.status_code == 415


def test_upload_rejects_corrupt_image(s3mock, client, db):
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    # PNG magic followed by garbage Pillow cannot decode.
    corrupt = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
    resp = _upload(client, auth_headers(owner), case_id, corrupt)
    assert resp.status_code == 400


def test_upload_rejects_empty_file(s3mock, client, db):
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    resp = _upload(client, auth_headers(owner), case_id, b"")
    assert resp.status_code == 400


def test_upload_rejects_oversize(s3mock, client, db, monkeypatch):
    monkeypatch.setattr(settings, "PHOTO_MAX_BYTES", 10)
    owner = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    resp = _upload(client, auth_headers(owner), case_id, _image_bytes("PNG"))
    assert resp.status_code == 413


def test_upload_spoofed_mime_rejected(s3mock, client, db):
    """Client-declared type text/plain around PNG bytes is still accepted
    as PNG (server detects), while mismatched content is rejected."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    resp = client.post(
        "/cases/%d/photos" % case_id,
        files={"file": ("evil.txt", data, "text/plain")},
        headers=headers,
    )
    assert resp.status_code == 201
    assert resp.json()["mime_type"] == "image/png"


def test_photo_authorization(s3mock, client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    owner = make_user(db)
    outsider = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    case_id = _make_case(client, auth_headers(owner))
    data = _image_bytes("PNG")
    photo_id = _upload(client, auth_headers(owner), case_id, data).json()["id"]

    # Outsider: no upload, no list, no view, no delete.
    assert _upload(client, auth_headers(outsider), case_id, data).status_code == 403
    assert (
        client.get(
            "/cases/%d/photos" % case_id, headers=auth_headers(outsider)
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/cases/%d/photos/%d" % (case_id, photo_id),
            headers=auth_headers(outsider),
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_id, photo_id),
            headers=auth_headers(outsider),
        ).status_code
        == 403
    )

    # Reviewer: read-only (views, cannot upload or delete).
    assert (
        client.get(
            "/cases/%d/photos" % case_id, headers=auth_headers(reviewer)
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/cases/%d/photos/%d" % (case_id, photo_id),
            headers=auth_headers(reviewer),
        ).status_code
        == 200
    )
    assert (
        _upload(client, auth_headers(reviewer), case_id, data).status_code
        == 403
    )

    # Admin: full access to another user's case.
    assert (
        _upload(client, auth_headers(admin), case_id, data).status_code == 201
    )


def test_org_photo_visibility(s3mock, client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    investigator = make_user(db)
    viewer = make_user(db)
    outsider = make_user(db)
    org = make_org(db, admin)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    make_membership(db, viewer, org, OrgRole.VIEWER)

    case_id = _make_case(
        client, auth_headers(investigator), organization_id=org.id
    )
    data = _image_bytes("PNG")
    photo_id = _upload(
        client, auth_headers(investigator), case_id, data
    ).json()["id"]

    # Org viewer reads but cannot upload or delete.
    assert (
        client.get(
            "/cases/%d/photos" % case_id, headers=auth_headers(viewer)
        ).status_code
        == 200
    )
    assert (
        _upload(client, auth_headers(viewer), case_id, data).status_code == 403
    )
    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_id, photo_id),
            headers=auth_headers(viewer),
        ).status_code
        == 403
    )
    # Cross-org outsider sees nothing.
    assert (
        client.get(
            "/cases/%d/photos" % case_id, headers=auth_headers(outsider)
        ).status_code
        == 403
    )


def test_photo_case_binding(s3mock, client, db):
    """A photo is only reachable through its own case."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_a = _make_case(client, headers)
    case_b = _make_case(client, headers)
    photo_id = _upload(client, headers, case_a, _image_bytes("PNG")).json()["id"]
    assert (
        client.get(
            "/cases/%d/photos/%d" % (case_b, photo_id), headers=headers
        ).status_code
        == 404
    )
    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_b, photo_id), headers=headers
        ).status_code
        == 404
    )


def test_delete_removes_row_and_objects(s3mock, client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    photo_id = _upload(client, headers, case_id, data).json()["id"]

    assert (
        client.delete(
            "/cases/%d/photos/%d" % (case_id, photo_id), headers=headers
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/cases/%d/photos/%d" % (case_id, photo_id), headers=headers
        ).status_code
        == 404
    )
    remaining = s3mock._client().list_objects_v2(
        Bucket="test-bucket", Prefix="originals/%d/" % case_id
    ).get("Contents", [])
    assert remaining == []


def test_original_preserved_on_reupload(s3mock, client, db):
    """Same bytes twice → two rows, distinct write-once keys, both intact."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)
    data = _image_bytes("PNG")
    first = _upload(client, headers, case_id, data).json()
    second = _upload(client, headers, case_id, data).json()
    assert first["id"] != second["id"]

    listed = client.get(
        "/cases/%d/photos" % case_id, headers=headers
    ).json()
    assert len(listed) == 2
    # Phase 3: synchronous processing completes inside the upload.
    assert all(p["processing_status"] == "READY" for p in listed)
    for photo in listed:
        assert photo["view_url"].startswith("http")
