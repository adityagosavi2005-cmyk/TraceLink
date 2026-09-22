"""Phase 7 integration tests: AI image enhancement/restoration.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

The Real-ESRGAN production adapter is replaced with FakeEnhancer
(tests/fake_enhancer.py) via the service factory, so every test is
deterministic (torch/weights never required). The isolated
real-model path lives in test_realesrgan_real.py (skipped without
weights/runtime).

Covers both photo types symmetrically: authorization (ADMIN and
REVIEWER only, ownership grants nothing), READY gating, derived
input, original/derived immutability, output SHA/dimensions/MIME,
provenance, COMPLETE/FAILED, safe errors, retry, stale
PROCESSING, multiple runs, storage layout/guards/cleanup, source
resolver validity (all five conditions), enhanced detection,
embeddings coexistence, source-aware similarity, and deletion.

Run from backend/:  python -m pytest tests/test_enhancement.py -v
"""

import hashlib
import io
from datetime import datetime, timedelta, timezone

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto, PhotoStatus  # noqa: F401
from app.models.enhancement import (  # noqa: F401
    EnhancementRun,
    EnhancementStatus,
    ImageSourceType,
)
from app.models.face_detection import (  # noqa: F401
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.face_embedding import FaceEmbedding  # noqa: F401
from app.models.organization import OrgRole
from app.models.sighting import Sighting  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import enhancement_service, storage
from tests.conftest import (
    TestSession,
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from tests.fake_detector import FakeDetector, box
from tests.fake_enhancer import FakeEnhancer, default_output
from tests.fake_representation import FakeRepresentation, yunet_landmarks

assert CasePhoto is not None and SightingPhoto is not None
assert FaceDetection is not None and FaceDetectionRun is not None
assert FaceEmbedding is not None and EnhancementRun is not None


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
    """Route the production detector factory to a FakeDetector."""
    holder = {"fake": FakeDetector()}
    from app.services import yunet_detector

    monkeypatch.setattr(
        yunet_detector, "get_face_detector", lambda: holder["fake"]
    )
    return holder


@pytest.fixture()
def enhancer_factory(monkeypatch):
    """Route the production enhancer factory to a FakeEnhancer."""
    holder = {"fake": FakeEnhancer()}
    from app.services import realesrgan_adapter

    monkeypatch.setattr(
        realesrgan_adapter,
        "get_image_enhancer",
        lambda: holder["fake"],
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


def _enhance_case(client, headers, case_id, photo_id):
    return client.post(
        "/cases/%d/photos/%d/enhancements" % (case_id, photo_id),
        headers=headers,
    )


def _enhance_sighting(client, headers, case_id, sighting_id, photo_id):
    return client.post(
        "/cases/%d/sightings/%d/photos/%d/enhancements"
        % (case_id, sighting_id, photo_id),
        headers=headers,
    )


def _photo_row(model, photo_id):
    db = TestSession()
    try:
        row = db.get(model, photo_id)
        db.expunge(row)
        return row
    finally:
        db.close()


def _run_count(kind, photo_id):
    db = TestSession()
    try:
        column = (
            EnhancementRun.case_photo_id
            if kind == "case"
            else EnhancementRun.sighting_photo_id
        )
        return (
            db.query(EnhancementRun).filter(column == photo_id).count()
        )
    finally:
        db.close()


def _detect(client, headers, url, fake_factory, faces, params=""):
    fake_factory["fake"] = FakeDetector(faces=faces)
    resp = client.post(url + params, headers=headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "COMPLETE", resp.text
    return resp.json()


def _one_face():
    return [box(5, 5, 20, 20, 0.95, yunet_landmarks(6, 6))]


def _one_enhanced_face():
    # Fits the 32x24 FakeEnhancer output frame.
    return [box(2, 2, 16, 16, 0.95, yunet_landmarks(3, 3))]


def _sface_embed(db, face, photo, vector):
    from app.services import face_representation_service

    face_db = db.get(FaceDetection, face.id)
    return face_representation_service.represent_face(
        db,
        face_db,
        photo,
        FakeRepresentation(
            vector=vector,
            representation_name="sface",
            representation_version="2021dec",
            model_name="sface",
            model_version="2021dec",
        ),
    )


def _vec(*vals):
    out = [0.0] * 128
    for i, v in enumerate(vals):
        out[i] = v
    return out


E1 = _vec(1.0)


# ---- Authorization ----

def _setup_trigger_world(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    reporter = make_user(db, role=UserRole.REPORTER)
    org_owner = make_user(db, role=UserRole.ADMIN)
    org = make_org(db, org_owner, name="EnhAuth")
    investigator = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    org_admin = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, org_admin, org, OrgRole.ORG_ADMIN)
    viewer = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, viewer, org, OrgRole.VIEWER)
    owner = make_user(db, role=UserRole.REPORTER)
    # The trigger case lives in the org so membership-based view
    # checks are meaningful (trigger permission itself is
    # role-only and ignores the case entirely).
    case_id = _make_case(
        client, auth_headers(admin), organization_id=org.id
    )
    photo = _upload_case_photo(client, auth_headers(admin), case_id)
    owner_case = _make_case(client, auth_headers(owner))
    owner_photo = _upload_case_photo(
        client, auth_headers(owner), owner_case
    )
    return {
        "admin_h": auth_headers(admin),
        "reviewer_h": auth_headers(reviewer),
        "reporter_h": auth_headers(reporter),
        "investigator_h": auth_headers(investigator),
        "org_admin_h": auth_headers(org_admin),
        "viewer_h": auth_headers(viewer),
        "owner_h": auth_headers(owner),
        "case_id": case_id,
        "photo_id": photo["id"],
        "owner_case": owner_case,
        "owner_photo": owner_photo["id"],
    }


def test_admin_and_reviewer_can_trigger(
    client, db, s3mock, enhancer_factory
):
    world = _setup_trigger_world(client, db)
    for headers in (world["admin_h"], world["reviewer_h"]):
        resp = _enhance_case(
            client, headers, world["case_id"], world["photo_id"]
        )
        assert resp.status_code == 201, resp.text
        assert resp.json()["status"] == "COMPLETE"
        # Raw image bytes are never returned.
        assert "image_bytes" not in resp.json()


def test_denied_roles_cannot_trigger(
    client, db, s3mock, enhancer_factory
):
    world = _setup_trigger_world(client, db)
    for headers in (
        world["reporter_h"],
        world["investigator_h"],
        world["org_admin_h"],
        world["viewer_h"],
    ):
        resp = _enhance_case(
            client, headers, world["case_id"], world["photo_id"]
        )
        assert resp.status_code == 403, resp.text
    assert _run_count("case", world["photo_id"]) == 0


def test_case_owner_without_role_cannot_trigger(
    client, db, s3mock, enhancer_factory
):
    world = _setup_trigger_world(client, db)
    # The reporter owns this case and uploaded this photo, yet
    # ownership grants no enhancement permission in v1.
    resp = _enhance_case(
        client, world["owner_h"], world["owner_case"],
        world["owner_photo"],
    )
    assert resp.status_code == 403, resp.text


def test_retry_requires_trigger_role(
    client, db, s3mock, enhancer_factory
):
    world = _setup_trigger_world(client, db)
    enhancer_factory["fake"] = FakeEnhancer(fail_with="nope")
    failed = _enhance_case(
        client, world["admin_h"], world["case_id"], world["photo_id"]
    ).json()
    assert failed["status"] == "FAILED"
    url = "/cases/%d/photos/%d/enhancements/%d/retry" % (
        world["case_id"], world["photo_id"], failed["id"],
    )
    resp = client.post(url, headers=world["reporter_h"])
    assert resp.status_code == 403, resp.text
    resp = client.post(url, headers=world["investigator_h"])
    assert resp.status_code == 403, resp.text


def test_view_roles_can_list_and_detail(
    client, db, s3mock, enhancer_factory
):
    world = _setup_trigger_world(client, db)
    run = _enhance_case(
        client, world["admin_h"], world["case_id"], world["photo_id"]
    ).json()
    # Case visibility governs viewing: an unrelated reporter sees
    # nothing, while org viewers and the owner can read runs.
    resp = client.get(
        "/cases/%d/photos/%d/enhancements"
        % (world["case_id"], world["photo_id"]),
        headers=world["reporter_h"],
    )
    assert resp.status_code == 403, resp.text
    resp = client.get(
        "/cases/%d/photos/%d/enhancements/%d"
        % (world["case_id"], world["photo_id"], run["id"]),
        headers=world["viewer_h"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["id"] == run["id"]
    # COMPLETE detail carries a presigned view URL, never bytes.
    assert resp.json()["view_url"] is not None
    assert "image_bytes" not in resp.json()


# ---- Enhancement execution ----

def _setup_photo(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    return headers, case_id, photo


def test_ready_gating(client, db, s3mock, enhancer_factory):
    headers, case_id, photo = _setup_photo(client, db)
    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.processing_status = PhotoStatus.FAILED
        session.commit()
    finally:
        session.close()
    resp = _enhance_case(client, headers, case_id, photo["id"])
    assert resp.status_code == 409, resp.text
    assert enhancer_factory["fake"].calls == []
    assert _run_count("case", photo["id"]) == 0


def test_derived_input_and_immutability(
    client, db, s3mock, enhancer_factory
):
    headers, case_id, photo = _setup_photo(client, db)
    before = _photo_row(CasePhoto, photo["id"])
    original = storage.get_original_bytes(before.storage_key_original)
    derived = storage.get_derived_bytes(before.storage_key_derived)
    resp = _enhance_case(client, headers, case_id, photo["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    fake = enhancer_factory["fake"]
    assert len(fake.calls) == 1
    # Enhancement input comes from the derived rendition, never the
    # original.
    assert fake.calls[0] == derived
    assert fake.calls[0] != original
    after = _photo_row(CasePhoto, photo["id"])
    # Original evidence and the Phase 3 derived artifact are
    # untouched: same keys, same bytes, same status.
    assert after.storage_key_original == before.storage_key_original
    assert after.storage_key_derived == before.storage_key_derived
    assert after.derived_sha256 == before.derived_sha256
    assert after.processing_status == PhotoStatus.READY
    assert storage.get_original_bytes(
        after.storage_key_original) == original
    assert storage.get_derived_bytes(
        after.storage_key_derived) == derived
    # Run carries complete provenance.
    assert body["status"] == "COMPLETE"
    assert body["source_derived_sha256"] == before.derived_sha256
    assert body["enhancer_name"] == "fake-enhancer"
    assert body["model_name"] == "fake-enhancer-model"
    assert body["model_sha256"] == "1" * 64
    expected_sha = hashlib.sha256(default_output()).hexdigest()
    assert body["output_sha256"] == expected_sha
    assert body["width"] == 32
    assert body["height"] == 24
    assert body["mime_type"] == "image/jpeg"
    assert body["byte_size"] == len(default_output())
    assert body["error_message"] is None
    assert body["case_photo_id"] == photo["id"]
    assert body["sighting_photo_id"] is None
    # Stored artifact matches the recorded SHA under the
    # documented enhanced/ key layout (storage keys themselves are
    # never exposed through the API).
    assert "storage_key" not in body
    key = storage.build_enhanced_key(case_id, photo["id"], expected_sha)
    assert key == "enhanced/%d/%d/%s.jpg" % (
        case_id, photo["id"], expected_sha)
    assert storage.get_enhanced_bytes(key) == default_output()


def test_sighting_enhancement_and_key_layout(
    client, db, s3mock, enhancer_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    reviewer_h = auth_headers(reviewer)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    photo = _upload_sighting_photo(
        client, headers, case_id, sighting_id
    )
    resp = _enhance_sighting(
        client, reviewer_h, case_id, sighting_id, photo["id"]
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETE"
    assert body["sighting_photo_id"] == photo["id"]
    assert body["case_photo_id"] is None
    assert body["sighting_id"] == sighting_id
    expected_sha = hashlib.sha256(default_output()).hexdigest()
    assert body["output_sha256"] == expected_sha
    key = storage.build_sighting_enhanced_key(
        case_id, sighting_id, photo["id"], expected_sha
    )
    assert storage.get_enhanced_bytes(key) == default_output()
    # Sighting trigger permission mirrors case photos.
    resp = _enhance_sighting(
        client, auth_headers(make_user(db, role=UserRole.REPORTER)),
        case_id, sighting_id, photo["id"],
    )
    assert resp.status_code == 403, resp.text


def test_failed_run_and_safe_errors(
    client, db, s3mock, enhancer_factory
):
    headers, case_id, photo = _setup_photo(client, db)
    enhancer_factory["fake"] = FakeEnhancer(fail_with="synthetic boom")
    resp = _enhance_case(client, headers, case_id, photo["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "FAILED"
    assert body["error_message"] == "synthetic boom"
    assert body["output_sha256"] is None
    # FAILED runs create no detections or embeddings downstream.
    assert client.post(
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        headers=headers,
    ).json()["face_count"] == 0
    # Phase 3 status is untouched by enhancement failure.
    assert _photo_row(
        CasePhoto, photo["id"]).processing_status == PhotoStatus.READY


def test_missing_model_is_controlled_failure(
    client, db, s3mock, monkeypatch
):
    headers, case_id, photo = _setup_photo(client, db)
    monkeypatch.setattr(
        settings, "ENHANCER_MODEL_PATH", "/nonexistent/weights.pth"
    )
    resp = _enhance_case(client, headers, case_id, photo["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "FAILED"
    # Safe message: no filesystem path leaks through the API.
    assert "nonexistent" not in body["error_message"]
    assert body["error_message"] == (
        "Image enhancement model is unavailable; try again later"
    )


def test_retry_failed_appends_new_run(
    client, db, s3mock, enhancer_factory
):
    headers, case_id, photo = _setup_photo(client, db)
    enhancer_factory["fake"] = FakeEnhancer(fail_with="boom")
    failed = _enhance_case(
        client, headers, case_id, photo["id"]
    ).json()
    assert failed["status"] == "FAILED"
    enhancer_factory["fake"] = FakeEnhancer()
    resp = client.post(
        "/cases/%d/photos/%d/enhancements/%d/retry"
        % (case_id, photo["id"], failed["id"]),
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["status"] == "COMPLETE"
    assert body["id"] != failed["id"]
    assert _run_count("case", photo["id"]) == 2


def test_retry_complete_conflicts(client, db, s3mock, enhancer_factory):
    headers, case_id, photo = _setup_photo(client, db)
    done = _enhance_case(
        client, headers, case_id, photo["id"]
    ).json()
    resp = client.post(
        "/cases/%d/photos/%d/enhancements/%d/retry"
        % (case_id, photo["id"], done["id"]),
        headers=headers,
    )
    assert resp.status_code == 409, resp.text


def test_stale_processing_allows_recovery(
    client, db, s3mock, enhancer_factory
):
    headers, case_id, photo = _setup_photo(client, db)
    for stale in (True, False):
        session = TestSession()
        try:
            row = session.get(CasePhoto, photo["id"])
            run = EnhancementRun(
                case_id=case_id,
                case_photo_id=photo["id"],
                sighting_photo_id=None,
                status=EnhancementStatus.PROCESSING,
                source_derived_sha256=row.derived_sha256,
                enhancer_name="fake-enhancer",
                enhancer_version="test-v1",
                model_name="fake-enhancer-model",
                model_version="test-mv1",
                model_sha256="1" * 64,
                mime_type="image/jpeg",
                started_at=(
                    datetime.now(timezone.utc)
                    - timedelta(seconds=700 if stale else 5)
                ),
            )
            session.add(run)
            session.commit()
            run_id = run.id
        finally:
            session.close()
        if stale:
            resp = client.post(
                "/cases/%d/photos/%d/enhancements/%d/retry"
                % (case_id, photo["id"], run_id),
                headers=headers,
            )
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "COMPLETE"
        else:
            # Fresh PROCESSING blocks both trigger and retry.
            resp = _enhance_case(client, headers, case_id, photo["id"])
            assert resp.status_code == 409, resp.text
            resp = client.post(
                "/cases/%d/photos/%d/enhancements/%d/retry"
                % (case_id, photo["id"], run_id),
                headers=headers,
            )
            assert resp.status_code == 409, resp.text
            # Clean up the fresh row so later tests stay isolated.
            session = TestSession()
            try:
                session.delete(session.get(EnhancementRun, run_id))
                session.commit()
            finally:
                session.close()


def test_multiple_runs_coexist_without_dedup(
    client, db, s3mock, enhancer_factory
):
    headers, case_id, photo = _setup_photo(client, db)
    first = _enhance_case(
        client, headers, case_id, photo["id"]
    ).json()
    second = _enhance_case(
        client, headers, case_id, photo["id"]
    ).json()
    assert first["status"] == "COMPLETE"
    assert second["status"] == "COMPLETE"
    assert first["id"] != second["id"]
    # Same model/parameters requested twice: separate runs, no
    # silent dedup.
    assert _run_count("case", photo["id"]) == 2
    resp = client.get(
        "/cases/%d/photos/%d/enhancements" % (case_id, photo["id"]),
        headers=headers,
    )
    assert [r["id"] for r in resp.json()] == [
        second["id"], first["id"],
    ]


# ---- Storage ----

def test_enhanced_key_layout_and_prefix_guards(
    client, db, s3mock, enhancer_factory
):
    assert storage.build_enhanced_key(3, 7, "a" * 64) == (
        "enhanced/3/7/" + "a" * 64 + ".jpg"
    )
    assert storage.build_sighting_enhanced_key(
        3, 9, 7, "b" * 64
    ) == "enhanced/sightings/3/9/7/" + "b" * 64 + ".jpg"
    assert storage.enhanced_photo_prefix(3, 7) == "enhanced/3/7/"
    assert storage.enhanced_sighting_photo_prefix(3, 9, 7) == (
        "enhanced/sightings/3/9/7/"
    )
    # Guards stay strict: public helpers cannot address other
    # scopes, and the derived helpers cannot address enhanced/.
    for bad in (
        "derived/3/7/x.jpg", "originals/3/7/x.jpg", "enhanced",
        "other/3/7/x.jpg",
    ):
        with pytest.raises(ValueError):
            storage.put_enhanced(bad, b"data", "image/jpeg")
        with pytest.raises(ValueError):
            storage.get_enhanced_bytes(bad)
    with pytest.raises(ValueError):
        storage.put_derived("enhanced/3/7/x.jpg", b"data", "image/jpeg")
    with pytest.raises(ValueError):
        storage.get_derived_bytes("enhanced/3/7/x.jpg")


def test_storage_failure_marks_failed(
    client, db, s3mock, enhancer_factory, monkeypatch
):
    headers, case_id, photo = _setup_photo(client, db)

    def _boom(key, data, content_type):
        raise RuntimeError("disk is gone")

    monkeypatch.setattr(storage, "put_enhanced", _boom)
    resp = _enhance_case(client, headers, case_id, photo["id"])
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "FAILED"
    assert "unavailable" in body["error_message"]


# ---- Face detection on enhanced sources ----

def _setup_enhanced(client, db, s3mock, fake_factory, enhancer_factory):
    """Case photo with one COMPLETE enhancement; returns context."""
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    run = _enhance_case(
        client, headers, case_id, photo["id"]
    ).json()
    assert run["status"] == "COMPLETE"
    return headers, case_id, photo, run


def test_normal_detection_path_unchanged(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, _ = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    # Even with an enhancement present, plain /detect stays on the
    # Phase 3 source: no silent latest-enhancement selection.
    body = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_face(),
    )
    run = body["run"]
    assert run["source_type"] == "DERIVED"
    assert run["enhancement_run_id"] is None
    assert run["source_sha256"] == photo["derived_sha256"]
    assert run["source_derived_sha"] == photo["derived_sha256"]
    faces = body["faces"]
    assert len(faces) == 1
    assert faces[0]["frame_width"] == photo["derived_width"]
    assert faces[0]["frame_height"] == photo["derived_height"]


def test_enhanced_detection_explicit(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    body = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )
    run = body["run"]
    assert run["source_type"] == "ENHANCED"
    assert run["enhancement_run_id"] == erun["id"]
    assert run["source_sha256"] == erun["output_sha256"]
    # Grandparent pointer still identifies the Phase 3 source.
    assert run["source_derived_sha"] == photo["derived_sha256"]
    # Frame dimensions follow the actual enhanced source.
    assert run["source_width"] == 32
    assert run["source_height"] == 24
    faces = body["faces"]
    assert len(faces) == 1
    assert faces[0]["frame_width"] == 32
    assert faces[0]["frame_height"] == 24
    # The detector consumed enhanced bytes, not derived bytes.
    fake = fake_factory["fake"]
    assert len(fake.calls) == 1
    assert fake.calls[0][0] == default_output()


def test_enhanced_redetect_appends_and_coexists(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    normal = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_face(),
    )
    first = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/redetect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )
    second = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/redetect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )
    assert first["run"]["id"] != second["run"]["id"]
    assert first["run"]["source_type"] == "ENHANCED"
    # Normal and enhanced records coexist as separate runs.
    kinds = sorted(
        r["source_type"] for r in (normal["run"], first["run"])
    )
    assert kinds == ["DERIVED", "ENHANCED"]


def _tamper_enhanced(case_id, photo_id, output_sha):
    key = storage.build_enhanced_key(case_id, photo_id, output_sha)
    storage.put_enhanced(key, b"not-an-image", "image/jpeg")


def test_enhanced_source_validity_conditions(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    url = "/cases/%d/photos/%d/faces/detect" % (
        case_id, photo["id"])
    fake_factory["fake"] = FakeDetector(faces=_one_enhanced_face())
    # 1. Run does not exist.
    resp = client.post(
        url + "?enhancement_run_id=999999", headers=headers
    )
    assert resp.status_code == 404, resp.text
    # 2. Run is not COMPLETE (FAILED here).
    enhancer_factory["fake"] = FakeEnhancer(fail_with="x")
    failed = _enhance_case(
        client, headers, case_id, photo["id"]).json()
    enhancer_factory["fake"] = FakeEnhancer()
    resp = client.post(
        url + "?enhancement_run_id=%d" % failed["id"], headers=headers
    )
    assert resp.status_code == 409, resp.text
    # 3. Output object deleted from storage (row still COMPLETE).
    storage.delete_prefix(
        storage.enhanced_photo_prefix(case_id, photo["id"]))
    resp = client.post(
        url + "?enhancement_run_id=%d" % erun["id"], headers=headers
    )
    assert resp.status_code == 409, resp.text
    # Restore the artifact, then 4. tamper with stored bytes.
    storage.put_enhanced(
        storage.build_enhanced_key(
            case_id, photo["id"], erun["output_sha256"]),
        default_output(), "image/jpeg",
    )
    _tamper_enhanced(case_id, photo["id"], erun["output_sha256"])
    resp = client.post(
        url + "?enhancement_run_id=%d" % erun["id"], headers=headers
    )
    assert resp.status_code == 409, resp.text
    # Restore again, then 5. rotate the Phase 3 source out from
    # under the enhancement (simulated reprocess).
    storage.put_enhanced(
        storage.build_enhanced_key(
            case_id, photo["id"], erun["output_sha256"]),
        default_output(), "image/jpeg",
    )
    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.derived_sha256 = "e" * 64
        session.commit()
    finally:
        session.close()
    resp = client.post(
        url + "?enhancement_run_id=%d" % erun["id"], headers=headers
    )
    assert resp.status_code == 409, resp.text


# ---- Embeddings on enhanced sources ----

def _setup_embedded(client, db, s3mock, fake_factory, enhancer_factory):
    """Case photo with normal + enhanced detection runs + faces."""
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    normal = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_face(),
    )
    enhanced = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )
    session = TestSession()
    try:
        nfaces = (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id == normal["run"]["id"])
            .order_by(FaceDetection.ordinal.asc()).all()
        )
        efaces = (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id == enhanced["run"]["id"])
            .order_by(FaceDetection.ordinal.asc()).all()
        )
        for row in list(nfaces) + list(efaces):
            session.expunge(row)
        return headers, case_id, photo, erun, nfaces, efaces
    finally:
        session.close()


def test_normal_and_enhanced_embeddings_coexist(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun, nfaces, efaces = _setup_embedded(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    session = TestSession()
    try:
        normal_row = _sface_embed(
            session, nfaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        normal_id = normal_row.id
        session.expunge(normal_row)
        enhanced_row = _sface_embed(
            session, efaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        # Same depicted face, separate valid embeddings.
        assert enhanced_row.id != normal_id
        assert enhanced_row.source_type == ImageSourceType.ENHANCED
        assert enhanced_row.source_sha256 == erun["output_sha256"]
        assert enhanced_row.source_derived_sha256 == (
            photo["derived_sha256"]
        )
        session.expunge(enhanced_row)
        # Idempotent per source: re-representing reuses both rows
        # without recompute.
        again_normal = _sface_embed(
            session, nfaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        assert again_normal.id == normal_id
    finally:
        session.close()


def test_second_enhancement_yields_third_embedding(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun, nfaces, efaces = _setup_embedded(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    erun_b = _enhance_case(
        client, headers, case_id, photo["id"]).json()
    assert erun_b["id"] != erun["id"]
    enhanced_b = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun_b["id"],
    )
    session = TestSession()
    try:
        bfaces = (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id == enhanced_b["run"]["id"])
            .order_by(FaceDetection.ordinal.asc()).all()
        )
        for row in bfaces:
            session.expunge(row)
        emb_n = _sface_embed(
            session, nfaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        emb_a = _sface_embed(
            session, efaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        emb_b = _sface_embed(
            session, bfaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        assert len({emb_n.id, emb_a.id, emb_b.id}) == 3
    finally:
        session.close()


def test_tampered_enhanced_source_rejects_embedding(
    client, db, s3mock, fake_factory, enhancer_factory
):
    from app.services.face_representation import RepresentationError

    headers, case_id, photo, erun, nfaces, efaces = _setup_embedded(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    _tamper_enhanced(case_id, photo["id"], erun["output_sha256"])
    session = TestSession()
    try:
        with pytest.raises(RepresentationError):
            _sface_embed(
                session, efaces[0],
                _photo_row(CasePhoto, photo["id"]), E1,
            )
        # The normal embedding still works: tampering is scoped to
        # the enhanced source.
        row = _sface_embed(
            session, nfaces[0],
            _photo_row(CasePhoto, photo["id"]), E1,
        )
        assert row.source_type == ImageSourceType.DERIVED
    finally:
        session.close()


def test_stale_enhancement_rejects_embedding(
    client, db, s3mock, fake_factory, enhancer_factory
):
    from app.services.face_representation import RepresentationError

    headers, case_id, photo, erun, nfaces, efaces = _setup_embedded(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    session = TestSession()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.derived_sha256 = "e" * 64
        session.commit()
    finally:
        session.close()
    session = TestSession()
    try:
        with pytest.raises(RepresentationError):
            _sface_embed(
                session, efaces[0],
                _photo_row(CasePhoto, photo["id"]), E1,
            )
    finally:
        session.close()


# ---- Source-aware similarity ----

def _setup_similarity_world(
    client, db, s3mock, fake_factory, enhancer_factory
):
    """Org case photo (normal+enhanced) vs sighting (normal+enhanced)."""
    org_owner = make_user(db, role=UserRole.ADMIN)
    org = make_org(db, org_owner, name="EnhSim")
    admin = make_user(db, role=UserRole.ADMIN)
    admin_h = auth_headers(admin)
    inv = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, inv, org, OrgRole.INVESTIGATOR)
    inv_h = auth_headers(inv)

    case_id = _make_case(client, admin_h, organization_id=org.id)
    photo = _upload_case_photo(client, admin_h, case_id)
    erun = _enhance_case(
        client, admin_h, case_id, photo["id"]).json()
    normal = _detect(
        client, admin_h,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_face(),
    )
    enhanced = _detect(
        client, admin_h,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )

    sighting_case = _make_case(
        client, admin_h, organization_id=org.id)
    sighting_id = _make_sighting(client, admin_h, sighting_case)
    sphote = _upload_sighting_photo(
        client, admin_h, sighting_case, sighting_id)
    serun = _enhance_sighting(
        client, admin_h, sighting_case, sighting_id,
        sphote["id"]).json()
    snormal = _detect(
        client, admin_h,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (sighting_case, sighting_id, sphote["id"]),
        fake_factory, _one_face(),
    )
    senhanced = _detect(
        client, admin_h,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (sighting_case, sighting_id, sphote["id"]),
        fake_factory, _one_enhanced_face(),
        params="?enhancement_run_id=%d" % serun["id"],
    )

    session = TestSession()
    try:
        faces = {}
        for key, run_id, pid, model in (
            ("n", normal["run"]["id"], photo["id"], CasePhoto),
            ("e", enhanced["run"]["id"], photo["id"], CasePhoto),
        ):
            rows = (
                session.query(FaceDetection)
                .filter(FaceDetection.run_id == run_id)
                .order_by(FaceDetection.ordinal.asc()).all()
            )
            for row in rows:
                session.expunge(row)
            faces[key] = rows[0]
            _sface_embed(
                session, rows[0], _photo_row(model, pid), E1)
        for key, run_id, pid in (
            ("sn", snormal["run"]["id"], sphote["id"]),
            ("se", senhanced["run"]["id"], sphote["id"]),
        ):
            rows = (
                session.query(FaceDetection)
                .filter(FaceDetection.run_id == run_id)
                .order_by(FaceDetection.ordinal.asc()).all()
            )
            for row in rows:
                session.expunge(row)
            faces[key] = rows[0]
            _sface_embed(
                session, rows[0],
                _photo_row(SightingPhoto, pid), E1)
    finally:
        session.close()
    return {
        "inv_h": inv_h, "case_id": case_id, "photo": photo,
        "erun": erun, "sighting_case": sighting_case,
        "sighting_id": sighting_id, "sphoto": sphote,
        "serun": serun, "faces": faces,
    }


def _search_case_face(client, headers, world, face_key):
    face = world["faces"][face_key]
    return client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (world["case_id"], world["photo"]["id"], face.id),
        json={"top_k": 10, "threshold": -1.0},
        headers=headers,
    )


def _search_sighting_face(client, headers, world, face_key):
    face = world["faces"][face_key]
    return client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar" % (
            world["sighting_case"], world["sighting_id"],
            world["sphoto"]["id"], face.id),
        json={"top_k": 10, "threshold": -1.0},
        headers=headers,
    )


def test_similarity_all_source_combinations(
    client, db, s3mock, fake_factory, enhancer_factory
):
    world = _setup_similarity_world(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    # normal -> {normal, enhanced}: one photo, two candidates.
    resp = _search_case_face(client, world["inv_h"], world, "n")
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 2
    assert {r["photo_id"] for r in results} == {world["sphoto"]["id"]}
    assert sorted(r["source_type"] for r in results) == [
        "DERIVED", "ENHANCED",
    ]
    # enhanced -> {normal, enhanced}.
    resp = _search_case_face(client, world["inv_h"], world, "e")
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["results"]) == 2
    # Reverse direction: sighting normal and enhanced queries each
    # see both case representations.
    for key in ("sn", "se"):
        resp = _search_sighting_face(
            client, world["inv_h"], world, key)
        assert resp.status_code == 200, resp.text
        results = resp.json()["results"]
        assert len(results) == 2, resp.text
        assert {r["photo_id"] for r in results} == {
            world["photo"]["id"]
        }


def test_similarity_no_merging_and_provenance(
    client, db, s3mock, fake_factory, enhancer_factory
):
    world = _setup_similarity_world(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    resp = _search_case_face(client, world["inv_h"], world, "n")
    results = resp.json()["results"]
    # No score merging: two separate candidates, one per source.
    assert len(results) == 2
    assert results[0]["face_id"] != results[1]["face_id"]
    by_source = {r["source_type"]: r for r in results}
    assert by_source["DERIVED"]["enhancement_run_id"] is None
    assert by_source["ENHANCED"]["enhancement_run_id"] == (
        world["serun"]["id"]
    )
    assert by_source["ENHANCED"]["photo_type"] == "sighting"
    assert by_source["ENHANCED"]["sighting_id"] == world["sighting_id"]


def test_similarity_excludes_tampered_enhanced(
    client, db, s3mock, fake_factory, enhancer_factory
):
    world = _setup_similarity_world(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    _tamper_sighting_enhanced(
        world["sighting_case"], world["sighting_id"],
        world["sphoto"]["id"], world["serun"]["output_sha256"],
    )
    resp = _search_case_face(client, world["inv_h"], world, "n")
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    # Only the valid normal candidate remains; the tampered
    # enhanced source is excluded without touching the query.
    assert len(results) == 1
    assert results[0]["source_type"] == "DERIVED"


def _tamper_sighting_enhanced(
    case_id, sighting_id, photo_id, output_sha
):
    key = storage.build_sighting_enhanced_key(
        case_id, sighting_id, photo_id, output_sha)
    storage.put_enhanced(key, b"not-an-image", "image/jpeg")


# ---- Deletion / cascade ----

def test_photo_deletion_sweeps_enhanced(
    client, db, s3mock, fake_factory, enhancer_factory
):
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )
    _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        fake_factory, _one_face(),
    )
    key = storage.build_enhanced_key(
        case_id, photo["id"], erun["output_sha256"])
    assert storage.get_enhanced_bytes(key) == default_output()
    resp = client.delete(
        "/cases/%d/photos/%d" % (case_id, photo["id"]),
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    with pytest.raises(Exception):
        storage.get_enhanced_bytes(key)
    assert _run_count("case", photo["id"]) == 0
    # The run is gone with its photo: detail reads as not found.
    resp = client.get(
        "/cases/%d/photos/%d/enhancements/%d"
        % (case_id, photo["id"], erun["id"]),
        headers=headers,
    )
    assert resp.status_code == 404, resp.text


def test_case_deletion_sweeps_enhanced(
    client, db, s3mock, fake_factory, enhancer_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    erun = _enhance_case(
        client, headers, case_id, photo["id"]).json()
    sighting_id = _make_sighting(client, headers, case_id)
    sphote = _upload_sighting_photo(
        client, headers, case_id, sighting_id)
    serun = _enhance_sighting(
        client, headers, case_id, sighting_id,
        sphote["id"]).json()
    resp = client.delete("/cases/%d" % case_id, headers=headers)
    assert resp.status_code == 200, resp.text
    with pytest.raises(Exception):
        storage.get_enhanced_bytes(storage.build_enhanced_key(
            case_id, photo["id"], erun["output_sha256"]))
    with pytest.raises(Exception):
        storage.get_enhanced_bytes(
            storage.build_sighting_enhanced_key(
                case_id, sighting_id, sphote["id"],
                serun["output_sha256"]))
    assert _run_count("case", photo["id"]) == 0
    assert _run_count("sighting", sphote["id"]) == 0


def test_sighting_deletion_sweeps_enhanced(
    client, db, s3mock, fake_factory, enhancer_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    sphote = _upload_sighting_photo(
        client, headers, case_id, sighting_id)
    serun = _enhance_sighting(
        client, headers, case_id, sighting_id,
        sphote["id"]).json()
    resp = client.delete(
        "/cases/%d/sightings/%d" % (case_id, sighting_id),
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    with pytest.raises(Exception):
        storage.get_enhanced_bytes(
            storage.build_sighting_enhanced_key(
                case_id, sighting_id, sphote["id"],
                serun["output_sha256"]))
    assert _run_count("sighting", sphote["id"]) == 0


def test_storage_failure_blocks_photo_deletion(
    client, db, s3mock, fake_factory, enhancer_factory, monkeypatch
):
    headers, case_id, photo, erun = _setup_enhanced(
        client, db, s3mock, fake_factory, enhancer_factory
    )

    def _boom(prefix):
        raise RuntimeError("storage is gone")

    monkeypatch.setattr(storage, "delete_prefix", _boom)
    resp = client.delete(
        "/cases/%d/photos/%d" % (case_id, photo["id"]),
        headers=headers,
    )
    assert resp.status_code == 502, resp.text
    # Nothing destructive proceeded: photo and run rows survive.
    resp = client.get(
        "/cases/%d/photos/%d" % (case_id, photo["id"]),
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert _run_count("case", photo["id"]) == 1
