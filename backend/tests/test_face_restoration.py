"""Phase 8 integration tests: face-level GFPGAN restoration.

S3 is mocked with moto; the suite still uses the disposable SQLite
database, so nothing touches real storage or the dev database.

The GFPGAN production adapter is replaced with FakeRestorer
(tests/fake_restorer.py) via the service factory, so every test is
deterministic (gfpgan package/weights never required). The
isolated real-model path lives in test_gfpgan_real.py (skipped
without weights/runtime).

Covers both photo types symmetrically: preparation determinism
and geometry, explicit face selection, DERIVED-only input
(ENHANCED rejection, no chaining), run lifecycle and per-face
active guard, retry, multi-face independence, restored SFace
embeddings and coexistence, restoration-aware similarity with
synthesized-detail provenance, storage layout/guards/cleanup,
and deletion.

Run from backend/:  python -m pytest tests/test_face_restoration.py -v
"""

import hashlib
import io
import json
from datetime import datetime, timedelta, timezone

import pytest
from moto import mock_aws
from PIL import Image

from app.core.config import settings
from app.models.case_photo import CasePhoto, PhotoStatus  # noqa: F401
from app.models.enhancement import (  # noqa: F401
    EnhancementRun,
    EnhancementStatus,
)
from app.models.face_detection import (  # noqa: F401
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.face_embedding import FaceEmbedding  # noqa: F401
from app.models.face_restoration import (  # noqa: F401
    FaceRestorationRun,
    FaceRestorationStatus,
)
from app.models.organization import OrgRole
from app.models.sighting import Sighting  # noqa: F401
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import face_restoration_service, storage
from app.services.face_preparation import (
    CANONICAL_GEOMETRY_VERSION,
    FACE_PREP_VERSION,
    GFPGAN_INPUT_SIZE,
    PreparationError,
    canonical_restored_landmarks,
    map_landmarks_to_prepared,
    prepare_face,
)
from app.services.face_representation import REPRESENTATION_LANDMARK_NAMES
from tests.conftest import (
    TestSession,
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from tests.fake_detector import FakeDetector, box
from tests.fake_enhancer import FakeEnhancer
from tests.fake_representation import FakeRepresentation, yunet_landmarks
from tests.fake_restorer import FakeRestorer, default_output

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


@pytest.fixture()
def detector_factory(monkeypatch):
    """Route the production detector factory to a FakeDetector."""
    holder = {"fake": FakeDetector()}
    from app.services import yunet_detector

    monkeypatch.setattr(
        yunet_detector, "get_face_detector", lambda: holder["fake"]
    )
    return holder


@pytest.fixture()
def restorer_factory(monkeypatch):
    """Route the production restorer factory to a FakeRestorer."""
    holder = {"fake": FakeRestorer()}
    from app.services import gfpgan_adapter

    monkeypatch.setattr(
        gfpgan_adapter, "get_face_restorer", lambda: holder["fake"]
    )
    return holder


@pytest.fixture()
def representation_factory(monkeypatch):
    """Route the production representation factory to a fake."""
    holder = {"fake": FakeRepresentation(
        representation_name="sface",
        representation_version="2021dec",
        model_name="sface",
        model_version="2021dec",
    )}
    from app.services import sface_representation

    monkeypatch.setattr(
        sface_representation,
        "get_face_representation",
        lambda: holder["fake"],
    )
    return holder


@pytest.fixture()
def enhancer_factory(monkeypatch):
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


def _detect(client, headers, url, factory, faces, params=""):
    factory["fake"]._faces = list(faces)
    factory["fake"].fail_with = None
    resp = client.post(url + params, headers=headers)
    assert resp.status_code == 200, resp.text
    return resp.json()


def _one_face():
    return [box(5, 5, 20, 20, 0.95, yunet_landmarks(6, 6))]


def _three_faces():
    return [
        box(2, 2, 14, 14, 0.95, yunet_landmarks(3, 3)),
        box(20, 5, 34, 19, 0.90, yunet_landmarks(21, 6)),
        box(40, 20, 55, 35, 0.85, yunet_landmarks(41, 21)),
    ]


def _setup_face(client, db, factory, role=UserRole.ADMIN, faces=None):
    user = make_user(db, role=role)
    headers = auth_headers(user)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    body = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        factory, faces if faces is not None else _one_face(),
    )
    assert body["run"]["status"] == "COMPLETE"
    return user, headers, case_id, photo, body["faces"]


def _restore_url(case_id, photo_id, face_id, run=""):
    return "/cases/%d/photos/%d/faces/%d/restorations%s" % (
        case_id, photo_id, face_id, run
    )


def _restore(client, headers, case_id, photo_id, face_id):
    resp = client.post(
        _restore_url(case_id, photo_id, face_id), headers=headers
    )
    return resp


def _db():
    return TestSession()


# ---------------- preparation unit tests ----------------


def test_prep_deterministic_and_versioned():
    first = prepare_face(
        _image_bytes(), 64, 48, 5, 5, 20, 20, yunet_landmarks(6, 6)
    )
    second = prepare_face(
        _image_bytes(), 64, 48, 5, 5, 20, 20, yunet_landmarks(6, 6)
    )
    assert first[0] == second[0]
    assert first[1] == second[1] == hashlib.sha256(first[0]).hexdigest()
    assert (first[2], first[3]) == (
        GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE
    )
    assert first[5]["prep_version"] == FACE_PREP_VERSION
    assert first[5]["output_width"] == GFPGAN_INPUT_SIZE


def test_prep_edge_face_pads_and_records():
    # Full-frame box forces a square larger than the frame height.
    _, _, _, _, _, transform = prepare_face(
        _image_bytes(), 64, 48, 0, 0, 64, 48, yunet_landmarks(6, 6)
    )
    assert transform["crop_side"] == 64
    assert transform["pad_top"] + transform["pad_bottom"] > 0
    assert transform["pad_left"] == 0
    assert transform["pad_right"] == 0


def test_prep_rejects_missing_and_malformed_landmarks():
    with pytest.raises(PreparationError):
        prepare_face(_image_bytes(), 64, 48, 5, 5, 20, 20, None)
    bad = yunet_landmarks(6, 6)
    bad["nose_tip"] = ["x", None]
    with pytest.raises(PreparationError):
        prepare_face(_image_bytes(), 64, 48, 5, 5, 20, 20, bad)
    with pytest.raises(PreparationError):
        prepare_face(_image_bytes(), 64, 48, 20, 20, 5, 5,
                     yunet_landmarks(6, 6))


def test_mapped_landmark_math_uses_actual_transform():
    transform = {
        "crop_x0": 10,
        "crop_y0": 4,
        "pad_left": 0,
        "pad_top": 0,
        "scale": 2.0,
    }
    mapped = map_landmarks_to_prepared(
        yunet_landmarks(20, 14), transform
    )
    # right_eye (20,14) -> ((20-10)*2, (14-4)*2) == (20, 20).
    assert mapped["right_eye"] == [20.0, 20.0]
    assert set(mapped) == set(REPRESENTATION_LANDMARK_NAMES)


def test_canonical_landmarks_versioned_and_in_frame():
    landmarks = canonical_restored_landmarks(256, 256)
    assert set(landmarks) == set(REPRESENTATION_LANDMARK_NAMES)
    for _, (x, y) in landmarks.items():
        assert 0 < x < 256
        assert 0 < y < 256
    assert CANONICAL_GEOMETRY_VERSION.startswith("phase8-")


# ---------------- trigger / lifecycle ----------------


def test_trigger_requires_admin_or_reviewer(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory, role=UserRole.ADMIN
    )
    outsider = make_user(db, role=UserRole.REPORTER)
    resp = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"]),
        headers=auth_headers(outsider),
    )
    assert resp.status_code == 403
    reviewer = make_user(db, role=UserRole.REVIEWER)
    resp = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"]),
        headers=auth_headers(reviewer),
    )
    assert resp.status_code == 201, resp.text


def test_trigger_records_full_provenance(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    face = faces[0]
    resp = _restore(client, headers, case_id, photo["id"], face["id"])
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["status"] == "COMPLETE"
    assert run["face_detection_id"] == face["id"]
    assert run["source_derived_sha256"] == photo["derived_sha256"]
    assert run["prep_version"] == FACE_PREP_VERSION
    assert len(run["prepared_input_sha256"]) == 64
    assert run["restorer_name"] == "fake-restorer"
    assert run["model_sha256"] == "2" * 64
    assert run["restored_geometry_kind"] == "MAPPED"
    assert run["restored_geometry_version"] == FACE_PREP_VERSION
    assert run["output_sha256"] == hashlib.sha256(
        default_output()).hexdigest()
    key = storage.build_restored_face_key(
        case_id, photo["id"], face["id"], run["output_sha256"]
    )
    assert key.startswith(
        "restored-faces/%d/%d/%d/" % (case_id, photo["id"], face["id"])
    )
    assert key.endswith(run["output_sha256"] + ".jpg")
    assert run["mime_type"] == "image/jpeg"
    assert (run["width"], run["height"]) == (
        GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE
    )
    assert run["byte_size"] == len(default_output())
    # Restorer consumed prepared bytes, not derived bytes.
    fake = restorer_factory["fake"]
    assert len(fake.calls) == 1
    assert hashlib.sha256(fake.calls[0]).hexdigest() == (
        run["prepared_input_sha256"]
    )


def test_trigger_wrong_photo_face_is_404(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    other = _upload_case_photo(client, headers, case_id)
    resp = client.post(
        _restore_url(case_id, other["id"], faces[0]["id"]),
        headers=headers,
    )
    assert resp.status_code == 404


def test_trigger_stale_detection_is_409(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    session = _db()
    try:
        row = session.get(CasePhoto, photo["id"])
        row.derived_sha256 = "f" * 64
        session.commit()
    finally:
        session.close()
    resp = _restore(client, headers, case_id, photo["id"], faces[0]["id"])
    assert resp.status_code == 409


def test_trigger_rejects_enhanced_detection(
    client, db, s3mock, detector_factory, restorer_factory,
    enhancer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    eresp = client.post(
        "/cases/%d/photos/%d/enhancements" % (case_id, photo["id"]),
        headers=headers,
    )
    assert eresp.status_code == 201, eresp.text
    erun = eresp.json()
    body = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        detector_factory, _one_face(),
        params="?enhancement_run_id=%d" % erun["id"],
    )
    assert body["run"]["source_type"] == "ENHANCED"
    enhanced_face = body["faces"][0]
    resp = _restore(
        client, headers, case_id, photo["id"], enhanced_face["id"]
    )
    assert resp.status_code == 409
    assert "derived" in resp.json()["detail"].lower()


def test_failed_run_and_retry(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    restorer_factory["fake"].fail_with = "Fake restoration exploded"
    resp = _restore(client, headers, case_id, photo["id"], faces[0]["id"])
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["status"] == "FAILED"
    assert run["error_message"] == "Fake restoration exploded"
    assert run["output_sha256"] is None
    restorer_factory["fake"].fail_with = None
    retry = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/%d/retry" % run["id"]),
        headers=headers,
    )
    assert retry.status_code == 200, retry.text
    second = retry.json()
    assert second["status"] == "COMPLETE"
    assert second["id"] != run["id"]
    # COMPLETE runs cannot be retried; unknown runs are 404.
    again = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/%d/retry" % second["id"]),
        headers=headers,
    )
    assert again.status_code == 409
    missing = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/999999/retry"),
        headers=headers,
    )
    assert missing.status_code == 404


def test_stillborn_model_failure(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    restorer_factory["fake"].bad_model = True
    resp = _restore(client, headers, case_id, photo["id"], faces[0]["id"])
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["status"] == "FAILED"
    assert run["model_sha256"] == "0" * 64


def _insert_processing_run(photo_id, face_id, run_id, hours_old=0):
    session = _db()
    try:
        face = session.get(FaceDetection, face_id)
        photo = session.get(CasePhoto, photo_id)
        started = datetime.now(timezone.utc) - timedelta(hours=hours_old)
        run = FaceRestorationRun(
            case_id=photo.case_id,
            case_photo_id=photo.id,
            sighting_photo_id=None,
            sighting_id=None,
            face_detection_id=face.id,
            face_detection_run_id=face.run_id,
            status=FaceRestorationStatus.PROCESSING,
            source_derived_sha256=photo.derived_sha256,
            bbox_snapshot="{}",
            landmarks_snapshot="{}",
            prep_version=FACE_PREP_VERSION,
            prep_transform="{}",
            prepared_input_sha256="0" * 64,
            restorer_name="fake-restorer",
            restorer_version="test-v1",
            model_name="fake-restorer-model",
            model_version="test-mv1",
            model_sha256="2" * 64,
            restored_geometry_kind="MAPPED",
            restored_geometry_version=FACE_PREP_VERSION,
            restored_geometry="{}",
            mime_type="image/jpeg",
            started_at=started,
        )
        session.add(run)
        session.commit()
        return run.id
    finally:
        session.close()


def test_per_face_active_guard(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory, faces=_three_faces()
    )
    assert len(faces) == 3
    proc_id = _insert_processing_run(
        photo["id"], faces[0]["id"], faces[0]["run_id"]
    )
    # Same face blocked while its run is fresh...
    blocked = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    )
    assert blocked.status_code == 409
    # ...but another face proceeds independently.
    other = _restore(
        client, headers, case_id, photo["id"], faces[1]["id"]
    )
    assert other.status_code == 201, other.text
    assert other.json()["status"] == "COMPLETE"
    # Retry of a fresh PROCESSING run also conflicts.
    retry = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/%d/retry" % proc_id),
        headers=headers,
    )
    assert retry.status_code == 409


def test_stale_processing_retry_creates_new_run(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    proc_id = _insert_processing_run(
        photo["id"], faces[0]["id"], faces[0]["run_id"], hours_old=2
    )
    retry = client.post(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/%d/retry" % proc_id),
        headers=headers,
    )
    assert retry.status_code == 200, retry.text
    assert retry.json()["id"] != proc_id
    assert retry.json()["status"] == "COMPLETE"


def test_list_and_detail_with_presigned_url(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    first = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    second = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    listing = client.get(
        _restore_url(case_id, photo["id"], faces[0]["id"]),
        headers=headers,
    )
    assert listing.status_code == 200
    ids = [r["id"] for r in listing.json()]
    assert ids == [second["id"], first["id"]]
    detail = client.get(
        _restore_url(case_id, photo["id"], faces[0]["id"],
                     "/%d" % first["id"]),
        headers=headers,
    )
    assert detail.status_code == 200
    assert detail.json()["view_url"] is not None
    assert detail.json()["expires_in"] is not None
    missing = client.get(
        _restore_url(case_id, photo["id"], faces[0]["id"], "/999999"),
        headers=headers,
    )
    assert missing.status_code == 404


# ---------------- multi-face independence ----------------


def test_three_faces_restore_independently(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory, faces=_three_faces()
    )
    runs = [
        _restore(client, headers, case_id, photo["id"], f["id"]).json()
        for f in faces
    ]
    assert [r["status"] for r in runs] == ["COMPLETE"] * 3
    assert len({r["id"] for r in runs}) == 3
    keys = {
        storage.build_restored_face_key(
            case_id, photo["id"], f["id"], r["output_sha256"]
        )
        for f, r in zip(faces, runs)
    }
    assert len(keys) == 3  # face id scopes each key
    # Re-restoring Face A appends history; B/C untouched.
    again = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    assert again["id"] not in {r["id"] for r in runs}
    listing_b = client.get(
        _restore_url(case_id, photo["id"], faces[1]["id"]),
        headers=headers,
    ).json()
    assert [r["id"] for r in listing_b] == [runs[1]["id"]]
    listing_a = client.get(
        _restore_url(case_id, photo["id"], faces[0]["id"]),
        headers=headers,
    ).json()
    assert [r["id"] for r in listing_a] == [again["id"], runs[0]["id"]]


def test_canonical_geometry_when_realigned(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    restorer_factory["fake"] = FakeRestorer(
        realigned=True, width=256, height=256
    )
    resp = _restore(client, headers, case_id, photo["id"], faces[0]["id"])
    assert resp.status_code == 201, resp.text
    run = resp.json()
    assert run["restored_geometry_kind"] == "CANONICAL"
    assert run["restored_geometry_version"] == (
        CANONICAL_GEOMETRY_VERSION
    )
    assert (run["width"], run["height"]) == (256, 256)
    # Canonical geometry is usable by SFace.
    fake_rep = FakeRepresentation()
    session = _db()
    try:
        face = session.get(FaceDetection, faces[0]["id"])
        brow = session.get(CasePhoto, photo["id"])
        row = face_restoration_service.represent_restored_face(
            session, face, brow, run["id"], "case", fake_rep
        )
        assert row.face_restoration_run_id == run["id"]
    finally:
        session.close()


# ---------------- restored embeddings ----------------


def _sface_rep(vector=None):
    """FakeRepresentation carrying the production SFace identity.

    Similarity's retrieval identity is read from the production
    SFace adapter, so only embeddings stored under that identity
    participate in retrieval tests.
    """
    kwargs = {}
    if vector is not None:
        kwargs["vector"] = vector
    return FakeRepresentation(
        representation_name="sface",
        representation_version="2021dec",
        model_name="sface",
        model_version="2021dec",
        **kwargs,
    )


def _embed(client, headers, case_id, photo_id, face_id, run_id):
    return client.post(
        _restore_url(case_id, photo_id, face_id, "/%d/embedding" % run_id),
        headers=headers,
    )


def test_restored_embedding_uses_restored_frame(
    client, db, s3mock, detector_factory, restorer_factory,
    representation_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    run = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    fake_rep = FakeRepresentation()
    session = _db()
    try:
        face = session.get(FaceDetection, faces[0]["id"])
        brow = session.get(CasePhoto, photo["id"])
        row = face_restoration_service.represent_restored_face(
            session, face, brow, run["id"], "case", fake_rep
        )
        assert row.face_detection_id == face.id
        assert row.face_restoration_run_id == run["id"]
        assert row.source_derived_sha256 == photo["derived_sha256"]
        assert row.source_sha256 == run["output_sha256"]
        # Idempotent reuse.
        again = face_restoration_service.represent_restored_face(
            session, face, brow, run["id"], "case", fake_rep
        )
        assert again.id == row.id
    finally:
        session.close()
    # SFace consumed the artifact bytes with restored-frame
    # geometry: full-frame box, never the source YuNet coordinates.
    assert len(fake_rep.calls) == 1
    consumed_bytes, geometry = fake_rep.calls[0]
    assert consumed_bytes == default_output()
    assert (geometry.x_min, geometry.y_min) == (0, 0)
    assert (geometry.x_max, geometry.y_max) == (
        GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE
    )
    source_landmarks = yunet_landmarks(6, 6)
    assert geometry.landmarks != source_landmarks
    # Endpoint path agrees.
    resp = _embed(
        client, headers, case_id, photo["id"], faces[0]["id"],
        run["id"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["face_restoration_run_id"] == run["id"]


def test_normal_and_restored_embeddings_coexist(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    run = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    fake_rep = FakeRepresentation()
    session = _db()
    try:
        face = session.get(FaceDetection, faces[0]["id"])
        brow = session.get(CasePhoto, photo["id"])
        detection_run = session.get(
            FaceDetectionRun, face.run_id
        )
        from app.services import face_representation_service

        normal = face_representation_service.represent_face(
            session, face, brow, fake_rep, detection_run
        )
        assert normal.face_restoration_run_id is None
        restored = face_restoration_service.represent_restored_face(
            session, face, brow, run["id"], "case", fake_rep
        )
        assert restored.id != normal.id
        count = session.query(FaceEmbedding).filter(
            FaceEmbedding.face_detection_id == face.id
        ).count()
        assert count == 2
    finally:
        session.close()


def test_restored_uniqueness_per_run_and_content_dedup(
    client, db, s3mock, detector_factory, restorer_factory
):
    from sqlalchemy.exc import IntegrityError

    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    run_a = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    # Different output bytes -> second run + second embedding.
    restorer_factory["fake"] = FakeRestorer(
        output_bytes=default_output(color="red")
    )
    run_b = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    assert run_b["output_sha256"] != run_a["output_sha256"]
    fake_rep = FakeRepresentation()
    session = _db()
    try:
        face = session.get(FaceDetection, faces[0]["id"])
        brow = session.get(CasePhoto, photo["id"])
        row_a = face_restoration_service.represent_restored_face(
            session, face, brow, run_a["id"], "case", fake_rep
        )
        row_b = face_restoration_service.represent_restored_face(
            session, face, brow, run_b["id"], "case", fake_rep
        )
        assert row_a.id != row_b.id
        # Same (face, representation, run) twice violates the
        # partial unique index.
        dup = FaceEmbedding(
            face_detection_id=face.id,
            source_derived_sha256=brow.derived_sha256,
            source_type=row_a.source_type,
            source_sha256=row_a.source_sha256,
            face_restoration_run_id=run_a["id"],
            representation_name=row_a.representation_name,
            representation_version=row_a.representation_version,
            model_name=row_a.model_name,
            model_version=row_a.model_version,
            model_sha256=row_a.model_sha256,
            dimension=row_a.dimension,
            embedding=list(row_a.embedding),
        )
        session.add(dup)
        with pytest.raises(IntegrityError):
            session.commit()
        session.rollback()
    finally:
        session.close()


def test_restored_candidate_invalidation(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    run = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    fake_rep = FakeRepresentation()
    session = _db()
    try:
        face = session.get(FaceDetection, faces[0]["id"])
        brow = session.get(CasePhoto, photo["id"])
        row = face_restoration_service.represent_restored_face(
            session, face, brow, run["id"], "case", fake_rep
        )
        assert (
            face_restoration_service.validate_restoration_candidate(
                session, row, face, brow
            ) is not None
        )
        # Tampered artifact bytes.
        key = storage.build_restored_face_key(
            case_id, photo["id"], faces[0]["id"],
            run["output_sha256"],
        )
        storage.put_restored_face(key, b"tampered", "image/jpeg")
        assert (
            face_restoration_service.validate_restoration_candidate(
                session, row, face, brow
            ) is None
        )
        # Missing artifact.
        storage.delete_prefix(key)
        assert (
            face_restoration_service.validate_restoration_candidate(
                session, row, face, brow
            ) is None
        )
        # Stale derived SHA.
        brow.derived_sha256 = "e" * 64
        session.commit()
        assert (
            face_restoration_service.validate_restoration_candidate(
                session, row, face, brow
            ) is None
        )
    finally:
        session.close()


# ---------------- similarity ----------------


def _setup_world(client, db, detector_factory, restorer_factory):
    """Case photo + sighting photo, each with one restored face."""
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    photo = _upload_case_photo(client, headers, case_id)
    sighting_id = _make_sighting(client, headers, case_id)
    sp = _upload_sighting_photo(client, headers, case_id, sighting_id)
    case_body = _detect(
        client, headers,
        "/cases/%d/photos/%d/faces/detect" % (case_id, photo["id"]),
        detector_factory, _one_face(),
    )
    sight_body = _detect(
        client, headers,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, sp["id"]),
        detector_factory, _one_face(),
    )
    case_face = case_body["faces"][0]
    sight_face = sight_body["faces"][0]
    case_run = _restore(
        client, headers, case_id, photo["id"], case_face["id"]
    ).json()
    sight_run = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/restorations"
        % (case_id, sighting_id, sp["id"], sight_face["id"]),
        headers=headers,
    ).json()
    assert case_run["status"] == "COMPLETE"
    assert sight_run["status"] == "COMPLETE"
    session = _db()
    try:
        cface = session.get(FaceDetection, case_face["id"])
        cphoto = session.get(CasePhoto, photo["id"])
        face_restoration_service.represent_restored_face(
            session, cface, cphoto, case_run["id"], "case",
            _sface_rep(),
        )
        sface = session.get(FaceDetection, sight_face["id"])
        sphoto = session.get(SightingPhoto, sp["id"])
        face_restoration_service.represent_restored_face(
            session, sface, sphoto, sight_run["id"], "sighting",
            _sface_rep(),
        )
    finally:
        session.close()
    return (headers, case_id, photo, case_face, case_run,
            sighting_id, sp, sight_face, sight_run)


def test_similarity_restored_candidate_with_provenance(
    client, db, s3mock, detector_factory, restorer_factory
):
    (headers, case_id, photo, case_face, case_run,
     sighting_id, sp, sight_face, sight_run) = _setup_world(
        client, db, detector_factory, restorer_factory
    )
    resp = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (case_id, photo["id"], case_face["id"]),
        json={"top_k": 10, "threshold": 0.0,
              "restoration_run_id": case_run["id"]},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 1
    candidate = results[0]
    assert candidate["face_id"] == sight_face["id"]
    assert candidate["face_restoration_run_id"] == sight_run["id"]
    assert candidate["is_restored"] is True
    assert candidate["synthesized_detail_warning"] is not None
    assert "human review" in candidate["synthesized_detail_warning"]


def test_similarity_excludes_invalid_restored_candidate(
    client, db, s3mock, detector_factory, restorer_factory
):
    (headers, case_id, photo, case_face, case_run,
     sighting_id, sp, sight_face, sight_run) = _setup_world(
        client, db, detector_factory, restorer_factory
    )
    # Remove the candidate-side artifact: no valid candidates.
    storage.delete_prefix(
        storage.build_sighting_restored_face_key(
            case_id, sighting_id, sp["id"], sight_face["id"],
            sight_run["output_sha256"],
        )
    )
    resp = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (case_id, photo["id"], case_face["id"]),
        json={"top_k": 10, "threshold": 0.0,
              "restoration_run_id": case_run["id"]},
        headers=headers,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []
    # Querying with an unrestored embedding request still works.
    normal = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (case_id, photo["id"], case_face["id"]),
        json={"top_k": 10, "threshold": 0.0},
        headers=headers,
    )
    assert normal.status_code == 409  # no normal embedding exists


# ---------------- storage / deletion ----------------


def test_storage_keys_guards_and_deletion(
    client, db, s3mock, detector_factory, restorer_factory
):
    _, headers, case_id, photo, faces = _setup_face(
        client, db, detector_factory
    )
    run = _restore(
        client, headers, case_id, photo["id"], faces[0]["id"]
    ).json()
    key = storage.build_restored_face_key(
        case_id, photo["id"], faces[0]["id"], run["output_sha256"]
    )
    assert key == "restored-faces/%d/%d/%d/%s.jpg" % (
        case_id, photo["id"], faces[0]["id"], run["output_sha256"])
    assert storage.get_restored_face_bytes(key) == default_output()
    sighting_id = _make_sighting(client, headers, case_id)
    sp = _upload_sighting_photo(client, headers, case_id, sighting_id)
    body = _detect(
        client, headers,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, sp["id"]),
        detector_factory, _one_face(),
    )
    sface = body["faces"][0]
    srun = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/restorations"
        % (case_id, sighting_id, sp["id"], sface["id"]),
        headers=headers,
    ).json()
    skey = storage.build_sighting_restored_face_key(
        case_id, sighting_id, sp["id"], sface["id"],
        srun["output_sha256"],
    )
    assert skey == "restored-faces/sightings/%d/%d/%d/%d/%s.jpg" % (
        case_id, sighting_id, sp["id"], sface["id"],
        srun["output_sha256"])
    assert storage.get_restored_face_bytes(skey) == default_output()
    with pytest.raises(ValueError):
        storage.put_restored_face(
            "enhanced/1/2/x.jpg", b"data", "image/jpeg"
        )
    with pytest.raises(ValueError):
        storage.get_restored_face_bytes("derived/1/2/x.jpg")
    # Deleting the case photo removes restored objects + run rows.
    del_resp = client.delete(
        "/cases/%d/photos/%d" % (case_id, photo["id"]),
        headers=headers,
    )
    assert del_resp.status_code == 200, del_resp.text
    remaining = [
        obj["Key"] for obj in
        s3mock._client().list_objects_v2(
            Bucket="test-bucket", Prefix="restored-faces/"
        ).get("Contents", [])
    ]
    assert all(
        "/%d/%d/" % (case_id, photo["id"]) not in key
        for key in remaining
    )
    session = _db()
    try:
        count = session.query(FaceRestorationRun).filter(
            FaceRestorationRun.case_photo_id == photo["id"]
        ).count()
        assert count == 0
    finally:
        session.close()


def test_sighting_photo_deletion_cleans_restoration(
    client, db, s3mock, detector_factory, restorer_factory
):
    admin = make_user(db, role=UserRole.ADMIN)
    headers = auth_headers(admin)
    case_id = _make_case(client, headers)
    sighting_id = _make_sighting(client, headers, case_id)
    sp = _upload_sighting_photo(client, headers, case_id, sighting_id)
    body = _detect(
        client, headers,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_id, sighting_id, sp["id"]),
        detector_factory, _one_face(),
    )
    sface = body["faces"][0]
    srun = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/restorations"
        % (case_id, sighting_id, sp["id"], sface["id"]),
        headers=headers,
    ).json()
    assert srun["status"] == "COMPLETE"
    del_resp = client.delete(
        "/cases/%d/sightings/%d/photos/%d"
        % (case_id, sighting_id, sp["id"]),
        headers=headers,
    )
    assert del_resp.status_code == 200, del_resp.text
    session = _db()
    try:
        count = session.query(FaceRestorationRun).filter(
            FaceRestorationRun.sighting_photo_id == sp["id"]
        ).count()
        assert count == 0
    finally:
        session.close()