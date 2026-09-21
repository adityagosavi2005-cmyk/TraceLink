"""Phase 6 tests: face-similarity retrieval.

SQLite exercises the Python fallback path with the shared
cosine helpers; PostgreSQL ``<=>`` equivalence is covered by
tests/test_similarity_pgvector.py. Detection runs use the real
detection service with FakeDetector (scripted YuNet-style faces
WITH landmarks); embeddings use FakeRepresentation with
hand-built vectors so similarity values are exact.

Run from backend/:  python -m pytest tests/test_similarity.py -v
"""

import io

import pytest
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
from app.models.organization import OrgRole
from app.models.sighting_photo import SightingPhoto  # noqa: F401
from app.models.user import UserRole
from app.services import face_representation_service, similarity_service, storage
from tests.conftest import (
    TestSession,
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from tests.fake_detector import FakeDetector, box
from tests.fake_representation import (
    FakeRepresentation,
    yunet_landmarks,
)

assert CasePhoto is not None and SightingPhoto is not None
assert FaceDetection is not None and FaceDetectionRun is not None
assert FaceEmbedding is not None


def _vec(*vals: float) -> list[float]:
    out = [0.0] * 128
    for i, v in enumerate(vals):
        out[i] = v
    return out


E1 = _vec(1.0)
E2 = _vec(0.0, 1.0)
E_NEG = _vec(-1.0)


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


def _make_case(client, headers, org_id):
    resp = client.post(
        "/cases",
        json={
            "title": "T", "description": "D",
            "organization_id": org_id,
        },
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


def _detect(client, reviewer_headers, url, fake_factory, faces):
    fake_factory["fake"] = FakeDetector(faces=faces)
    resp = client.post(url, headers=reviewer_headers)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "COMPLETE", resp.text
    return resp.json()


def _faces_for_case_photo(photo_id):
    db = TestSession()
    try:
        rows = (
            db.query(FaceDetection)
            .filter(FaceDetection.case_photo_id == photo_id)
            .order_by(FaceDetection.ordinal.asc())
            .all()
        )
        for row in rows:
            db.expunge(row)
        return rows
    finally:
        db.close()


def _faces_for_sighting_photo(photo_id):
    db = TestSession()
    try:
        rows = (
            db.query(FaceDetection)
            .filter(FaceDetection.sighting_photo_id == photo_id)
            .order_by(FaceDetection.ordinal.asc())
            .all()
        )
        for row in rows:
            db.expunge(row)
        return rows
    finally:
        db.close()


def _photo_row(model, photo_id):
    db = TestSession()
    try:
        row = db.get(model, photo_id)
        db.expunge(row)
        return row
    finally:
        db.close()


def _embed(db, face, photo, vector):
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


def _one_face():
    return [box(5, 5, 20, 20, 0.95, yunet_landmarks(6, 6))]


def _two_faces():
    return [
        box(5, 5, 20, 20, 0.95, yunet_landmarks(6, 6)),
        box(30, 10, 55, 40, 0.8, yunet_landmarks(32, 12)),
    ]


def _setup_world(client, db, s3mock, fake_factory):
    """Two orgs; orgA holds query case + sighting candidates."""
    org_a = make_org(db, make_user(db, role=UserRole.ADMIN), name="SimA")
    org_b = make_org(db, make_user(db, role=UserRole.ADMIN), name="SimB")
    inv_a = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, inv_a, org_a, OrgRole.INVESTIGATOR)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    # VIEWER membership: keeps the reviewer read-only for
    # can_edit_case while scoping retrieval to orgA.
    make_membership(db, reviewer, org_a, OrgRole.VIEWER)
    admin = make_user(db, role=UserRole.ADMIN)
    reporter = make_user(db, role=UserRole.REPORTER)
    inv_h = auth_headers(inv_a)
    rev_h = auth_headers(reviewer)

    # Query side: case photo with one face.
    case_q = _make_case(client, inv_h, org_a.id)
    photo_q = _upload_case_photo(client, inv_h, case_q)
    _detect(
        client, rev_h,
        "/cases/%d/photos/%d/faces/detect" % (case_q, photo_q["id"]),
        fake_factory, _one_face(),
    )
    qface = _faces_for_case_photo(photo_q["id"])[0]
    session = TestSession()
    try:
        _embed(session, qface, _photo_row(CasePhoto, photo_q["id"]), E1)
    finally:
        session.close()

    # Candidate side: sighting photo with two faces.
    case_s = _make_case(client, inv_h, org_a.id)
    sighting = _make_sighting(client, inv_h, case_s)
    photo_s = _upload_sighting_photo(client, inv_h, case_s, sighting)
    _detect(
        client, rev_h,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_s, sighting, photo_s["id"]),
        fake_factory, _two_faces(),
    )
    sfaces = _faces_for_sighting_photo(photo_s["id"])
    session = TestSession()
    try:
        _embed(session, sfaces[0], _photo_row(
            SightingPhoto, photo_s["id"]), E1)
        _embed(session, sfaces[1], _photo_row(
            SightingPhoto, photo_s["id"]), E2)
    finally:
        session.close()

    # Same-type decoy: case photo face identical to the query.
    case_d = _make_case(client, inv_h, org_a.id)
    photo_d = _upload_case_photo(client, inv_h, case_d)
    _detect(
        client, rev_h,
        "/cases/%d/photos/%d/faces/detect" % (case_d, photo_d["id"]),
        fake_factory, _one_face(),
    )
    dface = _faces_for_case_photo(photo_d["id"])[0]
    session = TestSession()
    try:
        _embed(session, dface, _photo_row(CasePhoto, photo_d["id"]), E1)
    finally:
        session.close()

    # Foreign org candidate: identical vector, must be scoped out.
    inv_b = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, inv_b, org_b, OrgRole.INVESTIGATOR)
    inv_b_h = auth_headers(inv_b)
    case_b = _make_case(client, inv_b_h, org_b.id)
    sighting_b = _make_sighting(client, inv_b_h, case_b)
    photo_b = _upload_sighting_photo(client, inv_b_h, case_b, sighting_b)
    _detect(
        client, rev_h,
        "/cases/%d/sightings/%d/photos/%d/faces/detect"
        % (case_b, sighting_b, photo_b["id"]),
        fake_factory, _one_face(),
    )
    bface = _faces_for_sighting_photo(photo_b["id"])[0]
    session = TestSession()
    try:
        _embed(session, bface, _photo_row(
            SightingPhoto, photo_b["id"]), E1)
    finally:
        session.close()

    return {
        "org_a": org_a, "org_b": org_b,
        "inv_a": inv_a, "inv_h": inv_h,
        "reviewer": reviewer, "rev_h": rev_h,
        "admin": admin, "admin_h": auth_headers(admin),
        "reporter": reporter, "rep_h": auth_headers(reporter),
        "case_q": case_q, "photo_q": photo_q, "qface": qface,
        "case_s": case_s, "sighting": sighting, "photo_s": photo_s,
        "sfaces": sfaces,
        "case_b": case_b, "sighting_b": sighting_b, "photo_b": photo_b,
        "bface": bface,
    }


def _search(client, headers, world, payload):
    return client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (world["case_q"], world["photo_q"]["id"],
           world["qface"].id),
        json=payload,
        headers=headers,
    )


# ---- Metric behavior ----

def test_cosine_helpers_match_hand_computed_values():
    assert similarity_service.cosine_distance(E1, E1) == pytest.approx(0.0)
    assert similarity_service.cosine_similarity(E1, E1) == pytest.approx(1.0)
    assert similarity_service.cosine_distance(E1, E2) == pytest.approx(1.0)
    assert similarity_service.cosine_distance(E1, E_NEG) == pytest.approx(2.0)
    assert similarity_service.cosine_similarity(E1, E_NEG) == pytest.approx(-1.0)
    # Magnitude invariance: direction is what matters.
    big = _vec(3.0, 4.0)
    assert similarity_service.cosine_distance(big, big) == pytest.approx(0.0)
    assert similarity_service.cosine_distance(
        _vec(3.0, 4.0), _vec(6.0, 8.0)) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        similarity_service.cosine_distance([1.0], [1.0, 2.0])


def test_api_similarity_is_one_minus_distance_and_ranked(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query_face_id"] == world["qface"].id
    assert body["query_photo_type"] == "case"
    results = body["results"]
    # Identical vector first, orthogonal second; the same-type
    # decoy and the foreign-org candidate never appear.
    assert [r["similarity"] for r in results] == pytest.approx([1.0, 0.0])
    assert results[0]["face_id"] == world["sfaces"][0].id
    assert results[1]["face_id"] == world["sfaces"][1].id
    assert results[0]["photo_type"] == "sighting"
    assert results[0]["sighting_id"] == world["sighting"]
    assert results[0]["case_id"] == world["case_s"]
    assert results[0]["photo_id"] == world["photo_s"]["id"]


def test_no_raw_embeddings_exposed(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    assert "embedding" not in resp.text
    for result in resp.json()["results"]:
        assert set(result) == {
            "face_id", "photo_id", "photo_type",
            "sighting_id", "case_id", "similarity",
        }


# ---- Threshold + Top-K ----

def test_threshold_excludes_below_and_includes_equal(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": 1.0},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["similarity"] == pytest.approx(1.0)
    # Orthogonal query face: best candidate similarity is 0.0,
    # so threshold 1.0 yields a valid empty result (not an error).
    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"], world["sfaces"][1].id),
        json={"top_k": 10, "threshold": 1.0},
        headers=world["inv_h"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []


def test_top_k_keeps_highest(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 1, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    results = resp.json()["results"]
    assert len(results) == 1
    assert results[0]["face_id"] == world["sfaces"][0].id


def test_invalid_params_rejected(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    for payload in (
        {"top_k": 0, "threshold": 0.5},
        {"top_k": 101, "threshold": 0.5},
        {"top_k": 10, "threshold": 1.5},
        {"top_k": 10, "threshold": -1.5},
    ):
        resp = _search(client, world["inv_h"], world, payload)
        assert resp.status_code == 422, (payload, resp.text)


def test_request_defaults_apply(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (world["case_q"], world["photo_q"]["id"],
           world["qface"].id),
        headers=world["inv_h"],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["top_k"] == 10
    assert body["threshold"] == pytest.approx(0.70)
    # Default 0.70 keeps the identical candidate, drops orthogonal.
    assert len(body["results"]) == 1


# ---- Evidence direction ----

def test_sighting_query_searches_case_faces_only(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"], world["sfaces"][0].id),
        json={"top_k": 10, "threshold": -1.0},
        headers=world["inv_h"],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["query_photo_type"] == "sighting"
    assert body["results"], "expected case-face candidates"
    for result in body["results"]:
        assert result["photo_type"] == "case"
        assert result["sighting_id"] is None
    # Both case faces carry E1: query case face + same-type decoy.
    assert sorted(r["similarity"] for r in body["results"]) == pytest.approx(
        [1.0, 1.0]
    )
    assert len(body["results"]) == 2


def test_same_type_faces_never_match(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    face_ids = {r["face_id"] for r in resp.json()["results"]}
    assert world["qface"].id not in face_ids
    for result in resp.json()["results"]:
        assert result["photo_type"] == "sighting"


# ---- Current validity ----

def test_superseded_run_faces_excluded(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    _detect(
        client, world["rev_h"],
        "/cases/%d/sightings/%d/photos/%d/faces/redetect"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"]),
        fake_factory, _one_face(),
    )
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    # Old faces keep embeddings but leave the current run.
    assert resp.json()["results"] == []


def test_failed_run_faces_excluded(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    session = TestSession()
    try:
        run = session.get(
            FaceDetectionRun,
            world["sfaces"][0].run_id,
        )
        run.status = FaceDetectionStatus.FAILED
        session.commit()
    finally:
        session.close()
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []


def test_stale_source_sha_excluded(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    session = TestSession()
    try:
        photo = session.get(SightingPhoto, world["photo_s"]["id"])
        photo.derived_sha256 = "f" * 64
        session.commit()
    finally:
        session.close()
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []


# ---- Compatibility ----

def test_incompatible_representation_excluded(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    session = TestSession()
    try:
        face = session.get(FaceDetection, world["sfaces"][0].id)
        photo = session.get(SightingPhoto, world["photo_s"]["id"])
        session.expunge(face)
        session.expunge(photo)
        stale = FaceEmbedding(
            face_detection_id=face.id,
            source_derived_sha256=photo.derived_sha256,
            representation_name="sface",
            representation_version="2099future",
            model_name="sface",
            model_version="2021dec",
            model_sha256="0" * 64,
            dimension=128,
            embedding=E1,
        )
        session.add(stale)
        session.commit()
    finally:
        session.close()
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    # Future-version row ignored; the two current rows remain.
    assert len(resp.json()["results"]) == 2


# ---- Authorization + organization scope ----

def test_reporter_and_viewer_denied(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["rep_h"], world,
        {"top_k": 10, "threshold": 0.5},
    )
    assert resp.status_code == 403, resp.text
    viewer = make_user(db, role=UserRole.ORGANIZATION_MEMBER)
    make_membership(db, viewer, world["org_a"], OrgRole.VIEWER)
    resp = _search(
        client, auth_headers(viewer), world,
        {"top_k": 10, "threshold": 0.5},
    )
    assert resp.status_code == 403, resp.text


def test_reviewer_and_admin_allowed(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["rev_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    assert len(resp.json()["results"]) == 2
    # Admin scope crosses organizations: orgB candidate included.
    # (Counts are scoped per case because the session database
    # accumulates worlds across tests.)
    resp = _search(
        client, world["admin_h"], world,
        {"top_k": 100, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    by_case: dict[int, int] = {}
    for result in resp.json()["results"]:
        assert result["photo_type"] == "sighting"
        by_case[result["case_id"]] = by_case.get(result["case_id"], 0) + 1
    assert by_case.get(world["case_s"]) == 2
    assert by_case.get(world["case_b"]) == 1


def test_foreign_org_candidates_scoped_out(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["inv_h"], world,
        {"top_k": 10, "threshold": 1.0},
    )
    assert resp.status_code == 200, resp.text
    # orgB holds an identical-vector candidate: invisible to invA.
    assert len(resp.json()["results"]) == 1
    assert resp.json()["results"][0]["case_id"] == world["case_s"]
    # Admin sees across organizations: one identical candidate
    # per case in this world (other worlds accumulate globally,
    # so scope per case rather than counting absolutely).
    resp = _search(
        client, world["admin_h"], world,
        {"top_k": 100, "threshold": 1.0},
    )
    assert resp.status_code == 200, resp.text
    mine = [
        r for r in resp.json()["results"]
        if r["case_id"] in (world["case_s"], world["case_b"])
    ]
    assert len(mine) == 2
    assert {r["case_id"] for r in mine} == {
        world["case_s"], world["case_b"]
    }


def test_query_outside_scope_forbidden(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_b"], world["sighting_b"],
           world["photo_b"]["id"], world["bface"].id),
        json={"top_k": 10, "threshold": 0.5},
        headers=world["inv_h"],
    )
    assert resp.status_code == 403, resp.text


def test_retrieval_grants_no_edit_rights(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = _search(
        client, world["rev_h"], world,
        {"top_k": 10, "threshold": -1.0},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"], "reviewer must see candidates"
    # The reviewer retrieved a candidate from a case they do not
    # own; that visibility grants no modification rights.
    resp = client.patch(
        "/cases/%d" % world["case_s"],
        json={"title": "Hijacked"},
        headers=world["rev_h"],
    )
    assert resp.status_code == 403, resp.text


# ---- Query validation + multi-face ----

def test_query_without_embedding_conflicts(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    _detect(
        client, world["rev_h"],
        "/cases/%d/photos/%d/faces/redetect"
        % (world["case_q"], world["photo_q"]["id"]),
        fake_factory, _one_face(),
    )
    session = TestSession()
    try:
        run = (
            session.query(FaceDetectionRun)
            .filter(
                FaceDetectionRun.case_photo_id
                == world["photo_q"]["id"]
            )
            .order_by(FaceDetectionRun.id.desc())
            .first()
        )
        new_face = (
            session.query(FaceDetection)
            .filter(FaceDetection.run_id == run.id)
            .first()
        )
        new_face_id = new_face.id
    finally:
        session.close()
    resp = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (world["case_q"], world["photo_q"]["id"], new_face_id),
        json={"top_k": 10, "threshold": 0.5},
        headers=world["inv_h"],
    )
    assert resp.status_code == 409, resp.text


def test_face_photo_mismatch_is_404(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    resp = client.post(
        "/cases/%d/photos/%d/faces/%d/similar"
        % (world["case_q"], world["photo_q"]["id"],
           world["sfaces"][0].id),
        json={"top_k": 10, "threshold": 0.5},
        headers=world["inv_h"],
    )
    assert resp.status_code == 404, resp.text


def test_valid_query_zero_candidates_is_empty(
    client, db, s3mock, fake_factory
):
    world = _setup_world(client, db, s3mock, fake_factory)
    # Orthogonal query face: best candidate similarity is 0.0,
    # below threshold 0.5, so the valid query yields [].
    resp = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"], world["sfaces"][1].id),
        json={"top_k": 10, "threshold": 0.5},
        headers=world["inv_h"],
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["results"] == []


def test_faces_stay_independent_units(client, db, s3mock, fake_factory):
    world = _setup_world(client, db, s3mock, fake_factory)
    first = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"], world["sfaces"][0].id),
        json={"top_k": 10, "threshold": 1.0},
        headers=world["inv_h"],
    )
    second = client.post(
        "/cases/%d/sightings/%d/photos/%d/faces/%d/similar"
        % (world["case_s"], world["sighting"],
           world["photo_s"]["id"], world["sfaces"][1].id),
        json={"top_k": 10, "threshold": 0.5},
        headers=world["inv_h"],
    )
    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    # E1 face matches both case faces at 1.0; the orthogonal E2
    # face peaks at 0.0, below threshold 0.5, so it yields none.
    assert len(first.json()["results"]) == 2
    assert second.json()["results"] == []
