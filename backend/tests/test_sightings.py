"""Phase 2 tests: Sighting CRUD, validation, and authorization.

Mirrors the Case/Photo authorization doctrine: visibility derives from
the parent case; creation is open to reporter-class roles with view
access (REVIEWER excluded); edits/deletes are reporter-own-row or the
parent case's edit rule.

Run from backend/:  python -m pytest tests/test_sightings.py -v
"""

from datetime import datetime, timedelta, timezone

from tests.conftest import (
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from app.models.organization import OrgRole
from app.models.sighting import Sighting  # noqa: F401 (registers table)
from app.models.user import UserRole

assert Sighting is not None


def _now_iso():
    return datetime.now(timezone.utc).isoformat()


def _payload(**extra):
    payload = {
        "sighting_at": _now_iso(),
        "location_text": "Central Station, Platform 2",
        "description": "Person matching the description seen boarding.",
    }
    payload.update(extra)
    return payload


def _make_case(client, headers, **extra):
    payload = {"title": "T", "description": "D"}
    payload.update(extra)
    resp = client.post("/cases", json=payload, headers=headers)
    assert resp.status_code == 201
    return resp.json()["id"]


def _create(client, headers, case_id, **extra):
    return client.post(
        "/cases/%d/sightings" % case_id,
        json=_payload(**extra),
        headers=headers,
    )


def test_sighting_crud_owner_flow(client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)

    created = _create(client, headers, case_id)
    assert created.status_code == 201
    body = created.json()
    sighting_id = body["id"]
    assert body["case_id"] == case_id
    assert body["reported_by"] == owner.id
    assert body["status"] == "REPORTED"
    assert body["contact_info"] is None

    assert (
        client.get(
            "/cases/%d/sightings" % case_id, headers=headers
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/cases/%d/sightings/%d" % (case_id, sighting_id),
            headers=headers,
        ).status_code
        == 200
    )

    updated = client.patch(
        "/cases/%d/sightings/%d" % (case_id, sighting_id),
        json={"status": "UNDER_REVIEW"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "UNDER_REVIEW"

    assert (
        client.delete(
            "/cases/%d/sightings/%d" % (case_id, sighting_id),
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/cases/%d/sightings/%d" % (case_id, sighting_id),
            headers=headers,
        ).status_code
        == 404
    )


def test_sighting_unknown_case_404(client, db):
    owner = make_user(db)
    assert (
        _create(client, auth_headers(owner), 999999).status_code == 404
    )


def test_sighting_validation(client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    case_id = _make_case(client, headers)

    # Blank location rejected.
    assert (
        _create(client, headers, case_id, location_text="   ").status_code
        == 422
    )
    # Missing description rejected.
    payload = _payload()
    del payload["description"]
    assert (
        client.post(
            "/cases/%d/sightings" % case_id, json=payload, headers=headers
        ).status_code
        == 422
    )
    # Future sighting_at rejected.
    future = (datetime.now(timezone.utc) + timedelta(days=1)).isoformat()
    assert (
        _create(client, headers, case_id, sighting_at=future).status_code
        == 422
    )
    # Ancient sighting_at rejected.
    assert (
        _create(
            client, headers, case_id, sighting_at="1850-01-01T00:00:00Z"
        ).status_code
        == 422
    )
    # Latitude without longitude rejected (pair rule).
    assert (
        _create(client, headers, case_id, latitude=12.5).status_code
        == 422
    )
    # Out-of-range coordinates rejected.
    assert (
        _create(
            client,
            headers,
            case_id,
            latitude=95.0,
            longitude=10.0,
        ).status_code
        == 422
    )
    # Coordinate pair accepted.
    assert (
        _create(
            client,
            headers,
            case_id,
            latitude=12.5,
            longitude=77.2,
        ).status_code
        == 201
    )


def test_sighting_non_owner_forbidden(client, db):
    owner = make_user(db)
    other = make_user(db)
    case_id = _make_case(client, auth_headers(owner))
    sighting_id = _create(
        client, auth_headers(owner), case_id
    ).json()["id"]
    headers = auth_headers(other)
    base = "/cases/%d/sightings" % case_id
    assert client.post(base, json=_payload(), headers=headers).status_code == 403
    assert client.get(base, headers=headers).status_code == 403
    assert (
        client.get("%s/%d" % (base, sighting_id), headers=headers).status_code
        == 403
    )
    assert (
        client.patch(
            "%s/%d" % (base, sighting_id),
            json={"status": "VERIFIED"},
            headers=headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "%s/%d" % (base, sighting_id), headers=headers
        ).status_code
        == 403
    )


def test_sighting_admin_full_access(client, db):
    owner = make_user(db)
    admin = make_user(db, role=UserRole.ADMIN)
    case_id = _make_case(client, auth_headers(owner))
    sighting_id = _create(
        client, auth_headers(owner), case_id
    ).json()["id"]
    headers = auth_headers(admin)
    # Admin can create on another user's case.
    assert _create(client, headers, case_id).status_code == 201
    base = "/cases/%d/sightings" % case_id
    assert client.get(base, headers=headers).status_code == 200
    assert (
        client.get("%s/%d" % (base, sighting_id), headers=headers).status_code
        == 200
    )
    assert (
        client.patch(
            "%s/%d" % (base, sighting_id),
            json={"status": "VERIFIED"},
            headers=headers,
        ).status_code
        == 200
    )
    assert (
        client.delete(
            "%s/%d" % (base, sighting_id), headers=headers
        ).status_code
        == 200
    )


def test_sighting_reviewer_read_only(client, db):
    owner = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    headers = auth_headers(reviewer)
    case_id = _make_case(client, auth_headers(owner))
    sighting_id = _create(
        client, auth_headers(owner), case_id
    ).json()["id"]
    base = "/cases/%d/sightings" % case_id
    # REVIEWER cannot create.
    assert (
        client.post(base, json=_payload(), headers=headers).status_code
        == 403
    )
    # REVIEWER can list and read.
    assert client.get(base, headers=headers).status_code == 200
    assert (
        client.get("%s/%d" % (base, sighting_id), headers=headers).status_code
        == 200
    )
    # REVIEWER cannot modify or delete.
    assert (
        client.patch(
            "%s/%d" % (base, sighting_id),
            json={"status": "VERIFIED"},
            headers=headers,
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "%s/%d" % (base, sighting_id), headers=headers
        ).status_code
        == 403
    )


def test_sighting_org_visibility(client, db):
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
    base = "/cases/%d/sightings" % case_id

    # Org VIEWER may report/create but cannot edit or delete.
    created = _create(client, auth_headers(viewer), case_id)
    assert created.status_code == 201
    own_id = created.json()["id"]
    # VIEWER can edit their OWN row (reporter-own-row rule).
    assert (
        client.patch(
            "%s/%d" % (base, own_id),
            json={"description": "Updated by reporter"},
            headers=auth_headers(viewer),
        ).status_code
        == 200
    )
    # VIEWER cannot edit someone else's row.
    others_id = _create(
        client, auth_headers(investigator), case_id
    ).json()["id"]
    assert (
        client.patch(
            "%s/%d" % (base, others_id),
            json={"description": "Nope"},
            headers=auth_headers(viewer),
        ).status_code
        == 403
    )
    assert (
        client.delete(
            "%s/%d" % (base, others_id), headers=auth_headers(viewer)
        ).status_code
        == 403
    )
    # Cross-org outsider sees nothing.
    assert (
        client.get(base, headers=auth_headers(outsider)).status_code
        == 403
    )
    assert (
        client.post(
            base, json=_payload(), headers=auth_headers(outsider)
        ).status_code
        == 403
    )


def test_sighting_case_binding(client, db):
    """A sighting is only reachable through its own case."""
    owner = make_user(db)
    headers = auth_headers(owner)
    case_a = _make_case(client, headers)
    case_b = _make_case(client, headers)
    sighting_id = _create(client, headers, case_a).json()["id"]
    other_base = "/cases/%d/sightings/%d" % (case_b, sighting_id)
    assert client.get(other_base, headers=headers).status_code == 404
    assert (
        client.patch(
            other_base, json={"status": "VERIFIED"}, headers=headers
        ).status_code
        == 404
    )
    assert client.delete(other_base, headers=headers).status_code == 404


def test_sighting_index_scoping(client, db):
    # NOTE: the suite shares one file-backed test database per session
    # and API writes commit through their own sessions, so rows from
    # earlier tests remain visible to ADMIN/REVIEWER ("see all").
    # Baselines below keep this test order-independent without
    # weakening any visibility assertion.
    owner = make_user(db)
    other = make_user(db)
    admin = make_user(db, role=UserRole.ADMIN)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    admin_before = client.get(
        "/sightings", headers=auth_headers(admin)
    ).json()
    case_id = _make_case(client, auth_headers(owner))
    created = _create(client, auth_headers(owner), case_id).json()

    seen_owner = client.get("/sightings", headers=auth_headers(owner)).json()
    assert [s["id"] for s in seen_owner] == [created["id"]]
    assert client.get("/sightings", headers=auth_headers(other)).json() == []

    seen_admin = client.get("/sightings", headers=auth_headers(admin)).json()
    assert len(seen_admin) == len(admin_before) + 1
    assert created["id"] in {s["id"] for s in seen_admin}

    seen_reviewer = client.get(
        "/sightings", headers=auth_headers(reviewer)
    ).json()
    assert {s["id"] for s in seen_reviewer} == {
        s["id"] for s in seen_admin
    }
