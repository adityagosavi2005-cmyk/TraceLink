"""Smoke tests: existing Case CRUD, validation, and authorization.

Covers owner access, unauthorized non-owner rejection, ADMIN access,
the new REVIEWER read rule, structured-field round-trips, and
pre-Phase-0 (legacy) record compatibility.
"""

from sqlalchemy import insert

from tests.conftest import auth_headers, make_user
from app.models.case import Case
from app.models.user import UserRole


def _create(client, headers, payload):
    return client.post("/cases", json=payload, headers=headers)


def test_case_crud_owner_flow(client, db):
    owner = make_user(db)
    headers = auth_headers(owner)

    created = _create(
        client, headers, {"title": "T", "description": "D"}
    )
    assert created.status_code == 201
    body = created.json()
    case_id = body["id"]
    # Phase 0 additions default to NULL for personal cases.
    assert body["organization_id"] is None
    assert body["full_name"] is None

    assert client.get("/cases", headers=headers).status_code == 200
    assert client.get("/cases/%d" % case_id, headers=headers).status_code == 200

    updated = client.patch(
        "/cases/%d" % case_id,
        json={"status": "UNDER_REVIEW"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["status"] == "UNDER_REVIEW"

    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code
        == 200
    )
    assert (
        client.get("/cases/%d" % case_id, headers=headers).status_code == 404
    )


def test_case_structured_fields_round_trip(client, db):
    owner = make_user(db)
    payload = {
        "title": "Missing Person",
        "description": "Narrative stays here",
        "full_name": "Jane Doe",
        "age_years": 29,
        "age_estimate_note": "approx",
        "last_seen_location": "Central Station",
        "clothing_description": "Red jacket",
        "distinguishing_marks": "Scar, left cheek",
        "contact_info": "officer@example.com",
    }
    resp = _create(client, auth_headers(owner), payload)
    assert resp.status_code == 201
    body = resp.json()
    for key, value in payload.items():
        if key in ("title", "description"):
            continue
        assert body[key] == value


def test_case_validation(client, db):
    owner = make_user(db)
    headers = auth_headers(owner)
    # Blank title rejected server-side.
    assert (
        _create(client, headers, {"title": "   ", "description": "D"}
                ).status_code
        == 422
    )
    # Missing description rejected.
    assert _create(client, headers, {"title": "T"}).status_code == 422
    # Age out of range rejected.
    assert (
        _create(
            client, headers, {"title": "T", "description": "D", "age_years": 400}
        ).status_code
        == 422
    )


def test_case_list_is_owner_scoped(client, db):
    owner = make_user(db)
    other = make_user(db)
    _create(client, auth_headers(owner), {"title": "Mine", "description": "D"})
    seen = client.get("/cases", headers=auth_headers(other)).json()
    assert seen == []


def test_case_non_owner_forbidden(client, db):
    owner = make_user(db)
    other = make_user(db)
    case_id = _create(
        client, auth_headers(owner), {"title": "T", "description": "D"}
    ).json()["id"]
    headers = auth_headers(other)
    assert client.get("/cases/%d" % case_id, headers=headers).status_code == 403
    assert (
        client.patch(
            "/cases/%d" % case_id, json={"title": "X"}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code == 403
    )


def test_case_admin_full_access(client, db):
    owner = make_user(db)
    admin = make_user(db, role=UserRole.ADMIN)
    case_id = _create(
        client, auth_headers(owner), {"title": "T", "description": "D"}
    ).json()["id"]
    headers = auth_headers(admin)
    assert client.get("/cases/%d" % case_id, headers=headers).status_code == 200
    assert (
        client.patch(
            "/cases/%d" % case_id, json={"title": "Edited"}, headers=headers
        ).status_code
        == 200
    )
    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code == 200
    )


def test_case_reviewer_read_only(client, db):
    owner = make_user(db)
    reviewer = make_user(db, role=UserRole.REVIEWER)
    headers = auth_headers(reviewer)
    case_id = _create(
        client, auth_headers(owner), {"title": "T", "description": "D"}
    ).json()["id"]
    # REVIEWER cannot create cases (unchanged from before Phase 0).
    assert (
        _create(client, headers, {"title": "R", "description": "D"}
                ).status_code
        == 403
    )
    # REVIEWER can list and read (new Phase 0 visibility rule).
    assert client.get("/cases", headers=headers).status_code == 200
    assert client.get("/cases/%d" % case_id, headers=headers).status_code == 200
    # REVIEWER cannot modify or delete.
    assert (
        client.patch(
            "/cases/%d" % case_id, json={"title": "X"}, headers=headers
        ).status_code
        == 403
    )
    assert (
        client.delete("/cases/%d" % case_id, headers=headers).status_code == 403
    )


def test_case_unknown_organization_rejected(client, db):
    owner = make_user(db)
    resp = _create(
        client,
        auth_headers(owner),
        {"title": "T", "description": "D", "organization_id": 999999},
    )
    assert resp.status_code == 404


def test_case_org_attach_requires_membership(client, db):
    from tests.conftest import make_org

    admin = make_user(db, role=UserRole.ADMIN)
    outsider = make_user(db)
    org = make_org(db, admin)
    resp = _create(
        client,
        auth_headers(outsider),
        {"title": "T", "description": "D", "organization_id": org.id},
    )
    assert resp.status_code == 403


def test_legacy_case_row_still_valid(client, db):
    """A pre-Phase-0 row (legacy columns only) must read back cleanly,
    with every Phase 0 field defaulting to NULL."""
    owner = make_user(db)
    db.execute(
        insert(Case).values(
            title="Legacy", description="Old", created_by=owner.id
        )
    )
    db.commit()
    resp = client.get("/cases", headers=auth_headers(owner))
    assert resp.status_code == 200
    legacy = [c for c in resp.json() if c["title"] == "Legacy"][0]
    assert legacy["organization_id"] is None
    assert legacy["full_name"] is None
    assert legacy["age_years"] is None
    assert legacy["status"] == "OPEN"
