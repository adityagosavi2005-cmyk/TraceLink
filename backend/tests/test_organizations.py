"""Smoke tests: Organization/Membership foundation and org-scoped cases."""

from sqlalchemy.exc import IntegrityError

from tests.conftest import (
    auth_headers,
    make_membership,
    make_org,
    make_user,
)
from app.models.organization import Membership, OrgRole
from app.models.user import UserRole


def test_org_admin_only_creation(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    reporter = make_user(db)
    assert (
        client.post(
            "/organizations",
            json={"name": "X"},
            headers=auth_headers(reporter),
        ).status_code
        == 403
    )
    resp = client.post(
        "/organizations",
        json={"name": "North Unit", "description": "Police unit"},
        headers=auth_headers(admin),
    )
    assert resp.status_code == 201
    assert resp.json()["name"] == "North Unit"


def test_org_name_validation(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    resp = client.post(
        "/organizations",
        json={"name": "   "},
        headers=auth_headers(admin),
    )
    assert resp.status_code == 422


def test_org_visibility(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    member = make_user(db)
    outsider = make_user(db)
    org = make_org(db, admin)
    make_membership(db, member, org, OrgRole.INVESTIGATOR)

    # Member sees their org; outsider sees none; admin sees all.
    assert (
        len(
            client.get(
                "/organizations", headers=auth_headers(member)
            ).json()
        )
        == 1
    )
    assert (
        client.get("/organizations", headers=auth_headers(outsider)).json()
        == []
    )
    assert (
        len(
            client.get(
                "/organizations", headers=auth_headers(admin)
            ).json()
        )
        >= 1
    )
    # Detail: member 200, outsider 403, unknown 404.
    assert (
        client.get(
            "/organizations/%d" % org.id, headers=auth_headers(member)
        ).status_code
        == 200
    )
    assert (
        client.get(
            "/organizations/%d" % org.id, headers=auth_headers(outsider)
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/organizations/999999", headers=auth_headers(admin)
        ).status_code
        == 404
    )


def test_membership_management(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    user = make_user(db)
    org = make_org(db, admin)
    headers = auth_headers(admin)

    # Add.
    resp = client.post(
        "/organizations/%d/members" % org.id,
        json={"user_id": user.id, "role": "INVESTIGATOR"},
        headers=headers,
    )
    assert resp.status_code == 201
    # Duplicate add conflicts.
    assert (
        client.post(
            "/organizations/%d/members" % org.id,
            json={"user_id": user.id, "role": "INVESTIGATOR"},
            headers=headers,
        ).status_code
        == 409
    )
    # Unknown user.
    assert (
        client.post(
            "/organizations/%d/members" % org.id,
            json={"user_id": 999999, "role": "VIEWER"},
            headers=headers,
        ).status_code
        == 404
    )
    # Non-manager cannot add.
    assert (
        client.post(
            "/organizations/%d/members" % org.id,
            json={"user_id": admin.id, "role": "VIEWER"},
            headers=auth_headers(user),
        ).status_code
        == 403
    )
    # Role update.
    updated = client.patch(
        "/organizations/%d/members/%d" % (org.id, user.id),
        json={"role": "VIEWER"},
        headers=headers,
    )
    assert updated.status_code == 200
    assert updated.json()["role"] == "VIEWER"
    # Removal works and is visible in the member list.
    assert (
        client.delete(
            "/organizations/%d/members/%d" % (org.id, user.id),
            headers=headers,
        ).status_code
        == 200
    )
    members = client.get(
        "/organizations/%d/members" % org.id, headers=headers
    ).json()
    assert all(m["user_id"] != user.id for m in members)


def test_last_org_admin_protected(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    org = make_org(db, admin)
    make_membership(db, admin, org, OrgRole.ORG_ADMIN)
    headers = auth_headers(admin)
    # Cannot demote the last ORG_ADMIN.
    assert (
        client.patch(
            "/organizations/%d/members/%d" % (org.id, admin.id),
            json={"role": "VIEWER"},
            headers=headers,
        ).status_code
        == 409
    )
    # Cannot remove the last ORG_ADMIN.
    assert (
        client.delete(
            "/organizations/%d/members/%d" % (org.id, admin.id),
            headers=headers,
        ).status_code
        == 409
    )


def test_org_admin_manages_membership(client, db):
    """A non-global-admin ORG_ADMIN can manage their own organization."""
    admin = make_user(db, role=UserRole.ADMIN)
    org_admin = make_user(db)
    newcomer = make_user(db)
    org = make_org(db, admin)
    make_membership(db, org_admin, org, OrgRole.ORG_ADMIN)
    resp = client.post(
        "/organizations/%d/members" % org.id,
        json={"user_id": newcomer.id, "role": "VIEWER"},
        headers=auth_headers(org_admin),
    )
    assert resp.status_code == 201


def test_membership_unique_constraint(db):
    admin = make_user(db, role=UserRole.ADMIN)
    user = make_user(db)
    org = make_org(db, admin)
    make_membership(db, user, org, OrgRole.VIEWER)
    db.add(Membership(user_id=user.id, organization_id=org.id))
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
    else:
        raise AssertionError("duplicate membership was accepted")


def test_org_case_visibility_and_edit_roles(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    investigator = make_user(db)
    viewer = make_user(db)
    outsider = make_user(db)
    org = make_org(db, admin)
    make_membership(db, investigator, org, OrgRole.INVESTIGATOR)
    make_membership(db, viewer, org, OrgRole.VIEWER)

    # Investigator files an organization case.
    case_id = client.post(
        "/cases",
        json={"title": "Org case", "description": "D", "organization_id": org.id},
        headers=auth_headers(investigator),
    ).json()["id"]

    # Org viewer can read but not edit; outsider cannot read.
    assert (
        client.get(
            "/cases/%d" % case_id, headers=auth_headers(viewer)
        ).status_code
        == 200
    )
    assert (
        client.patch(
            "/cases/%d" % case_id,
            json={"title": "X"},
            headers=auth_headers(viewer),
        ).status_code
        == 403
    )
    assert (
        client.get(
            "/cases/%d" % case_id, headers=auth_headers(outsider)
        ).status_code
        == 403
    )
    # Investigator (org editor) can edit; listing includes org cases.
    assert (
        client.patch(
            "/cases/%d" % case_id,
            json={"contact_info": "desk@example.com"},
            headers=auth_headers(investigator),
        ).status_code
        == 200
    )
    listed = client.get("/cases", headers=auth_headers(viewer)).json()
    assert any(c["id"] == case_id for c in listed)
