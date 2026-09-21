"""Smoke tests: registration, login, authenticated access, admin paths."""

from tests.conftest import auth_headers, make_user
from app.models.user import UserRole


def test_register_ok(client):
    resp = client.post(
        "/auth/register",
        json={
            "name": "Reporter",
            "email": "reporter@example.com",
            "password": "Password123!",
        },
    )
    assert resp.status_code == 201
    assert "user_id" in resp.json()


def test_register_duplicate_conflict(client):
    payload = {
        "name": "Dup",
        "email": "dup@example.com",
        "password": "Password123!",
    }
    assert client.post("/auth/register", json=payload).status_code == 201
    resp = client.post("/auth/register", json=payload)
    assert resp.status_code == 409


def test_register_weak_password_rejected(client):
    resp = client.post(
        "/auth/register",
        json={"name": "Weak", "email": "weak@example.com", "password": "short"},
    )
    assert resp.status_code == 422


def test_login_ok_and_me(client, db):
    user = make_user(db)
    resp = client.post(
        "/auth/login",
        json={"email": user.email, "password": "Password123!"},
    )
    assert resp.status_code == 200
    token = resp.json()["access_token"]
    assert token

    me = client.get(
        "/auth/me", headers={"Authorization": "Bearer %s" % token}
    )
    assert me.status_code == 200
    assert me.json()["email"] == user.email
    assert me.json()["role"] == "REPORTER"


def test_login_wrong_password_unauthorized(client, db):
    user = make_user(db)
    resp = client.post(
        "/auth/login",
        json={"email": user.email, "password": "WrongPassword!"},
    )
    assert resp.status_code == 401


def test_login_unknown_email_unauthorized(client):
    resp = client.post(
        "/auth/login",
        json={"email": "nobody@example.com", "password": "Password123!"},
    )
    assert resp.status_code == 401


def test_me_without_token_rejected(client):
    assert client.get("/auth/me").status_code in (401, 403)


def test_admin_test_endpoint(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    reporter = make_user(db, role=UserRole.REPORTER)
    assert (
        client.get("/auth/admin-test", headers=auth_headers(admin)).status_code
        == 200
    )
    assert (
        client.get(
            "/auth/admin-test", headers=auth_headers(reporter)
        ).status_code
        == 403
    )


def test_create_admin_paths(client, db):
    admin = make_user(db, role=UserRole.ADMIN)
    reporter = make_user(db, role=UserRole.REPORTER)
    payload = {
        "name": "New Admin",
        "email": "newadmin@example.com",
        "password": "Password123!",
    }
    # Non-admin forbidden.
    assert (
        client.post(
            "/auth/admin", json=payload, headers=auth_headers(reporter)
        ).status_code
        == 403
    )
    # Admin creates.
    resp = client.post(
        "/auth/admin", json=payload, headers=auth_headers(admin)
    )
    assert resp.status_code == 201
    # Duplicate email conflicts (consistent with /auth/register).
    assert (
        client.post(
            "/auth/admin", json=payload, headers=auth_headers(admin)
        ).status_code
        == 409
    )
