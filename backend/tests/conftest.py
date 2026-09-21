"""Phase 0 smoke-test foundation.

Runs against a disposable SQLite database (file-based, removed after the
session) so the suite never touches the development PostgreSQL instance.
SQLite is sufficient here because Phase 0 uses only portable column types;
the Alembic migration itself is exercised against PostgreSQL by the
deployment step (`alembic upgrade head`).

Run from backend/:  python -m pytest tests/ -v
"""

import os
import sys

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Must be set BEFORE app modules are imported: Settings() reads them
# at import time (environment variables take precedence over .env).
os.environ["DATABASE_URL"] = "sqlite:///./test_phase0.db"
os.environ["JWT_SECRET_KEY"] = "phase0-test-secret-not-for-production"
os.environ["JWT_ALGORITHM"] = "HS256"
os.environ["ACCESS_TOKEN_EXPIRE_MINUTES"] = "30"

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.core.security import create_access_token, hash_password
from app.main import app
from app.models.case import Case
from app.models.face_detection import FaceDetection, FaceDetectionRun
from app.models.face_embedding import FaceEmbedding
from app.models.organization import Membership, Organization
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User, UserRole

# Import models so all tables are registered on Base.metadata.
assert Case is not None and Membership is not None
assert Organization is not None and User is not None
assert Sighting is not None and SightingPhoto is not None
assert FaceDetection is not None and FaceDetectionRun is not None
assert FaceEmbedding is not None

TEST_DB_PATH = os.path.join(BACKEND_DIR, "test_phase0.db")

test_engine = create_engine(
    "sqlite:///./test_phase0.db",
    connect_args={"check_same_thread": False},
)
TestSession = sessionmaker(bind=test_engine, autoflush=False, autocommit=False)


def _override_get_db():
    db = TestSession()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


@pytest.fixture(scope="session", autouse=True)
def _database():
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)
    Base.metadata.create_all(test_engine)
    yield
    test_engine.dispose()
    if os.path.exists(TEST_DB_PATH):
        os.remove(TEST_DB_PATH)


@pytest.fixture()
def db():
    session = TestSession()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture()
def client():
    return TestClient(app)


_user_counter = {"n": 0}


def make_user(db_session, role=UserRole.REPORTER):
    """Create a user row directly with the requested global role."""
    _user_counter["n"] += 1
    n = _user_counter["n"]
    user = User(
        name="User %d" % n,
        email="user%d@example.com" % n,
        password_hash=hash_password("Password123!"),
        role=role,
    )
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def token_for(user):
    return create_access_token(user_id=user.id, role=user.role.value)


def auth_headers(user):
    return {"Authorization": "Bearer %s" % token_for(user)}


def make_org(db_session, creator, name="Org"):
    org = Organization(
        name="%s %d" % (name, creator.id),
        description="Phase 0 test organization",
        created_by=creator.id,
    )
    db_session.add(org)
    db_session.commit()
    db_session.refresh(org)
    return org


def make_membership(db_session, user, org, role):
    membership = Membership(
        user_id=user.id, organization_id=org.id, role=role
    )
    db_session.add(membership)
    db_session.commit()
    db_session.refresh(membership)
    return membership
