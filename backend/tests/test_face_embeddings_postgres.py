"""Phase 5 PostgreSQL integration test: pgvector persistence.

SQLite cannot faithfully represent VECTOR(128), so this module is
the PostgreSQL-backed counterpart to the SQLite service suite.
It is fully environment-gated and never touches the developer's
local database by default:

- Set TRACELINK_TEST_DATABASE_URL to an ISOLATED PostgreSQL test
  database URL (postgresql+psycopg://...) to enable it.
- Without that variable (or without psycopg/pgvector installed),
  every test skips with an explicit reason.

The test database is only READ plus TEMPORARY tables (which vanish
with the session): no application table is created, migrated, or
modified here. Migration verification stays an explicit operator
step:  alembic upgrade head  (see the Phase 5 handoff notes).

When the Phase 5 migration HAS been applied to the test database,
the suite additionally asserts the real face_embeddings.embedding
column is vector(128) with the expected identity constraint.

Run from backend/ with e.g.:
  TRACELINK_TEST_DATABASE_URL=postgresql+psycopg://USER:PASS@localhost:5432/tracelink_test \\
      python -m pytest tests/test_face_embeddings_postgres.py -v
"""

import os

import pytest

psycopg = pytest.importorskip("psycopg")
pgvector = pytest.importorskip("pgvector")

TEST_DATABASE_URL = os.environ.get("TRACELINK_TEST_DATABASE_URL")

pytestmark = pytest.mark.skipif(
    not TEST_DATABASE_URL,
    reason=(
        "TRACELINK_TEST_DATABASE_URL is not set; "
        "PostgreSQL vector tests require an isolated test database"
    ),
)


@pytest.fixture()
def pg_connection():
    conn = psycopg.connect(TEST_DATABASE_URL, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def test_pgvector_extension_is_available(pg_connection):
    row = pg_connection.execute(
        "SELECT 1 FROM pg_extension WHERE extname = 'vector'"
    ).fetchone()
    assert row is not None, (
        "pgvector extension is not enabled in the test database; "
        "run CREATE EXTENSION vector there first"
    )


def test_vector_128_round_trip(pg_connection):
    from pgvector.psycopg import register_vector

    register_vector(pg_connection)
    pg_connection.execute("CREATE TEMP TABLE tmp_phase5_probe (v vector(128))")
    vector = [float(i) / 128.0 for i in range(128)]
    pg_connection.execute(
        "INSERT INTO tmp_phase5_probe (v) VALUES (%s)", (vector,)
    )
    stored = pg_connection.execute(
        "SELECT v FROM tmp_phase5_probe"
    ).fetchone()[0]
    assert len(list(stored)) == 128
    assert list(stored) == pytest.approx(vector)


def test_migrated_schema_column_is_vector_128(pg_connection):
    exists = pg_connection.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_name = 'face_embeddings'"
    ).fetchone()
    if exists is None:
        pytest.skip(
            "face_embeddings is absent in the test database; "
            "apply 'alembic upgrade head' there first"
        )
    coltype = pg_connection.execute(
        "SELECT format_type(atttypid, atttypmod) "
        "FROM pg_attribute "
        "WHERE attrelid = 'face_embeddings'::regclass "
        "AND attname = 'embedding'"
    ).fetchone()[0]
    assert coltype == "vector(128)", coltype
    # Phase 7 extends the representation identity with the consumed
    # source image, so the migrated schema carries
    # uq_face_embeddings_source_identity (normal and enhanced
    # embeddings of the same face coexist).
    constraint = pg_connection.execute(
        "SELECT 1 FROM pg_constraint "
        "WHERE conrelid = 'face_embeddings'::regclass "
        "AND conname = 'uq_face_embeddings_source_identity'"
    ).fetchone()
    assert constraint is not None, (
        "uq_face_embeddings_source_identity is missing on "
        "face_embeddings"
    )
