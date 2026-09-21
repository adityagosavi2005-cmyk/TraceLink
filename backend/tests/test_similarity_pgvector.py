"""Phase 6 PostgreSQL integration test: pgvector cosine retrieval.

SQLite cannot execute ``<=>``, so this module verifies the real
operator behavior against an ISOLATED PostgreSQL test database:

- Set TRACELINK_TEST_DATABASE_URL to an isolated PostgreSQL test
  database URL (postgresql+psycopg://...) to enable it.
- Without that variable (or without psycopg/pgvector installed),
  every test skips with an explicit reason.

Only TEMPORARY tables (which vanish with the session) are used:
no application table is created, migrated, or modified here.

Run from backend/ with e.g.:
  TRACELINK_TEST_DATABASE_URL=postgresql+psycopg://USER:PASS@localhost:5432/tracelink_test \\
      python -m pytest tests/test_similarity_pgvector.py -v
"""

import os

import pytest

from app.services.similarity_service import cosine_distance

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


def _vec(*vals: float) -> list[float]:
    out = [0.0] * 128
    for i, v in enumerate(vals):
        out[i] = v
    return out


@pytest.fixture()
def pg_connection():
    conn = psycopg.connect(TEST_DATABASE_URL, autocommit=True)
    try:
        yield conn
    finally:
        conn.close()


def test_cosine_operator_orders_like_python_helper(pg_connection):
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    register_vector(pg_connection)
    pg_connection.execute(
        "CREATE TEMP TABLE tmp_phase6_vectors ("
        "id serial PRIMARY KEY, v vector(128))"
    )
    rows = {
        "identical": _vec(1.0),
        "orthogonal": _vec(0.0, 1.0),
        "opposite": _vec(-1.0),
    }
    for name, vector in rows.items():
        pg_connection.execute(
            "INSERT INTO tmp_phase6_vectors (v) VALUES (%s)",
            (Vector(vector),),
        )
    query = _vec(1.0)
    fetched = pg_connection.execute(
        "SELECT v, v <=> %s AS distance FROM tmp_phase6_vectors "
        "ORDER BY v <=> %s ASC",
        (Vector(query), Vector(query)),
    ).fetchall()
    assert len(fetched) == 3
    distances = [float(row[1]) for row in fetched]
    assert distances == sorted(distances)
    assert distances == pytest.approx([0.0, 1.0, 2.0])
    for stored, distance in fetched:
        assert distance == pytest.approx(
            cosine_distance(query, stored.to_list())
        )
        assert (1.0 - distance) == pytest.approx(
            1.0 - cosine_distance(query, stored.to_list())
        )


def test_threshold_predicate_matches_api_semantics(pg_connection):
    from pgvector import Vector
    from pgvector.psycopg import register_vector

    register_vector(pg_connection)
    pg_connection.execute(
        "CREATE TEMP TABLE tmp_phase6_threshold (v vector(128))"
    )
    pg_connection.execute(
        "INSERT INTO tmp_phase6_threshold (v) VALUES (%s)",
        (Vector(_vec(1.0)),),
    )
    pg_connection.execute(
        "INSERT INTO tmp_phase6_threshold (v) VALUES (%s)",
        (Vector(_vec(0.0, 1.0)),),
    )
    query = _vec(1.0)
    kept = pg_connection.execute(
        "SELECT 1 - (v <=> %s) AS similarity "
        "FROM tmp_phase6_threshold "
        "WHERE (1 - (v <=> %s)) >= %s "
        "ORDER BY v <=> %s ASC",
        (Vector(query), Vector(query), 1.0, Vector(query)),
    ).fetchall()
    assert len(kept) == 1
    assert float(kept[0][0]) == pytest.approx(1.0)
