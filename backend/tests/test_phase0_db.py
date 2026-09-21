"""Phase 0 database checks: migration chain shape and NULL semantics."""

import ast
import os

from sqlalchemy import insert

from app.models.case import Case
from tests.conftest import make_user


VERSIONS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "alembic",
    "versions",
)


def _chain():
    revisions, downs = {}, {}
    for fname in os.listdir(VERSIONS_DIR):
        if not fname.endswith(".py"):
            continue
        tree = ast.parse(
            open(os.path.join(VERSIONS_DIR, fname)).read()
        )
        rev = down = None
        for node in ast.walk(tree):
            if isinstance(node, ast.AnnAssign) and isinstance(
                node.target, ast.Name
            ):
                if node.target.id == "revision":
                    rev = ast.literal_eval(node.value)
                elif node.target.id == "down_revision":
                    down = ast.literal_eval(node.value)
        revisions[rev] = fname
        downs[rev] = down
    return revisions, downs


def test_migration_chain_has_single_head():
    revisions, downs = _chain()
    assert None not in revisions
    referenced = {d for d in downs.values()}
    heads = [r for r in revisions if r not in referenced]
    # Exactly one head, but never hardcoded: each new linear migration
    # (Phase 1 and beyond) becomes the head without touching this test.
    assert len(heads) == 1, "expected one head, found: %s" % (heads,)
    # Linear history back to the base migration: walking down from the
    # head must visit every revision exactly once (no branches/merges).
    seen, current = [], heads[0]
    while current is not None:
        assert current not in seen
        seen.append(current)
        current = downs[current]
    assert sorted(seen) == sorted(revisions)


def test_organization_id_nullable_and_null_by_default(db):
    owner = make_user(db)
    db.execute(
        insert(Case).values(
            title="Null org", description="D", created_by=owner.id
        )
    )
    db.commit()
    row = db.query(Case).filter(Case.title == "Null org").one()
    assert row.organization_id is None
    assert row.full_name is None
    assert row.age_years is None
    assert row.last_seen_at is None
