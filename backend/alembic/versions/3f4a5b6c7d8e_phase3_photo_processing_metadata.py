"""phase 3: photo derived-processing metadata

Revision ID: 3f4a5b6c7d8e
Revises: f1e2d3c4b5a6
Create Date: 2026-09-17

Phase 3 synchronous deterministic preprocessing:
- adds nullable processing metadata columns to `case_photos` and
  `sighting_photos` (processing_error, processing_version,
  derived_sha256, processing_started_at, derived_width,
  derived_height, derived_mime_type).
- no new tables, no enum changes: the existing shared `photostatus`
  enum already carries UPLOADED/QUEUED/PROCESSING/READY/FAILED.

Non-destructive: ADD COLUMN (all NULL-able, no backfill) only.
Pre-Phase-3 rows stay valid: UPLOADED with NULL derived pointer, and
the retry endpoint can process them lazily.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '3f4a5b6c7d8e'
down_revision: Union[str, Sequence[str], None] = 'f1e2d3c4b5a6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _phase3_columns():
    # Fresh Column objects per table: a Column instance cannot be
    # attached to two tables.
    return (
        sa.Column('processing_error', sa.String(length=500), nullable=True),
        sa.Column('processing_version', sa.String(length=50), nullable=True),
        sa.Column('derived_sha256', sa.String(length=64), nullable=True),
        sa.Column(
            'processing_started_at', sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column('derived_width', sa.Integer(), nullable=True),
        sa.Column('derived_height', sa.Integer(), nullable=True),
        sa.Column('derived_mime_type', sa.String(length=100), nullable=True),
    )


def upgrade() -> None:
    for table in ('case_photos', 'sighting_photos'):
        for column in _phase3_columns():
            op.add_column(table, column)


def downgrade() -> None:
    for table in ('case_photos', 'sighting_photos'):
        for column in reversed(_phase3_columns()):
            op.drop_column(table, column.name)
