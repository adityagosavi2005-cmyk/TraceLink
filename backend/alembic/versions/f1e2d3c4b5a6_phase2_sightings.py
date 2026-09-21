"""phase 2: sightings and sighting_photos tables

Revision ID: f1e2d3c4b5a6
Revises: e8f2a5c41d09
Create Date: 2026-09-16

Phase 2 sightings module:
- new `sightings` table referencing cases/users; a reported observation
  linked to one missing-person case.
- new `sighting_photos` table referencing sightings (owning FK) plus a
  denormalized `case_id` for scoping and triple-binding checks
  (photo.case_id == sighting.case_id == URL case_id).
- `sightings.status` starts at REPORTED; later phases triage it
  (no Phase 2 writer changes it past creation default).
- `sighting_photos.processing_status` starts at UPLOADED (same Phase 4
  reservation as case_photos).

Non-destructive: CREATE TABLE only. Existing users, cases,
organizations, memberships, and case_photos are untouched (in
particular the existing case_photos foreign key is NOT altered).
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f1e2d3c4b5a6'
down_revision: Union[str, Sequence[str], None] = 'e8f2a5c41d09'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The sightingstatus enum is created implicitly by create_table via
    # the status column (see Phase 0/1 enum lesson: no explicit
    # .create() alongside create_table).
    # The photostatus enum already exists (Phase 1 case_photos). It is
    # referenced via postgresql.ENUM with create_type=False: that flag
    # exists ONLY on the PG-specific type — generic sa.Enum silently
    # discards it, which re-emits CREATE TYPE and fails with
    # DuplicateObject.
    op.create_table(
        'sightings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('reported_by', sa.Integer(), nullable=False),
        sa.Column(
            'sighting_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column('location_text', sa.String(length=500), nullable=False),
        sa.Column('latitude', sa.Float(), nullable=True),
        sa.Column('longitude', sa.Float(), nullable=True),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('contact_info', sa.String(length=500), nullable=True),
        sa.Column(
            'status',
            sa.Enum(
                'REPORTED', 'UNDER_REVIEW', 'VERIFIED', 'DISMISSED',
                name='sightingstatus',
            ),
            nullable=False,
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ['case_id'], ['cases.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(['reported_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_sightings_case_id'),
        'sightings', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_sightings_id'), 'sightings', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_sightings_reported_by'),
        'sightings', ['reported_by'], unique=False,
    )
    op.create_index(
        op.f('ix_sightings_sighting_at'),
        'sightings', ['sighting_at'], unique=False,
    )
    op.create_index(
        op.f('ix_sightings_status'),
        'sightings', ['status'], unique=False,
    )
    op.create_index(
        'ix_sightings_case_sighting_at',
        'sightings', ['case_id', 'sighting_at'], unique=False,
    )

    op.create_table(
        'sighting_photos',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('sighting_id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('uploaded_by', sa.Integer(), nullable=False),
        sa.Column(
            'storage_key_original', sa.String(length=1024), nullable=False
        ),
        sa.Column(
            'storage_key_derived', sa.String(length=1024), nullable=True
        ),
        sa.Column('mime_type', sa.String(length=100), nullable=False),
        sa.Column('byte_size', sa.BigInteger(), nullable=False),
        sa.Column('width', sa.Integer(), nullable=False),
        sa.Column('height', sa.Integer(), nullable=False),
        sa.Column('sha256', sa.String(length=64), nullable=False),
        sa.Column(
            'processing_status',
            postgresql.ENUM(
                'UPLOADED', 'QUEUED', 'PROCESSING', 'READY', 'FAILED',
                name='photostatus', create_type=False,
            ),
            nullable=False,
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ['sighting_id'], ['sightings.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['case_id'], ['cases.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(['uploaded_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'storage_key_original', name='uq_sighting_photos_storage_key'
        ),
    )
    op.create_index(
        op.f('ix_sighting_photos_case_id'),
        'sighting_photos', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_sighting_photos_id'),
        'sighting_photos', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_sighting_photos_sha256'),
        'sighting_photos', ['sha256'], unique=False,
    )
    op.create_index(
        op.f('ix_sighting_photos_sighting_id'),
        'sighting_photos', ['sighting_id'], unique=False,
    )
    op.create_index(
        op.f('ix_sighting_photos_uploaded_by'),
        'sighting_photos', ['uploaded_by'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_sighting_photos_uploaded_by'),
        table_name='sighting_photos',
    )
    op.drop_index(
        op.f('ix_sighting_photos_sighting_id'),
        table_name='sighting_photos',
    )
    op.drop_index(
        op.f('ix_sighting_photos_sha256'), table_name='sighting_photos'
    )
    op.drop_index(
        op.f('ix_sighting_photos_id'), table_name='sighting_photos'
    )
    op.drop_index(
        op.f('ix_sighting_photos_case_id'), table_name='sighting_photos'
    )
    op.drop_table('sighting_photos')
    # NOTE: the shared photostatus enum is NOT dropped here; Phase 1
    # case_photos still uses it.

    op.drop_index(
        'ix_sightings_case_sighting_at', table_name='sightings'
    )
    op.drop_index(op.f('ix_sightings_status'), table_name='sightings')
    op.drop_index(
        op.f('ix_sightings_sighting_at'), table_name='sightings'
    )
    op.drop_index(
        op.f('ix_sightings_reported_by'), table_name='sightings'
    )
    op.drop_index(op.f('ix_sightings_id'), table_name='sightings')
    op.drop_index(op.f('ix_sightings_case_id'), table_name='sightings')
    op.drop_table('sightings')
    sa.Enum(name='sightingstatus').drop(op.get_bind(), checkfirst=True)
