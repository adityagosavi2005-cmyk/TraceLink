"""phase 1: case_photos table

Revision ID: e8f2a5c41d09
Revises: b7e2f41a9c06
Create Date: 2026-09-15

Phase 1 storage + case photos:
- new `case_photos` table referencing cases/users; rows hold storage
  keys and content metadata, never pixels.
- `processing_status` starts at UPLOADED for every row; later phases
  transition it (no Phase 1 writer changes it).

Non-destructive: CREATE TABLE only. Existing users, cases,
organizations, and memberships are untouched.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'e8f2a5c41d09'
down_revision: Union[str, Sequence[str], None] = 'b7e2f41a9c06'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # The photostatus enum is created implicitly by create_table via
    # the processing_status column (see Phase 0 enum lesson: no
    # explicit .create() alongside create_table).
    op.create_table(
        'case_photos',
        sa.Column('id', sa.Integer(), nullable=False),
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
            sa.Enum(
                'UPLOADED', 'QUEUED', 'PROCESSING', 'READY', 'FAILED',
                name='photostatus',
            ),
            nullable=False,
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'updated_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(['case_id'], ['cases.id']),
        sa.ForeignKeyConstraint(['uploaded_by'], ['users.id']),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'storage_key_original', name='uq_case_photos_storage_key'
        ),
    )
    op.create_index(
        op.f('ix_case_photos_case_id'),
        'case_photos', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_case_photos_id'), 'case_photos', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_case_photos_sha256'),
        'case_photos', ['sha256'], unique=False,
    )
    op.create_index(
        op.f('ix_case_photos_uploaded_by'),
        'case_photos', ['uploaded_by'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_case_photos_uploaded_by'), table_name='case_photos'
    )
    op.drop_index(op.f('ix_case_photos_sha256'), table_name='case_photos')
    op.drop_index(op.f('ix_case_photos_id'), table_name='case_photos')
    op.drop_index(
        op.f('ix_case_photos_case_id'), table_name='case_photos'
    )
    op.drop_table('case_photos')
    sa.Enum(name='photostatus').drop(op.get_bind(), checkfirst=True)
