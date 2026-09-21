"""phase 5: face embeddings (SFace representations)

Revision ID: 6ba28904ee47
Revises: a1b2c3d4e5f6
Create Date: 2026-09-20

Phase 5 face-representation foundation (additive only):
- ensures the pgvector `vector` extension exists (idempotent; the
  extension is already enabled on existing databases, this only
  makes fresh databases reproducible).
- `face_embeddings`: one row per (FaceDetection x representation /
  model version). Foreign key to face_detections with ON DELETE
  CASCADE so photo/sighting/case deletion removes derived
  representation data; uniqueness of the representation identity
  (face + representation/model name/version) is the final
  duplicate guard. model_sha256 is provenance only and is
  deliberately excluded from the identity.
- `embedding` is VECTOR(128): the raw SFace feature, stored
  unmodified. No similarity indexes here (Phase 6 concern).

Non-destructive: CREATE EXTENSION (IF NOT EXISTS) + CREATE TABLE
only, no changes to existing tables or enums.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector


# revision identifiers, used by Alembic.
revision: str = '6ba28904ee47'
down_revision: Union[str, Sequence[str], None] = 'a1b2c3d4e5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        'face_embeddings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('face_detection_id', sa.Integer(), nullable=False),
        sa.Column(
            'source_derived_sha256', sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            'representation_name', sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            'representation_version', sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            'model_name', sa.String(length=50), nullable=False
        ),
        sa.Column(
            'model_version', sa.String(length=50), nullable=False
        ),
        sa.Column(
            'model_sha256', sa.String(length=64), nullable=False
        ),
        sa.Column('dimension', sa.Integer(), nullable=False),
        sa.Column(
            'embedding', Vector(128), nullable=False
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.ForeignKeyConstraint(
            ['face_detection_id'], ['face_detections.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'face_detection_id',
            'representation_name',
            'representation_version',
            'model_name',
            'model_version',
            name='uq_face_embeddings_identity',
        ),
    )
    op.create_index(
        op.f('ix_face_embeddings_face_detection_id'),
        'face_embeddings', ['face_detection_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_embeddings_id'),
        'face_embeddings', ['id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_face_embeddings_id'), table_name='face_embeddings'
    )
    op.drop_index(
        op.f('ix_face_embeddings_face_detection_id'),
        table_name='face_embeddings',
    )
    op.drop_table('face_embeddings')
    # The vector extension is shared infrastructure that predates
    # Phase 5 (enabled manually on existing databases): never drop
    # it here.
