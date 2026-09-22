"""phase 7: enhancement runs + source-aware detection/embeddings

Revision ID: 7c1d9e02a4f7
Revises: 6ba28904ee47
Create Date: 2026-09-22

Phase 7 AI image enhancement/restoration foundation (additive only):
- `enhancement_runs`: one row per enhancement execution against one
  photo's Phase 3 derived image. Exactly one parent photo per run
  (case_photo_id XOR sighting_photo_id, enforced by CHECK, mirroring
  the FaceDetectionRun style). History preserved; runs are never
  deduplicated. Parent-photo foreign keys use ON DELETE CASCADE.
- `face_detection_runs`: additive source columns (source_type,
  source_sha256, enhancement_run_id, source_width, source_height).
  The existing source_derived_sha/width/height columns are kept as
  the Phase 3 grandparent pointer. Existing rows are backfilled to
  DERIVED with source_sha256 = source_derived_sha; no reprocessing.
- `face_embeddings`: additive source columns (source_type,
  source_sha256), backfilled to DERIVED from
  source_derived_sha256. The representation-identity uniqueness is
  extended with the source so normal and enhanced embeddings of the
  same face coexist (uq_face_embeddings_identity replaced by
  uq_face_embeddings_source_identity).

Non-destructive: CREATE TABLE + ADD COLUMN + backfill UPDATE only.
Existing Phase 4/5 rows stay valid; accepted data is never
rewritten.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '7c1d9e02a4f7'
down_revision: Union[str, Sequence[str], None] = '6ba28904ee47'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

ENHANCEMENT_STATUS = ('PROCESSING', 'COMPLETE', 'FAILED')
IMAGE_SOURCE_TYPE = ('DERIVED', 'ENHANCED')


def upgrade() -> None:
    enhancement_status = postgresql.ENUM(
        *ENHANCEMENT_STATUS, name='enhancementstatus'
    )
    image_source_type = postgresql.ENUM(
        *IMAGE_SOURCE_TYPE, name='imagesourcetype'
    )
    # imagesourcetype is introduced alongside ADD COLUMNs (no
    # create_table to create it implicitly), so it needs the
    # explicit create; checkfirst keeps re-runs idempotent, and on
    # non-PostgreSQL dialects this is a no-op. enhancementstatus is
    # created implicitly by the enhancement_runs create_table below
    # (single-create pattern, per the Phase 0/1 enum lesson: never
    # an explicit create alongside create_table).
    image_source_type.create(op.get_bind(), checkfirst=True)

    op.create_table(
        'enhancement_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('case_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_id', sa.Integer(), nullable=True),
        sa.Column('status', enhancement_status, nullable=False),
        sa.Column(
            'source_derived_sha256', sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            'enhancer_name', sa.String(length=50), nullable=False
        ),
        sa.Column(
            'enhancer_version', sa.String(length=50), nullable=False
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
        sa.Column(
            'parameters', sa.String(length=1000), nullable=True
        ),
        sa.Column(
            'output_sha256', sa.String(length=64), nullable=True
        ),
        sa.Column(
            'storage_key', sa.String(length=1024), nullable=True
        ),
        sa.Column(
            'mime_type', sa.String(length=100), nullable=False
        ),
        sa.Column('width', sa.Integer(), nullable=True),
        sa.Column('height', sa.Integer(), nullable=True),
        sa.Column('byte_size', sa.BigInteger(), nullable=True),
        sa.Column(
            'error_message', sa.String(length=500), nullable=True
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'started_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'finished_at', sa.DateTime(timezone=True), nullable=True
        ),
        sa.CheckConstraint(
            '(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)'
            ' OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)',
            name='ck_enhancement_runs_single_parent',
        ),
        sa.ForeignKeyConstraint(
            ['case_id'], ['cases.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['case_photo_id'], ['case_photos.id'], ondelete='CASCADE'
        ),
        sa.ForeignKeyConstraint(
            ['sighting_photo_id'],
            ['sighting_photos.id'],
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['sighting_id'], ['sightings.id'], ondelete='CASCADE'
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_enhancement_runs_case_id'),
        'enhancement_runs', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_enhancement_runs_case_photo_id'),
        'enhancement_runs', ['case_photo_id'], unique=False,
    )
    op.create_index(
        op.f('ix_enhancement_runs_id'),
        'enhancement_runs', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_enhancement_runs_sighting_id'),
        'enhancement_runs', ['sighting_id'], unique=False,
    )
    op.create_index(
        op.f('ix_enhancement_runs_sighting_photo_id'),
        'enhancement_runs', ['sighting_photo_id'], unique=False,
    )
    op.create_index(
        op.f('ix_enhancement_runs_status'),
        'enhancement_runs', ['status'], unique=False,
    )

    # ---- face_detection_runs: source-aware columns (nullable for
    # the backfill, then NOT NULL) ----
    op.add_column(
        'face_detection_runs',
        sa.Column('source_type', image_source_type, nullable=True),
    )
    op.add_column(
        'face_detection_runs',
        sa.Column(
            'source_sha256', sa.String(length=64), nullable=True
        ),
    )
    op.add_column(
        'face_detection_runs',
        sa.Column('enhancement_run_id', sa.Integer(), nullable=True),
    )
    op.add_column(
        'face_detection_runs',
        sa.Column('source_width', sa.Integer(), nullable=True),
    )
    op.add_column(
        'face_detection_runs',
        sa.Column('source_height', sa.Integer(), nullable=True),
    )
    op.execute(
        "UPDATE face_detection_runs SET source_type = 'DERIVED', "
        "source_sha256 = source_derived_sha, "
        "source_width = source_derived_width, "
        "source_height = source_derived_height"
    )
    op.alter_column(
        'face_detection_runs', 'source_type', nullable=False
    )
    op.alter_column(
        'face_detection_runs', 'source_sha256', nullable=False
    )
    op.alter_column(
        'face_detection_runs', 'source_width', nullable=False
    )
    op.alter_column(
        'face_detection_runs', 'source_height', nullable=False
    )
    op.create_foreign_key(
        'fk_face_detection_runs_enhancement_run_id',
        'face_detection_runs', 'enhancement_runs',
        ['enhancement_run_id'], ['id'], ondelete='CASCADE',
    )
    op.create_index(
        op.f('ix_face_detection_runs_enhancement_run_id'),
        'face_detection_runs', ['enhancement_run_id'], unique=False,
    )

    # ---- face_embeddings: source-aware columns + extended id ----
    op.add_column(
        'face_embeddings',
        sa.Column('source_type', image_source_type, nullable=True),
    )
    op.add_column(
        'face_embeddings',
        sa.Column(
            'source_sha256', sa.String(length=64), nullable=True
        ),
    )
    op.execute(
        "UPDATE face_embeddings SET source_type = 'DERIVED', "
        "source_sha256 = source_derived_sha256"
    )
    op.alter_column(
        'face_embeddings', 'source_type', nullable=False
    )
    op.alter_column(
        'face_embeddings', 'source_sha256', nullable=False
    )
    op.drop_constraint(
        'uq_face_embeddings_identity', 'face_embeddings',
        type_='unique',
    )
    op.create_unique_constraint(
        'uq_face_embeddings_source_identity', 'face_embeddings',
        [
            'face_detection_id', 'representation_name',
            'representation_version', 'model_name', 'model_version',
            'source_type', 'source_sha256',
        ],
    )


def downgrade() -> None:
    op.drop_constraint(
        'uq_face_embeddings_source_identity', 'face_embeddings',
        type_='unique',
    )
    op.create_unique_constraint(
        'uq_face_embeddings_identity', 'face_embeddings',
        [
            'face_detection_id', 'representation_name',
            'representation_version', 'model_name', 'model_version',
        ],
    )
    op.drop_column('face_embeddings', 'source_sha256')
    op.drop_column('face_embeddings', 'source_type')

    op.drop_index(
        op.f('ix_face_detection_runs_enhancement_run_id'),
        table_name='face_detection_runs',
    )
    op.drop_constraint(
        'fk_face_detection_runs_enhancement_run_id',
        'face_detection_runs', type_='foreignkey',
    )
    op.drop_column('face_detection_runs', 'source_height')
    op.drop_column('face_detection_runs', 'source_width')
    op.drop_column('face_detection_runs', 'enhancement_run_id')
    op.drop_column('face_detection_runs', 'source_sha256')
    op.drop_column('face_detection_runs', 'source_type')

    op.drop_index(
        op.f('ix_enhancement_runs_status'),
        table_name='enhancement_runs',
    )
    op.drop_index(
        op.f('ix_enhancement_runs_sighting_photo_id'),
        table_name='enhancement_runs',
    )
    op.drop_index(
        op.f('ix_enhancement_runs_sighting_id'),
        table_name='enhancement_runs',
    )
    op.drop_index(
        op.f('ix_enhancement_runs_id'), table_name='enhancement_runs'
    )
    op.drop_index(
        op.f('ix_enhancement_runs_case_photo_id'),
        table_name='enhancement_runs',
    )
    op.drop_index(
        op.f('ix_enhancement_runs_case_id'),
        table_name='enhancement_runs',
    )
    op.drop_table('enhancement_runs')
    postgresql.ENUM(
        *IMAGE_SOURCE_TYPE, name='imagesourcetype'
    ).drop(op.get_bind(), checkfirst=True)
    postgresql.ENUM(
        *ENHANCEMENT_STATUS, name='enhancementstatus'
    ).drop(op.get_bind(), checkfirst=True)
