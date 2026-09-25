"""phase 8: face restoration runs + restored embeddings

Revision ID: 8e5f2a1b3c4d
Revises: 7c1d9e02a4f7
Create Date: 2026-09-24

Phase 8 face-level GFPGAN restoration foundation (additive only):
- `face_restoration_runs`: one row per restoration execution for
  one explicitly selected FaceDetection. Exactly one parent photo
  per run (case_photo_id XOR sighting_photo_id, enforced by
  CHECK, mirroring the FaceDetectionRun/EnhancementRun style).
  The run row owns both the operation provenance (selected face,
  preparation transform, restorer/model identity, restored-frame
  geometry strategy) and the artifact metadata (output SHA,
  storage key, dimensions), mirroring EnhancementRun. History
  preserved; runs are never deduplicated. Parent-photo and face
  foreign keys use ON DELETE CASCADE.
- `face_embeddings.face_restoration_run_id`: nullable FK to the
  restoration run whose artifact SFace consumed (NULL for normal
  whole-photo embeddings). Existing DERIVED/ENHANCED source
  semantics and the existing source-identity uniqueness are
  untouched.
- Partial unique index `uq_face_embeddings_restored_identity`
  over (face + representation/model identity +
  face_restoration_run_id) WHERE the run id IS NOT NULL, so
  restored embeddings cannot duplicate per restoration run while
  normal (NULL) embeddings keep exactly the old uniqueness
  behavior. PostgreSQL treats NULLs as distinct in plain UNIQUE
  constraints, hence the partial index instead of extending the
  existing constraint. The same partial-index SQL is valid on
  SQLite (test metadata path uses equivalent SQLAlchemy Index
  constructs in the model).

Non-destructive: CREATE TABLE + ADD COLUMN + CREATE INDEX only.
Existing Phase 4/5/7 rows stay valid; accepted data is never
rewritten.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '8e5f2a1b3c4d'
down_revision: Union[str, Sequence[str], None] = '7c1d9e02a4f7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

RESTORATION_STATUS = ('PROCESSING', 'COMPLETE', 'FAILED')


def upgrade() -> None:
    # The facerestorationstatus enum is created implicitly by
    # create_table via the status column (single-create pattern;
    # no explicit .create() alongside create_table, per the
    # Phase 0/1 enum lesson).
    restoration_status = postgresql.ENUM(
        *RESTORATION_STATUS, name='facerestorationstatus'
    )

    op.create_table(
        'face_restoration_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('case_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_id', sa.Integer(), nullable=True),
        sa.Column('face_detection_id', sa.Integer(), nullable=False),
        sa.Column(
            'face_detection_run_id', sa.Integer(), nullable=False
        ),
        sa.Column('status', restoration_status, nullable=False),
        sa.Column(
            'source_derived_sha256', sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            'bbox_snapshot', sa.String(length=200), nullable=False
        ),
        sa.Column(
            'landmarks_snapshot', sa.String(length=2000),
            nullable=False,
        ),
        sa.Column(
            'prep_version', sa.String(length=50), nullable=False
        ),
        sa.Column(
            'prep_transform', sa.String(length=2000),
            nullable=False,
        ),
        sa.Column(
            'prepared_input_sha256', sa.String(length=64),
            nullable=False,
        ),
        sa.Column(
            'restorer_name', sa.String(length=50), nullable=False
        ),
        sa.Column(
            'restorer_version', sa.String(length=50),
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
        sa.Column(
            'aux_model_info', sa.String(length=1000),
            nullable=True,
        ),
        sa.Column(
            'parameters', sa.String(length=1000), nullable=True
        ),
        sa.Column(
            'restored_geometry_kind', sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            'restored_geometry_version', sa.String(length=50),
            nullable=False,
        ),
        sa.Column(
            'restored_geometry', sa.String(length=2000),
            nullable=False,
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
            name='ck_face_restoration_runs_single_parent',
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
        sa.ForeignKeyConstraint(
            ['face_detection_id'], ['face_detections.id'],
            ondelete='CASCADE',
        ),
        sa.ForeignKeyConstraint(
            ['face_detection_run_id'], ['face_detection_runs.id'],
            ondelete='CASCADE',
        ),
        sa.PrimaryKeyConstraint('id'),
    )
    op.create_index(
        op.f('ix_face_restoration_runs_case_id'),
        'face_restoration_runs', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_case_photo_id'),
        'face_restoration_runs', ['case_photo_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_face_detection_id'),
        'face_restoration_runs', ['face_detection_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_face_detection_run_id'),
        'face_restoration_runs', ['face_detection_run_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_id'),
        'face_restoration_runs', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_sighting_id'),
        'face_restoration_runs', ['sighting_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_sighting_photo_id'),
        'face_restoration_runs', ['sighting_photo_id'],
        unique=False,
    )
    op.create_index(
        op.f('ix_face_restoration_runs_status'),
        'face_restoration_runs', ['status'], unique=False,
    )

    # ---- face_embeddings: restored-run linkage (nullable) ----
    op.add_column(
        'face_embeddings',
        sa.Column(
            'face_restoration_run_id', sa.Integer(), nullable=True
        ),
    )
    op.create_foreign_key(
        'fk_face_embeddings_face_restoration_run_id',
        'face_embeddings', 'face_restoration_runs',
        ['face_restoration_run_id'], ['id'], ondelete='CASCADE',
    )
    op.create_index(
        op.f('ix_face_embeddings_face_restoration_run_id'),
        'face_embeddings', ['face_restoration_run_id'],
        unique=False,
    )
    # Partial unique index (valid on PostgreSQL and SQLite):
    # normal (NULL) rows keep exactly the old uniqueness; only
    # restored rows are constrained per restoration run.
    op.execute(
        "CREATE UNIQUE INDEX uq_face_embeddings_restored_identity "
        "ON face_embeddings (face_detection_id, "
        "representation_name, representation_version, model_name, "
        "model_version, face_restoration_run_id) "
        "WHERE face_restoration_run_id IS NOT NULL"
    )


def downgrade() -> None:
    op.execute(
        "DROP INDEX uq_face_embeddings_restored_identity"
    )
    op.drop_index(
        op.f('ix_face_embeddings_face_restoration_run_id'),
        table_name='face_embeddings',
    )
    op.drop_constraint(
        'fk_face_embeddings_face_restoration_run_id',
        'face_embeddings', type_='foreignkey',
    )
    op.drop_column('face_embeddings', 'face_restoration_run_id')

    op.drop_index(
        op.f('ix_face_restoration_runs_status'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_sighting_photo_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_sighting_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_face_detection_run_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_face_detection_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_case_photo_id'),
        table_name='face_restoration_runs',
    )
    op.drop_index(
        op.f('ix_face_restoration_runs_case_id'),
        table_name='face_restoration_runs',
    )
    op.drop_table('face_restoration_runs')
    postgresql.ENUM(
        *RESTORATION_STATUS, name='facerestorationstatus'
    ).drop(op.get_bind(), checkfirst=True)
