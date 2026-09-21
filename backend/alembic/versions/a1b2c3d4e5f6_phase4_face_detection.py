"""phase 4: face detection runs + detections

Revision ID: a1b2c3d4e5f6
Revises: 3f4a5b6c7d8e
Create Date: 2026-09-18

Phase 4 face-detection foundation (additive only):
- `face_detection_runs`: one row per detection execution against one
  photo's Phase 3 derived image. Exactly one parent photo per run
  (case_photo_id XOR sighting_photo_id, enforced by CHECK).
- `face_detections`: ONE shared table for faces from both CasePhoto
  and SightingPhoto, keyed to the owning run. Same single-parent
  CHECK, plus uniqueness of (run_id, ordinal).

Non-destructive: CREATE TABLE only, no changes to existing tables
or enums. Parent-photo foreign keys use ON DELETE CASCADE so
photo/sighting/case deletion removes derived face data; routers
also delete runs explicitly in application order.
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'a1b2c3d4e5f6'
down_revision: Union[str, Sequence[str], None] = '3f4a5b6c7d8e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FACE_STATUS = ('NOT_RUN', 'PROCESSING', 'COMPLETE', 'FAILED')


def upgrade() -> None:
    # The facedetectionstatus enum is created implicitly by
    # create_table via the status column (single-create pattern;
    # no explicit .create() alongside create_table, per the
    # Phase 0/1 enum lesson documented in f1e2d3c4b5a6).
    face_status = postgresql.ENUM(
        *FACE_STATUS, name='facedetectionstatus'
    )

    op.create_table(
        'face_detection_runs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('case_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_id', sa.Integer(), nullable=True),
        sa.Column('status', face_status, nullable=False),
        sa.Column('detector_name', sa.String(length=50), nullable=False),
        sa.Column(
            'detector_version', sa.String(length=50), nullable=False
        ),
        sa.Column('threshold', sa.Float(), nullable=False),
        sa.Column(
            'source_derived_sha', sa.String(length=64), nullable=False
        ),
        sa.Column('source_derived_width', sa.Integer(), nullable=False),
        sa.Column('source_derived_height', sa.Integer(), nullable=False),
        sa.Column('face_count', sa.Integer(), nullable=False),
        sa.Column('error', sa.String(length=500), nullable=True),
        sa.Column(
            'started_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.Column(
            'finished_at', sa.DateTime(timezone=True), nullable=True
        ),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            '(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)'
            ' OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)',
            name='ck_face_detection_runs_single_parent',
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
        op.f('ix_face_detection_runs_case_id'),
        'face_detection_runs', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detection_runs_case_photo_id'),
        'face_detection_runs', ['case_photo_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detection_runs_id'),
        'face_detection_runs', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detection_runs_sighting_id'),
        'face_detection_runs', ['sighting_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detection_runs_sighting_photo_id'),
        'face_detection_runs', ['sighting_photo_id'], unique=False,
    )

    op.create_table(
        'face_detections',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('run_id', sa.Integer(), nullable=False),
        sa.Column('case_id', sa.Integer(), nullable=False),
        sa.Column('case_photo_id', sa.Integer(), nullable=True),
        sa.Column('sighting_photo_id', sa.Integer(), nullable=True),
        sa.Column('ordinal', sa.Integer(), nullable=False),
        sa.Column('x_min', sa.Integer(), nullable=False),
        sa.Column('y_min', sa.Integer(), nullable=False),
        sa.Column('x_max', sa.Integer(), nullable=False),
        sa.Column('y_max', sa.Integer(), nullable=False),
        sa.Column('confidence', sa.Float(), nullable=False),
        sa.Column('frame_width', sa.Integer(), nullable=False),
        sa.Column('frame_height', sa.Integer(), nullable=False),
        sa.Column('landmarks', sa.JSON(), nullable=True),
        sa.Column(
            'created_at', sa.DateTime(timezone=True), nullable=False
        ),
        sa.CheckConstraint(
            '(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)'
            ' OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)',
            name='ck_face_detections_single_parent',
        ),
        sa.ForeignKeyConstraint(
            ['run_id'], ['face_detection_runs.id'], ondelete='CASCADE'
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
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint(
            'run_id', 'ordinal', name='uq_face_detections_run_ordinal'
        ),
    )
    op.create_index(
        op.f('ix_face_detections_case_id'),
        'face_detections', ['case_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detections_case_photo_id'),
        'face_detections', ['case_photo_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detections_confidence'),
        'face_detections', ['confidence'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detections_id'),
        'face_detections', ['id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detections_run_id'),
        'face_detections', ['run_id'], unique=False,
    )
    op.create_index(
        op.f('ix_face_detections_sighting_photo_id'),
        'face_detections', ['sighting_photo_id'], unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f('ix_face_detections_sighting_photo_id'),
        table_name='face_detections',
    )
    op.drop_index(
        op.f('ix_face_detections_run_id'), table_name='face_detections'
    )
    op.drop_index(
        op.f('ix_face_detections_id'), table_name='face_detections'
    )
    op.drop_index(
        op.f('ix_face_detections_confidence'),
        table_name='face_detections',
    )
    op.drop_index(
        op.f('ix_face_detections_case_photo_id'),
        table_name='face_detections',
    )
    op.drop_index(
        op.f('ix_face_detections_case_id'), table_name='face_detections'
    )
    op.drop_table('face_detections')
    op.drop_index(
        op.f('ix_face_detection_runs_sighting_photo_id'),
        table_name='face_detection_runs',
    )
    op.drop_index(
        op.f('ix_face_detection_runs_sighting_id'),
        table_name='face_detection_runs',
    )
    op.drop_index(
        op.f('ix_face_detection_runs_id'),
        table_name='face_detection_runs',
    )
    op.drop_index(
        op.f('ix_face_detection_runs_case_photo_id'),
        table_name='face_detection_runs',
    )
    op.drop_index(
        op.f('ix_face_detection_runs_case_id'),
        table_name='face_detection_runs',
    )
    op.drop_table('face_detection_runs')
    postgresql.ENUM(
        *FACE_STATUS, name='facedetectionstatus'
    ).drop(op.get_bind(), checkfirst=True)
