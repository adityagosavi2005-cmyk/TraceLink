from datetime import datetime, timezone
import enum
from sqlalchemy import (
    CheckConstraint,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enhancement import ImageSourceType


class FaceDetectionStatus(str, enum.Enum):
    """Outcome of one face-detection execution.

    Separate from PhotoStatus (Phase 3): that answers whether the
    derived image is ready; this answers what face detection did.
    NOT_RUN is never persisted -- it describes the absence of any
    run row for a photo. QUEUED is intentionally absent: Phase 4
    detection is synchronous.
    """

    NOT_RUN = "NOT_RUN"
    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class FaceDetectionRun(Base):
    """One face-detection execution against one photo's derived image.

    Exactly one parent photo per run (case_photo_id XOR
    sighting_photo_id). History is preserved: re-detection appends a
    new row instead of mutating previous runs.
    """

    __tablename__ = "face_detection_runs"
    __table_args__ = (
        CheckConstraint(
            "(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)"
            " OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)",
            name="ck_face_detection_runs_single_parent",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    case_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("case_photos.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    sighting_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("sighting_photos.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    sighting_id: Mapped[int | None] = mapped_column(
        ForeignKey("sightings.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    status: Mapped[FaceDetectionStatus] = mapped_column(
        Enum(FaceDetectionStatus),
        default=FaceDetectionStatus.PROCESSING,
        nullable=False,
    )

    detector_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    detector_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    threshold: Mapped[float] = mapped_column(
        Float, nullable=False
    )

    source_derived_sha: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    source_derived_width: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    source_derived_height: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    # ---- Phase 7: source-aware provenance ----
    # Which artifact YuNet actually consumed. source_derived_sha
    # above is kept as the Phase 3 grandparent pointer: for DERIVED
    # runs source_sha256 equals it; for ENHANCED runs source_sha256
    # is the enhancement output SHA and enhancement_run_id points
    # at the selected EnhancementRun. Frame/face coordinates are in
    # source-image pixels (derived pixels for DERIVED runs,
    # enhanced pixels for ENHANCED runs).
    source_type: Mapped[ImageSourceType] = mapped_column(
        Enum(ImageSourceType),
        default=ImageSourceType.DERIVED,
        nullable=False,
    )
    source_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    enhancement_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("enhancement_runs.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )
    source_width: Mapped[int] = mapped_column(
        Integer, nullable=False
    )
    source_height: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    face_count: Mapped[int] = mapped_column(
        Integer, nullable=False, default=0
    )

    error: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default=None
    )

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    faces: Mapped[list["FaceDetection"]] = relationship(
        "FaceDetection",
        back_populates="run",
        cascade="all, delete-orphan",
        # No passive_deletes: the ORM must delete child faces itself.
        # DB-level ON DELETE CASCADE is a second layer (PostgreSQL),
        # but SQLite does not enforce foreign keys, so relying on it
        # alone orphans face rows when a run is deleted.
        order_by="FaceDetection.ordinal",
    )


class FaceDetection(Base):
    """One detected face within a run, in source-image pixels.

    ONE shared table for CasePhoto and SightingPhoto faces. The
    parent-photo columns mirror the owning run; exactly one is set.
    No identity, embedding, or demographic fields: detection only.
    """

    __tablename__ = "face_detections"
    __table_args__ = (
        CheckConstraint(
            "(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)"
            " OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)",
            name="ck_face_detections_single_parent",
        ),
        UniqueConstraint(
            "run_id", "ordinal", name="uq_face_detections_run_ordinal"
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    run_id: Mapped[int] = mapped_column(
        ForeignKey("face_detection_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    case_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("case_photos.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    sighting_photo_id: Mapped[int | None] = mapped_column(
        ForeignKey("sighting_photos.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)

    x_min: Mapped[int] = mapped_column(Integer, nullable=False)
    y_min: Mapped[int] = mapped_column(Integer, nullable=False)
    x_max: Mapped[int] = mapped_column(Integer, nullable=False)
    y_max: Mapped[int] = mapped_column(Integer, nullable=False)

    confidence: Mapped[float] = mapped_column(Float, nullable=False)

    frame_width: Mapped[int] = mapped_column(Integer, nullable=False)
    frame_height: Mapped[int] = mapped_column(Integer, nullable=False)

    # Optional YuNet landmarks (e.g. eyes/nose/mouth points) for
    # future alignment. NULL when the detector provides none.
    landmarks: Mapped[dict | None] = mapped_column(
        JSON, nullable=True, default=None
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    run: Mapped[FaceDetectionRun] = relationship(
        "FaceDetectionRun", back_populates="faces"
    )

    embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        back_populates="face_detection",
        cascade="all, delete-orphan",
        # Same rationale as FaceDetectionRun.faces: the ORM must
        # delete child embeddings itself. DB-level ON DELETE CASCADE
        # is a second layer (PostgreSQL), but SQLite does not enforce
        # foreign keys, so relying on it alone orphans embedding rows
        # when a face is deleted.
        order_by="FaceEmbedding.id",
    )
