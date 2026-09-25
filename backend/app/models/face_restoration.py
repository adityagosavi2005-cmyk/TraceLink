from datetime import datetime, timezone
import enum
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


class FaceRestorationStatus(str, enum.Enum):
    """Outcome of one face-restoration execution.

    Separate from PhotoStatus (Phase 3), FaceDetectionStatus
    (Phase 4), and EnhancementStatus (Phase 7): this answers what
    face restoration did. History is preserved: retries and
    re-runs append new rows.
    """

    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class RestoredGeometryKind(str, enum.Enum):
    """Which strategy produced the restored-frame SFace geometry.

    MAPPED means source YuNet landmarks transformed through the
    recorded preparation transform (valid only when neither
    preparation nor the restorer warped the frame). CANONICAL
    means versioned synthetic restored-frame landmarks (used when
    the restorer realigned the face internally). The two are never
    silently mixed: every run records exactly one.
    """

    MAPPED = "MAPPED"
    CANONICAL = "CANONICAL"


class FaceRestorationRun(Base):
    """One face-restoration execution and its resulting artifact.

    Exactly one parent photo per run (case_photo_id XOR
    sighting_photo_id), mirroring the FaceDetectionRun and
    EnhancementRun constraint style. Exactly one selected face
    per run (face_detection_id + face_detection_run_id): several
    faces of one photo restore independently, each with its own
    history; runs are never deduplicated or overwritten.

    The restored artifact lives under the restored-faces/ storage
    scope and is selected explicitly by id -- there is no implicit
    "latest restoration". Original evidence, the Phase 3 derived
    rendition, and the selected FaceDetection are never modified
    by restoration. The prepared GFPGAN input is ephemeral: only
    its SHA and transform record are persisted here.
    """

    __tablename__ = "face_restoration_runs"
    __table_args__ = (
        CheckConstraint(
            "(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)"
            " OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)",
            name="ck_face_restoration_runs_single_parent",
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

    # Selected face: backend identity is always the face row, with
    # the owning run pinned so the coordinate frame is exact.
    face_detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    face_detection_run_id: Mapped[int] = mapped_column(
        ForeignKey("face_detection_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    status: Mapped[FaceRestorationStatus] = mapped_column(
        Enum(FaceRestorationStatus),
        default=FaceRestorationStatus.PROCESSING,
        nullable=False,
        index=True,
    )

    # Phase 3 grandparent this restoration consumed. Downstream
    # validity requires it to still equal the photo's current
    # derived_sha256.
    source_derived_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    # Snapshots of the selected face geometry at trigger time, so
    # later face/detection changes cannot rewrite history.
    bbox_snapshot: Mapped[str] = mapped_column(
        String(200), nullable=False
    )
    landmarks_snapshot: Mapped[str] = mapped_column(
        String(2000), nullable=False
    )

    # Preparation provenance: version + actual transform record +
    # SHA of the ephemeral prepared input bytes.
    prep_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    prep_transform: Mapped[str] = mapped_column(
        String(2000), nullable=False
    )
    prepared_input_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    # Restorer + model provenance. model_sha256 is the digest of
    # the exact weights file loaded (recorded at run time).
    # aux_model_info records auxiliary model identity when the
    # restorer reports any (otherwise NULL).
    restorer_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    restorer_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    model_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    model_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )
    aux_model_info: Mapped[str | None] = mapped_column(
        String(1000), nullable=True, default=None
    )

    # JSON-encoded inference parameters.
    parameters: Mapped[str | None] = mapped_column(
        String(1000), nullable=True, default=None
    )

    # Restored-frame geometry strategy. MAPPED preserves the
    # selected YuNet observation through the recorded transform;
    # CANONICAL marks explicitly synthetic landmarks (with
    # restored_geometry_version identifying the set).
    restored_geometry_kind: Mapped[str] = mapped_column(
        String(20), nullable=False
    )
    restored_geometry_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    restored_geometry: Mapped[str] = mapped_column(
        String(2000), nullable=False
    )

    # Set only on COMPLETE: digest + key of the restored artifact.
    output_sha256: Mapped[str | None] = mapped_column(
        String(64), nullable=True, default=None
    )
    storage_key: Mapped[str | None] = mapped_column(
        String(1024), nullable=True, default=None
    )

    mime_type: Mapped[str] = mapped_column(
        String(100), nullable=False
    )
    width: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )
    height: Mapped[int | None] = mapped_column(
        Integer, nullable=True, default=None
    )

    # Byte size of the stored restored artifact (COMPLETE only).
    byte_size: Mapped[int | None] = mapped_column(
        BigInteger, nullable=True, default=None
    )

    error_message: Mapped[str | None] = mapped_column(
        String(500), nullable=True, default=None
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
    finished_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True, default=None
    )

    embeddings: Mapped[list["FaceEmbedding"]] = relationship(
        "FaceEmbedding",
        back_populates="face_restoration_run",
        cascade="all, delete-orphan",
        # Same rationale as FaceDetectionRun.faces: the ORM must
        # delete child embeddings itself because SQLite does not
        # enforce foreign keys.
        order_by="FaceEmbedding.id",
    )
