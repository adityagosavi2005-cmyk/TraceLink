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
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class EnhancementStatus(str, enum.Enum):
    """Outcome of one image-enhancement execution.

    Separate from PhotoStatus (Phase 3) and FaceDetectionStatus
    (Phase 4): those answer whether the derived image is ready and
    what detection did; this answers what enhancement did. History
    is preserved: retries and re-runs append new rows.
    """

    PROCESSING = "PROCESSING"
    COMPLETE = "COMPLETE"
    FAILED = "FAILED"


class ImageSourceType(str, enum.Enum):
    """Which image artifact a downstream row was produced from.

    DERIVED means the Phase 3 derived rendition; ENHANCED means the
    output of one COMPLETE EnhancementRun. Every Phase 7
    face-detection run and embedding records this so normal and
    enhanced representations can coexist without collapsing.
    """

    DERIVED = "DERIVED"
    ENHANCED = "ENHANCED"


class EnhancementRun(Base):
    """One image-enhancement execution and its resulting artifact.

    Exactly one parent photo per run (case_photo_id XOR
    sighting_photo_id), mirroring the FaceDetectionRun constraint
    style. Multiple runs per photo are allowed as history; runs are
    never deduplicated or overwritten.

    The enhanced artifact lives under the enhanced/ storage scope
    and is selected explicitly by id -- there is no implicit
    "latest enhancement". Original evidence and the Phase 3
    derived rendition are never modified by enhancement.
    """

    __tablename__ = "enhancement_runs"
    __table_args__ = (
        CheckConstraint(
            "(case_photo_id IS NOT NULL AND sighting_photo_id IS NULL)"
            " OR (case_photo_id IS NULL AND sighting_photo_id IS NOT NULL)",
            name="ck_enhancement_runs_single_parent",
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

    status: Mapped[EnhancementStatus] = mapped_column(
        Enum(EnhancementStatus),
        default=EnhancementStatus.PROCESSING,
        nullable=False,
        index=True,
    )

    # Phase 3 grandparent this run enhanced. Downstream validity
    # requires it to still equal the photo's current derived_sha256.
    source_derived_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    # Enhancer + model provenance. model_sha256 is the digest of the
    # exact weights file loaded (recorded at run time).
    enhancer_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    enhancer_version: Mapped[str] = mapped_column(
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

    # JSON-encoded inference parameters (scale, tiling, ...).
    parameters: Mapped[str | None] = mapped_column(
        String(1000), nullable=True, default=None
    )

    # Set only on COMPLETE: digest + key of the enhanced artifact.
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

    # Byte size of the stored enhanced artifact (COMPLETE only).
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
