from datetime import datetime, timezone
from sqlalchemy import (
    BigInteger,
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base
from app.models.case_photo import PhotoStatus


class SightingPhoto(Base):
    """One uploaded image belonging to a sighting.

    Mirrors CasePhoto: the row references bytes in object storage, never
    pixels. `storage_key_original` is write-once; enhanced/derived
    renditions (Phase 4+) are separate objects under
    `storage_key_derived`. `case_id` is denormalized so sighting photos
    stay scoped to their case and the triple binding
    (photo.case_id == sighting.case_id == URL case_id) can be enforced.
    Access derives from the parent case's visibility rule.
    """

    __tablename__ = "sighting_photos"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    sighting_id: Mapped[int] = mapped_column(
        ForeignKey("sightings.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    uploaded_by: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    storage_key_original: Mapped[str] = mapped_column(
        String(1024),
        nullable=False,
        unique=True,
    )

    storage_key_derived: Mapped[str | None] = mapped_column(
        String(1024),
        nullable=True,
        default=None,
    )

    mime_type: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    byte_size: Mapped[int] = mapped_column(
        BigInteger,
        nullable=False,
    )

    width: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    height: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
    )

    processing_status: Mapped[PhotoStatus] = mapped_column(
        Enum(PhotoStatus),
        default=PhotoStatus.UPLOADED,
        nullable=False,
    )

    # ---- Phase 3: derived-processing metadata ----
    # Mirrors CasePhoto: all nullable so pre-Phase-3 rows stay valid.
    processing_error: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )

    processing_version: Mapped[str | None] = mapped_column(
        String(50),
        nullable=True,
        default=None,
    )

    derived_sha256: Mapped[str | None] = mapped_column(
        String(64),
        nullable=True,
        default=None,
    )

    processing_started_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    derived_width: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )

    derived_height: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )

    derived_mime_type: Mapped[str | None] = mapped_column(
        String(100),
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False,
    )
