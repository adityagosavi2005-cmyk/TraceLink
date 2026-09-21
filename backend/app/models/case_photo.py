from datetime import datetime, timezone
import enum
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


class PhotoStatus(str, enum.Enum):
    """Lifecycle of a photo's server-side handling.

    Phase 1 only ever writes UPLOADED. The remaining values are defined
    now so Phase 4 (processing jobs) can transition them without a
    schema change.
    """

    UPLOADED = "UPLOADED"
    QUEUED = "QUEUED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"


class CasePhoto(Base):
    """One uploaded evidence image belonging to a case.

    The row references bytes in object storage; it never holds pixels.
    `storage_key_original` is write-once: enhanced/derived renditions
    (Phase 4+) are separate objects under `storage_key_derived`.
    Access derives from the parent case's visibility rule.
    """

    __tablename__ = "case_photos"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id"),
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
    # All nullable so every pre-Phase-3 row stays valid unchanged
    # (UPLOADED with NULL derived pointer). Only the retry/upload
    # pipeline writes these; the original evidence columns above are
    # never modified by processing.
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
