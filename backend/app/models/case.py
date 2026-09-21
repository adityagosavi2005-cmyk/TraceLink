from datetime import datetime, timezone
import enum
from sqlalchemy import DateTime, Enum, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

class CaseStatus(str, enum.Enum):
    OPEN = "OPEN"
    UNDER_REVIEW = "UNDER_REVIEW"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"

class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True
    )

    title: Mapped[str] = mapped_column(
        String(200),
        nullable=False
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False
    )

    status: Mapped[CaseStatus] = mapped_column(
        Enum(CaseStatus),
        default=CaseStatus.OPEN,
        nullable=False
    )

    created_by: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True
    )

    # ---- Phase 0: organization scope ----
    # NULL means a PERSONAL case: visible to its owner (plus REVIEWER/ADMIN
    # under the visibility rule), shared with no organization. Existing rows
    # keep NULL and remain fully valid; no organization is ever invented
    # for them by migration.
    organization_id: Mapped[int | None] = mapped_column(
        ForeignKey("organizations.id"),
        nullable=True,
        index=True,
        default=None,
    )

    # ---- Phase 0: structured missing-person fields ----
    # `description` remains the free-text narrative. These optional columns
    # exist so posters, filters, fusion, and reports can query person data
    # without parsing prose. All nullable so every pre-Phase-0 row stays
    # valid unchanged.
    full_name: Mapped[str | None] = mapped_column(
        String(200),
        nullable=True,
        default=None,
    )

    age_years: Mapped[int | None] = mapped_column(
        Integer,
        nullable=True,
        default=None,
    )

    age_estimate_note: Mapped[str | None] = mapped_column(
        String(255),
        nullable=True,
        default=None,
    )

    last_seen_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )

    last_seen_location: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )

    clothing_description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )

    distinguishing_marks: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        default=None,
    )

    contact_info: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
        nullable=False
    )
