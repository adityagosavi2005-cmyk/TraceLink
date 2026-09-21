from datetime import datetime, timezone
import enum
from sqlalchemy import DateTime, Enum, Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base


class SightingStatus(str, enum.Enum):
    """Triage state of a reported sighting.

    Phase 2 only ever writes REPORTED (creation default). Later phases
    triage sightings through the remaining values without a schema
    change.
    """

    REPORTED = "REPORTED"
    UNDER_REVIEW = "UNDER_REVIEW"
    VERIFIED = "VERIFIED"
    DISMISSED = "DISMISSED"


class Sighting(Base):
    """A reported observation linked to one missing-person case.

    Access derives from the parent case's visibility rule. Deletion of
    the parent case removes its sightings (application-ordered cascade;
    see the cases router), which in turn removes their photos.
    """

    __tablename__ = "sightings"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    case_id: Mapped[int] = mapped_column(
        ForeignKey("cases.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    reported_by: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    sighting_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )

    location_text: Mapped[str] = mapped_column(
        String(500),
        nullable=False,
    )

    latitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )

    longitude: Mapped[float | None] = mapped_column(
        Float,
        nullable=True,
        default=None,
    )

    description: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    contact_info: Mapped[str | None] = mapped_column(
        String(500),
        nullable=True,
        default=None,
    )

    status: Mapped[SightingStatus] = mapped_column(
        Enum(SightingStatus),
        default=SightingStatus.REPORTED,
        nullable=False,
        index=True,
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
