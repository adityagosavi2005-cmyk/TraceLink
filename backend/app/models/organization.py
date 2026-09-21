from datetime import datetime, timezone

from sqlalchemy import (
    ForeignKey,
    String,
    Text,
    DateTime,
    Enum,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

import enum


class OrgRole(str, enum.Enum):
    """Role of a user WITHIN one organization.

    This is intentionally separate from the global UserRole.
    Global roles gate system-wide powers; OrgRole gates tenant visibility.
    """

    ORG_ADMIN = "ORG_ADMIN"
    INVESTIGATOR = "INVESTIGATOR"
    VIEWER = "VIEWER"


class Organization(Base):
    """A tenant such as a police unit, NGO, or agency.

    Cases may optionally belong to an organization via
    Case.organization_id (NULL = personal case, see Case model).
    """

    __tablename__ = "organizations"

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    name: Mapped[str] = mapped_column(
        String(200),
        nullable=False,
        index=True,
    )

    description: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
    )

    created_by: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
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


class Membership(Base):
    """Links a user to an organization with an organization-scoped role."""

    __tablename__ = "memberships"

    __table_args__ = (
        UniqueConstraint(
            "user_id",
            "organization_id",
            name="uq_membership_user_org",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    organization_id: Mapped[int] = mapped_column(
        ForeignKey("organizations.id"),
        nullable=False,
        index=True,
    )

    role: Mapped[OrgRole] = mapped_column(
        Enum(OrgRole),
        default=OrgRole.INVESTIGATOR,
        nullable=False,
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
