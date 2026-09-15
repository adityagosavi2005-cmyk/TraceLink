from datetime import datetime, timezone

from sqlalchemy import String, DateTime, Enum
from sqlalchemy.orm import Mapped, mapped_column

from app.core.database import Base

import enum


class UserRole(str, enum.Enum):
    REPORTER = "REPORTER"
    ORGANIZATION_MEMBER = "ORGANIZATION_MEMBER"
    REVIEWER = "REVIEWER"
    ADMIN = "ADMIN"


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        primary_key=True
    )

    name: Mapped[str] = mapped_column(
        String(100),
        nullable=False
    )

    email: Mapped[str] = mapped_column(
        String(255),
        unique=True,
        nullable=False,
        index=True
    )

    password_hash: Mapped[str] = mapped_column(
        String(255),
        nullable=False
    )

    role: Mapped[UserRole] = mapped_column(
        Enum(UserRole),
        default=UserRole.REPORTER,
        nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda:datetime.now(timezone.utc),
        nullable=False
    )

    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda:datetime.now(timezone.utc),
        onupdate=lambda:datetime.now(timezone.utc),
        nullable=False
    )