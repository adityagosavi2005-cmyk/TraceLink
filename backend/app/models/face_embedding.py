from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.services.face_representation import REPRESENTATION_DIMENSION


class FaceEmbedding(Base):
    """One numerical face representation for one detected face.

    Phase 5 stores HOW a Phase 4 FaceDetection was numerically
    represented; the detection row itself stores WHAT was found.
    One face may gain several rows over time as representation or
    model versions change: history is preserved, rows are never
    overwritten. No identity, similarity, or demographic fields:
    representation only.
    """

    __tablename__ = "face_embeddings"
    __table_args__ = (
        # Representation identity: the same face + the same
        # representation/model version must never duplicate.
        # model_sha256 is provenance, not identity, and is
        # deliberately excluded.
        UniqueConstraint(
            "face_detection_id",
            "representation_name",
            "representation_version",
            "model_name",
            "model_version",
            name="uq_face_embeddings_identity",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    face_detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 of the exact Phase 3 derived image the representation
    # was generated from. Must match the photo's derived_sha256.
    source_derived_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    representation_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    representation_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    model_name: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    model_version: Mapped[str] = mapped_column(
        String(50), nullable=False
    )
    # SHA-256 of the exact model file loaded. Provenance only.
    model_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    dimension: Mapped[int] = mapped_column(
        Integer, nullable=False
    )

    embedding: Mapped[list] = mapped_column(
        Vector(REPRESENTATION_DIMENSION), nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    face_detection: Mapped["FaceDetection"] = relationship(
        "FaceDetection", back_populates="embeddings"
    )
