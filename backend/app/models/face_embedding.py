from datetime import datetime, timezone

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    Enum,
    ForeignKey,
    Integer,
    String,
    Index,
    UniqueConstraint,
    text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base
from app.models.enhancement import ImageSourceType
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
        # representation/model version + the same consumed source
        # image must never duplicate. model_sha256 is provenance,
        # not identity, and is deliberately excluded.
        # Phase 7 extends the identity with the source so the same
        # face can hold one normal (DERIVED) embedding plus one
        # embedding per enhancement output without collapsing.
        # Phase 8 adds a partial unique index (restored rows only)
        # over the same identity plus face_restoration_run_id, so
        # restored embeddings cannot duplicate per restoration run
        # while normal (NULL) rows keep exactly the old behavior
        # (PostgreSQL treats NULLs as distinct in plain UNIQUE).
        UniqueConstraint(
            "face_detection_id",
            "representation_name",
            "representation_version",
            "model_name",
            "model_version",
            "source_type",
            "source_sha256",
            name="uq_face_embeddings_source_identity",
        ),
        Index(
            "uq_face_embeddings_restored_identity",
            "face_detection_id",
            "representation_name",
            "representation_version",
            "model_name",
            "model_version",
            "face_restoration_run_id",
            unique=True,
            postgresql_where=text(
                "face_restoration_run_id IS NOT NULL"
            ),
            sqlite_where=text(
                "face_restoration_run_id IS NOT NULL"
            ),
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True, index=True)

    face_detection_id: Mapped[int] = mapped_column(
        ForeignKey("face_detections.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # SHA-256 of the exact Phase 3 derived image the representation
    # traces back to. Must match the photo's derived_sha256.
    # (For enhanced faces this is the grandparent SHA; the exact
    # artifact consumed is identified by source_type/source_sha256.)
    source_derived_sha256: Mapped[str] = mapped_column(
        String(64), nullable=False
    )

    # ---- Phase 7: source-aware provenance ----
    # Which artifact SFace actually consumed. DERIVED rows point at
    # the Phase 3 image (source_sha256 == source_derived_sha256);
    # ENHANCED rows point at one enhancement output. The extended
    # uniqueness constraint lets normal and enhanced embeddings of
    # the same face coexist without collapsing.
    source_type: Mapped[ImageSourceType] = mapped_column(
        Enum(ImageSourceType),
        default=ImageSourceType.DERIVED,
        nullable=False,
    )
    source_sha256: Mapped[str] = mapped_column(
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

    # ---- Phase 8: restored-face provenance ----
    # NULL for normal (whole-photo source) embeddings; set to the
    # FaceRestorationRun whose artifact SFace consumed for restored
    # embeddings. DERIVED/ENHANCED source semantics are unchanged.
    face_restoration_run_id: Mapped[int | None] = mapped_column(
        ForeignKey("face_restoration_runs.id", ondelete="CASCADE"),
        nullable=True,
        default=None,
        index=True,
    )

    face_detection: Mapped["FaceDetection"] = relationship(
        "FaceDetection", back_populates="embeddings"
    )

    face_restoration_run: Mapped["FaceRestorationRun | None"] = (
        relationship(
            "FaceRestorationRun", back_populates="embeddings"
        )
    )
