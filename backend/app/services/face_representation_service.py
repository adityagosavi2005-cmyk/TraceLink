"""Face-representation orchestration (Phase 5).

Internal service only: there are no public Phase 5 endpoints.
Consumes Phase 4 FaceDetection rows (never reruns YuNet), reads
the Phase 3 derived image through the existing storage service,
and persists one FaceEmbedding per face with full provenance.

Flow per face:
    current valid COMPLETE run -> derived bytes -> SHA verify ->
    geometry -> adapter.represent -> validate -> insert-or-reuse.

Determinism / idempotency: before generating, look up an existing
row by (face_detection_id, representation_name,
representation_version, model_name, model_version) whose source
SHA matches the current derived image; reuse it when present.
Version changes create new rows; history is never overwritten.
The database unique constraint is the final duplicate guard.

There is intentionally no embedding PROCESSING/FAILED state: a
failed deterministic transformation simply creates no row.

Faces are committed independently in ordinal order, so a rerun
after a partial failure resumes the remaining faces.
"""

import hashlib

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.face_detection import FaceDetectionRun, FaceDetectionStatus
from app.models.face_embedding import FaceEmbedding
from app.services import face_detection_service
from app.services import storage
from app.services.face_representation import (
    FaceGeometry,
    FaceRepresentation,
    RepresentationError,
    validate_vector,
)


def active_representation_identity(
    representation: FaceRepresentation,
) -> tuple[str, str, str, str, str]:
    """Reuse identity, read from the adapter itself.

    Taken from the engine that will generate the vector (rather
    than read separately from configuration) so the lookup key and
    the persisted provenance can never drift apart.
    """
    return (
        representation.representation_name,
        representation.representation_version,
        representation.model_name,
        representation.model_version,
    )


def get_existing_embedding(
    db: Session,
    face_detection_id: int,
    identity: tuple[str, str, str, str, str],
) -> FaceEmbedding | None:
    """Newest row matching the representation identity, if any."""
    rep_name, rep_version, model_name, model_version = identity
    return (
        db.query(FaceEmbedding)
        .filter(
            FaceEmbedding.face_detection_id == face_detection_id,
            FaceEmbedding.representation_name == rep_name,
            FaceEmbedding.representation_version == rep_version,
            FaceEmbedding.model_name == model_name,
            FaceEmbedding.model_version == model_version,
        )
        .order_by(FaceEmbedding.id.desc())
        .first()
    )


def _ensure_current_face(db: Session, face, photo) -> None:
    """Reject faces that are not on a current valid detection run."""
    run = db.get(FaceDetectionRun, face.run_id)
    if (
        run is None
        or run.status != FaceDetectionStatus.COMPLETE
        or run.source_derived_sha != photo.derived_sha256
    ):
        raise RepresentationError(
            "Face representation failed: face detection is not current"
        )


def represent_face(
    db: Session,
    face,
    photo,
    representation: FaceRepresentation | None = None,
    run=None,
) -> FaceEmbedding:
    """Generate (or reuse) the embedding for one FaceDetection.

    The face must belong to a current valid detection run: callers
    processing a whole photo pass the run (represent_photo_faces
    does this); otherwise the face's own run must be COMPLETE and
    match the photo's current derived SHA. Raises
    RepresentationError when no valid representation can be
    produced; creates no row in that case.
    """
    if representation is None:
        from app.services.sface_representation import (
            get_face_representation,
        )

        representation = get_face_representation()
    if run is not None and face.run_id != run.id:
        raise RepresentationError(
            "Face representation failed: face detection is not current"
        )
    _ensure_current_face(db, face, photo)
    identity = active_representation_identity(representation)
    try:
        derived_bytes = storage.get_derived_bytes(
            photo.storage_key_derived
        )
    except Exception as exc:
        raise RepresentationError(
            "Face representation failed: "
            "derived image is unavailable; try again later"
        ) from exc
    if hashlib.sha256(derived_bytes).hexdigest() != photo.derived_sha256:
        raise RepresentationError(
            "Face representation failed: "
            "derived image changed during processing"
        )
    existing = get_existing_embedding(db, face.id, identity)
    if existing is not None:
        if existing.source_derived_sha256 == photo.derived_sha256:
            return existing
        raise RepresentationError(
            "Face representation failed: "
            "a representation already exists for a different "
            "source image"
        )
    geometry = FaceGeometry(
        x_min=face.x_min,
        y_min=face.y_min,
        x_max=face.x_max,
        y_max=face.y_max,
        landmarks=face.landmarks,
    )
    vector = validate_vector(
        representation.represent(derived_bytes, geometry)
    )
    row = FaceEmbedding(
        face_detection_id=face.id,
        source_derived_sha256=photo.derived_sha256,
        representation_name=representation.representation_name,
        representation_version=representation.representation_version,
        model_name=representation.model_name,
        model_version=representation.model_version,
        model_sha256=representation.model_sha256,
        dimension=representation.dimension,
        embedding=vector,
    )
    db.add(row)
    try:
        db.commit()
    except IntegrityError as exc:
        # Lost a uniqueness race: reuse the winner when it is valid
        # for the current source image, else surface a clean error.
        db.rollback()
        winner = get_existing_embedding(db, face.id, identity)
        if (
            winner is not None
            and winner.source_derived_sha256 == photo.derived_sha256
        ):
            return winner
        raise RepresentationError(
            "Face representation failed: "
            "a representation already exists for a different "
            "source image"
        ) from exc
    db.refresh(row)
    return row


def represent_photo_faces(
    db: Session,
    kind: str,
    photo,
    representation: FaceRepresentation | None = None,
) -> list[FaceEmbedding]:
    """Generate (or reuse) embeddings for a photo's current faces.

    Returns one embedding per face of the current valid COMPLETE
    run. Zero faces yields zero embeddings; a photo with no
    current valid run (never detected, FAILED, or stale) yields
    zero embeddings without creating anything.
    """
    face_detection_service.ensure_ready(photo)
    if representation is None:
        from app.services.sface_representation import (
            get_face_representation,
        )

        representation = get_face_representation()
    run = face_detection_service.get_current_run(db, kind, photo)
    if run is None:
        return []
    faces = face_detection_service.get_run_faces(db, run.id)
    return [
        represent_face(db, face, photo, representation, run)
        for face in faces
    ]
