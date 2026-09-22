"""Face-representation orchestration (Phase 5 + Phase 7 sources).

Internal service only: there are no public Phase 5 endpoints.
Consumes Phase 4 FaceDetection rows (never reruns YuNet), reads
the resolved source image through the centralized image_source
resolver, and persists one FaceEmbedding per face with full
provenance.

Flow per face:
    valid COMPLETE run -> resolved source bytes -> SHA verify ->
    geometry -> adapter.represent -> validate -> insert-or-reuse.

The source comes from the run row itself: DERIVED runs consume the
Phase 3 image, ENHANCED runs consume their selected enhancement
output (re-verified here; stale/tampered sources raise). Normal
and enhanced embeddings of the same face coexist: the lookup and
uniqueness identity include the consumed source.

Determinism / idempotency: before generating, look up an existing
row by (face_detection_id, representation_name,
representation_version, model_name, model_version, source_type,
source_sha256); reuse it when present. Version or source changes
create new rows; history is never overwritten. The database unique
constraint is the final duplicate guard.

There is intentionally no embedding PROCESSING/FAILED state: a
failed deterministic transformation simply creates no row.

Faces are committed independently in ordinal order, so a rerun
after a partial failure resumes the remaining faces.
"""

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.enhancement import ImageSourceType
from app.models.face_detection import FaceDetectionRun, FaceDetectionStatus
from app.models.face_embedding import FaceEmbedding
from app.services import face_detection_service
from app.services import image_source
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
    source_type: ImageSourceType | None = None,
    source_sha256: str | None = None,
) -> FaceEmbedding | None:
    """Newest row matching the representation identity, if any.

    When a source is given, only rows produced from that exact
    source image match, so normal and enhanced embeddings coexist
    instead of colliding. Without a source the newest row of any
    source matches (legacy lookup for validity checks).
    """
    rep_name, rep_version, model_name, model_version = identity
    query = db.query(FaceEmbedding).filter(
        FaceEmbedding.face_detection_id == face_detection_id,
        FaceEmbedding.representation_name == rep_name,
        FaceEmbedding.representation_version == rep_version,
        FaceEmbedding.model_name == model_name,
        FaceEmbedding.model_version == model_version,
    )
    if source_type is not None:
        query = query.filter(
            FaceEmbedding.source_type == source_type,
            FaceEmbedding.source_sha256 == source_sha256,
        )
    return query.order_by(FaceEmbedding.id.desc()).first()


def _resolve_face_source(db: Session, face, photo, run=None):
    """Validate the face's run and resolve the bytes SFace consumes.

    Normal faces must sit on a current valid detection run;
    enhanced faces must sit on a COMPLETE run whose enhancement
    chain still verifies (COMPLETE, artifact present, SHAs match).
    Raises RepresentationError otherwise.
    """
    if run is not None and face.run_id != run.id:
        raise RepresentationError(
            "Face representation failed: face detection is not current"
        )
    run = run if run is not None else db.get(FaceDetectionRun, face.run_id)
    if run is None or run.status != FaceDetectionStatus.COMPLETE:
        raise RepresentationError(
            "Face representation failed: face detection is not current"
        )
    try:
        return image_source.resolve_run_source(db, run, photo)
    except image_source.SourceResolutionError as exc:
        # Resolver messages are safe (no internals); surface them
        # so enhanced staleness reads differently from "not current".
        raise RepresentationError(
            "Face representation failed: %s" % exc
        ) from exc


def represent_face(
    db: Session,
    face,
    photo,
    representation: FaceRepresentation | None = None,
    run=None,
) -> FaceEmbedding:
    """Generate (or reuse) the embedding for one FaceDetection.

    The face must belong to a valid detection run: callers
    processing a whole photo pass the run (represent_photo_faces
    does this); otherwise the face's own run is used. Normal runs
    must be COMPLETE and current for the derived image; enhanced
    runs must be COMPLETE with a still-valid enhancement chain.
    SFace consumes the run's own source (derived or enhanced
    bytes). Raises RepresentationError when no valid
    representation can be produced; creates no row in that case.
    """
    if representation is None:
        from app.services.sface_representation import (
            get_face_representation,
        )

        representation = get_face_representation()
    source = _resolve_face_source(db, face, photo, run)
    identity = active_representation_identity(representation)
    existing = get_existing_embedding(
        db,
        face.id,
        identity,
        source.source_type,
        source.source_sha256,
    )
    if existing is not None:
        return existing
    geometry = FaceGeometry(
        x_min=face.x_min,
        y_min=face.y_min,
        x_max=face.x_max,
        y_max=face.y_max,
        landmarks=face.landmarks,
    )
    vector = validate_vector(
        representation.represent(source.image_bytes, geometry)
    )
    row = FaceEmbedding(
        face_detection_id=face.id,
        source_derived_sha256=photo.derived_sha256,
        source_type=source.source_type,
        source_sha256=source.source_sha256,
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
        # Lost a uniqueness race: reuse the winner when it matches
        # this exact source, else surface a clean error.
        db.rollback()
        winner = get_existing_embedding(
            db,
            face.id,
            identity,
            source.source_type,
            source.source_sha256,
        )
        if winner is not None:
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
