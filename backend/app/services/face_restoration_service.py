"""Face-restoration orchestration (Phase 8).

Shared by the case-photo and sighting-photo restoration routers so
both photo types restore identically. Runs synchronously inside the
triggering request; no queues or workers (Phase 8 v1).

Flow per explicit trigger for one selected face:
    READY photo -> valid COMPLETE DERIVED detection -> derived
    bytes -> SHA verify -> prepare face -> SHA input -> PROCESSING
    run -> restorer -> validate output -> SHA-256 -> restored-faces/
    object -> COMPLETE run. Any failure -> FAILED run (the photo's
    Phase 3 status, the original/derived artifacts, and the
    selected FaceDetection are never touched).

The restorer stays face-agnostic (prepared bytes in). Face
selection is validated BEFORE any run is created (unknown or
stale faces -> 404/409, never history pollution). History is
preserved: every trigger appends a new run, even for the same
face. The active guard is scoped per (photo, face): at most one
fresh PROCESSING run per selected face; Face A PROCESSING never
blocks Face B.

This module performs persistence and storage reads on the caller's
behalf, but never authorization and never key construction: routers
enforce permissions and build restored-face keys via storage.
"""

import hashlib
import json
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.case_photo import PhotoStatus
from app.models.enhancement import ImageSourceType
from app.models.face_detection import (
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.face_embedding import FaceEmbedding
from app.models.face_restoration import (
    FaceRestorationRun,
    FaceRestorationStatus,
    RestoredGeometryKind,
)
from app.services import image_source, storage
from app.services.face_preparation import (
    CANONICAL_GEOMETRY_VERSION,
    FACE_PREP_VERSION,
    PreparationError,
    canonical_restored_landmarks,
    map_landmarks_to_prepared,
    prepare_face,
)
from app.services.face_representation import (
    FaceGeometry,
    RepresentationError,
    validate_vector,
)
from app.services.face_restoration import (
    FaceRestorer,
    RestorationError,
)
from app.services.preprocessing import (
    STALE_PROCESSING_THRESHOLD_SECONDS,
)


def active_restorer_identity() -> tuple[str, str, str, str]:
    """Currently valid (restorer, version, model, model version)."""
    from app.services.gfpgan_adapter import GFPGANAdapter

    adapter = GFPGANAdapter()
    return (
        adapter.name,
        adapter.version,
        adapter.model_name,
        adapter.model_version,
    )


def _photo_parent(photo, kind: str) -> tuple[int | None, int | None]:
    if kind == "case":
        return photo.id, None
    return None, photo.id


def _runs_query(db: Session, kind: str, photo_id: int):
    column = (
        FaceRestorationRun.case_photo_id
        if kind == "case"
        else FaceRestorationRun.sighting_photo_id
    )
    return db.query(FaceRestorationRun).filter(column == photo_id)


def list_runs(
    db: Session, kind: str, photo_id: int, face_id: int
) -> list[FaceRestorationRun]:
    """All runs for one selected face, newest first."""
    column = (
        FaceRestorationRun.case_photo_id
        if kind == "case"
        else FaceRestorationRun.sighting_photo_id
    )
    return (
        db.query(FaceRestorationRun)
        .filter(
            column == photo_id,
            FaceRestorationRun.face_detection_id == face_id,
        )
        .order_by(FaceRestorationRun.id.desc())
        .all()
    )


def get_run(
    db: Session, kind: str, photo, face_id: int, run_id: int
) -> FaceRestorationRun | None:
    """One run scoped to its photo and face (None otherwise)."""
    run = db.get(FaceRestorationRun, run_id)
    if run is None:
        return None
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    if (
        run.case_id != photo.case_id
        or run.case_photo_id != case_photo_id
        or run.sighting_photo_id != sighting_photo_id
        or run.face_detection_id != face_id
    ):
        return None
    return run


def ensure_ready(photo) -> None:
    """Reject restoration unless Phase 3 produced a derived image."""
    if (
        photo.processing_status != PhotoStatus.READY
        or not photo.storage_key_derived
        or not photo.derived_sha256
        or photo.derived_width is None
        or photo.derived_height is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Photo is not ready for face restoration",
        )


def load_selected_face(
    db: Session, kind: str, photo, face_id: int
) -> tuple[FaceDetection, FaceDetectionRun]:
    """Validate an explicitly selected face (no ordinal inference).

    The face must exist, belong to the requested photo and case,
    sit on a COMPLETE DERIVED detection run that is still current
    for the photo's derived SHA. Faces from ENHANCED detection
    runs are rejected: Phase 8 v1 consumes Phase 3 derived bytes
    only (no Real-ESRGAN -> GFPGAN chaining).
    """
    face = db.get(FaceDetection, face_id)
    if face is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face not found",
        )
    if kind == "case":
        if (
            face.case_photo_id != photo.id
            or face.sighting_photo_id is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
    else:
        if (
            face.sighting_photo_id != photo.id
            or face.case_photo_id is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
    if face.case_id != photo.case_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face not found",
        )
    run = db.get(FaceDetectionRun, face.run_id)
    if run is None or run.status != FaceDetectionStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face detection is not current for this face; "
            "run face detection first",
        )
    if (
        run.source_type != ImageSourceType.DERIVED
        or run.enhancement_run_id is not None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration supports only faces detected "
            "on the Phase 3 derived image",
        )
    if (
        run.source_derived_sha != photo.derived_sha256
        or run.source_sha256 != photo.derived_sha256
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face detection is stale for the current derived "
            "image; run face detection again",
        )
    return face, run


def is_run_stale(run: FaceRestorationRun, now=None) -> bool:
    """True when a PROCESSING run may be safely recovered."""
    if run.status != FaceRestorationStatus.PROCESSING:
        return False
    if run.started_at is None:
        return True
    now = now or datetime.now(timezone.utc)
    started = run.started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (
        now - started
    ).total_seconds() > STALE_PROCESSING_THRESHOLD_SECONDS


def ensure_no_active_run(
    db: Session, kind: str, photo_id: int, face_id: int
) -> None:
    """Reject a trigger that would race an in-progress run.

    Scoped per (photo, face): a fresh PROCESSING run for Face A
    never blocks Face B.
    """
    column = (
        FaceRestorationRun.case_photo_id
        if kind == "case"
        else FaceRestorationRun.sighting_photo_id
    )
    active = (
        db.query(FaceRestorationRun)
        .filter(
            column == photo_id,
            FaceRestorationRun.face_detection_id == face_id,
            FaceRestorationRun.status
            == FaceRestorationStatus.PROCESSING,
        )
        .order_by(FaceRestorationRun.id.desc())
        .first()
    )
    if active is None or is_run_stale(active):
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Face restoration is already running for this face; "
        "try again later",
    )


def _finish_run_failed(
    db: Session, run: FaceRestorationRun, message: str
) -> FaceRestorationRun:
    db.rollback()
    run.status = FaceRestorationStatus.FAILED
    run.finished_at = datetime.now(timezone.utc)
    run.error_message = message
    db.commit()
    db.refresh(run)
    return run


def _new_run(
    db: Session,
    kind: str,
    photo,
    face,
    detection_run,
    prepared_sha: str,
    transform: dict,
    identity: tuple[str, str, str, str],
    model_sha256: str,
    parameters: str,
) -> FaceRestorationRun:
    name, version, model_name, model_version = identity
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    run = FaceRestorationRun(
        case_id=photo.case_id,
        case_photo_id=case_photo_id,
        sighting_photo_id=sighting_photo_id,
        sighting_id=getattr(photo, "sighting_id", None),
        face_detection_id=face.id,
        face_detection_run_id=detection_run.id,
        status=FaceRestorationStatus.PROCESSING,
        source_derived_sha256=photo.derived_sha256,
        bbox_snapshot=json.dumps(
            {
                "x_min": face.x_min,
                "y_min": face.y_min,
                "x_max": face.x_max,
                "y_max": face.y_max,
                "frame_width": face.frame_width,
                "frame_height": face.frame_height,
            },
            sort_keys=True,
        ),
        landmarks_snapshot=json.dumps(
            face.landmarks, sort_keys=True
        ),
        prep_version=FACE_PREP_VERSION,
        prep_transform=json.dumps(transform, sort_keys=True),
        prepared_input_sha256=prepared_sha,
        restorer_name=name,
        restorer_version=version,
        model_name=model_name,
        model_version=model_version,
        model_sha256=model_sha256,
        parameters=parameters,
        # Decided during _execute_run once the restored output is
        # known; placeholder keeps the row valid meanwhile.
        restored_geometry_kind=RestoredGeometryKind.MAPPED.value,
        restored_geometry_version=FACE_PREP_VERSION,
        restored_geometry=json.dumps({}),
        mime_type="image/jpeg",
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _record_stillborn(
    db: Session,
    kind: str,
    photo,
    face,
    detection_run,
    identity: tuple[str, str, str, str],
    parameters: str,
    message: str,
) -> FaceRestorationRun:
    """Persist a FAILED run when the model cannot even load."""
    name, version, model_name, model_version = identity
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    run = FaceRestorationRun(
        case_id=photo.case_id,
        case_photo_id=case_photo_id,
        sighting_photo_id=sighting_photo_id,
        sighting_id=getattr(photo, "sighting_id", None),
        face_detection_id=face.id,
        face_detection_run_id=detection_run.id,
        status=FaceRestorationStatus.FAILED,
        source_derived_sha256=photo.derived_sha256,
        bbox_snapshot=json.dumps(
            {
                "x_min": face.x_min,
                "y_min": face.y_min,
                "x_max": face.x_max,
                "y_max": face.y_max,
                "frame_width": face.frame_width,
                "frame_height": face.frame_height,
            },
            sort_keys=True,
        ),
        landmarks_snapshot=json.dumps(
            face.landmarks, sort_keys=True
        ),
        prep_version=FACE_PREP_VERSION,
        prep_transform=json.dumps({}),
        prepared_input_sha256="0" * 64,
        restorer_name=name,
        restorer_version=version,
        model_name=model_name,
        model_version=model_version,
        # Provenance unknown: the weights file could not be read.
        model_sha256="0" * 64,
        parameters=parameters,
        restored_geometry_kind=RestoredGeometryKind.MAPPED.value,
        restored_geometry_version=FACE_PREP_VERSION,
        restored_geometry=json.dumps({}),
        mime_type="image/jpeg",
        error_message=message,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _decide_restored_geometry(
    transform: dict,
    landmarks_snapshot: dict,
    restored,
    prepared_width: int,
    prepared_height: int,
) -> tuple[str, str, dict]:
    """Choose MAPPED vs CANONICAL restored-frame geometry.

    Locked rule: MAPPED only when the restorer reports no
    internal realignment AND the output frame equals the prepared
    frame, so the recorded crop/pad/scale transform still maps
    source landmarks exactly. Otherwise CANONICAL with the
    versioned synthetic landmark set. Never mixed silently.
    """
    if (
        not restored.realigned
        and restored.width == prepared_width
        and restored.height == prepared_height
    ):
        mapped = map_landmarks_to_prepared(
            landmarks_snapshot, transform
        )
        return (
            RestoredGeometryKind.MAPPED.value,
            FACE_PREP_VERSION,
            {
                "kind": RestoredGeometryKind.MAPPED.value,
                "version": FACE_PREP_VERSION,
                "landmarks": mapped,
                "frame_width": restored.width,
                "frame_height": restored.height,
            },
        )
    canonical = canonical_restored_landmarks(
        restored.width, restored.height
    )
    return (
        RestoredGeometryKind.CANONICAL.value,
        CANONICAL_GEOMETRY_VERSION,
        {
            "kind": RestoredGeometryKind.CANONICAL.value,
            "version": CANONICAL_GEOMETRY_VERSION,
            "landmarks": canonical,
            "frame_width": restored.width,
            "frame_height": restored.height,
        },
    )


def trigger_restoration(
    db: Session,
    kind: str,
    photo,
    face_id: int,
    restored_key_for_sha,
    restorer: FaceRestorer | None = None,
) -> FaceRestorationRun:
    """Start one face-restoration execution (new history row)."""
    ensure_ready(photo)
    face, detection_run = load_selected_face(db, kind, photo, face_id)
    ensure_no_active_run(db, kind, photo.id, face.id)
    if restorer is None:
        from app.services.gfpgan_adapter import get_face_restorer

        restorer = get_face_restorer()
    identity = (
        restorer.name,
        restorer.version,
        restorer.model_name,
        restorer.model_version,
    )
    parameters = json.dumps({"prep_version": FACE_PREP_VERSION})
    try:
        model_sha256 = restorer.model_sha256
    except RestorationError as exc:
        return _record_stillborn(
            db, kind, photo, face, detection_run, identity,
            parameters, str(exc),
        )
    try:
        # read_derived_bytes re-verifies SHA(photo bytes) against
        # the photo row, so tampered/rotated sources fail here.
        derived_bytes = image_source.read_derived_bytes(photo)
    except image_source.SourceResolutionError as exc:
        run = _new_run(
            db, kind, photo, face, detection_run,
            "0" * 64, {}, identity, model_sha256, parameters,
        )
        return _finish_run_failed(db, run, str(exc))
    try:
        prepared = prepare_face(
            derived_bytes,
            photo.derived_width,
            photo.derived_height,
            face.x_min,
            face.y_min,
            face.x_max,
            face.y_max,
            face.landmarks,
        )
    except PreparationError as exc:
        run = _new_run(
            db, kind, photo, face, detection_run,
            "0" * 64, {}, identity, model_sha256, parameters,
        )
        return _finish_run_failed(db, run, str(exc))
    (prepared_bytes, prepared_sha, prep_w, prep_h, _, transform) = (
        prepared
    )
    if prep_w * prep_h > settings.RESTORATION_MAX_FACE_PIXELS:
        run = _new_run(
            db, kind, photo, face, detection_run,
            prepared_sha, transform, identity, model_sha256,
            parameters,
        )
        return _finish_run_failed(
            db, run, "Face restoration failed; try again later"
        )
    run = _new_run(
        db, kind, photo, face, detection_run,
        prepared_sha, transform, identity, model_sha256,
        parameters,
    )
    return _execute_run(
        db, photo, face, run, transform, prepared_bytes,
        prep_w, prep_h, restorer, restored_key_for_sha,
    )


def _execute_run(
    db: Session,
    photo,
    face,
    run: FaceRestorationRun,
    transform: dict,
    prepared_bytes: bytes,
    prep_w: int,
    prep_h: int,
    restorer,
    restored_key_for_sha,
) -> FaceRestorationRun:
    """Run the restorer on prepared bytes, persist the artifact."""
    try:
        restored = restorer.restore(prepared_bytes)
    except RestorationError as exc:
        return _finish_run_failed(db, run, str(exc))
    except Exception:
        return _finish_run_failed(
            db, run, "Face restoration failed; try again later"
        )
    if (
        not restored.image_bytes
        or restored.width <= 0
        or restored.height <= 0
    ):
        return _finish_run_failed(
            db, run, "Face restoration failed; try again later"
        )
    try:
        geometry_kind, geometry_version, geometry = (
            _decide_restored_geometry(
                transform,
                json.loads(run.landmarks_snapshot),
                restored,
                prep_w,
                prep_h,
            )
        )
    except (PreparationError, ValueError):
        return _finish_run_failed(
            db, run, "Face restoration failed; try again later"
        )
    # Auxiliary-model provenance when the restorer reports it
    # (GFPGANAdapter exposes aux_model_info after a real load;
    # test doubles leave it None and the column stays NULL).
    aux_model_info = getattr(restorer, "aux_model_info", None)
    if aux_model_info is not None:
        aux_model_info = str(aux_model_info)
        if len(aux_model_info) > 1000:
            aux_model_info = aux_model_info[:1000]
        run.aux_model_info = aux_model_info
    output_sha = hashlib.sha256(restored.image_bytes).hexdigest()
    storage_key = restored_key_for_sha(output_sha)
    try:
        storage.put_restored_face(
            storage_key, restored.image_bytes, restored.mime_type
        )
    except Exception:
        return _finish_run_failed(
            db, run, "Restored face storage is unavailable; retry later"
        )
    run.restored_geometry_kind = geometry_kind
    run.restored_geometry_version = geometry_version
    run.restored_geometry = json.dumps(geometry, sort_keys=True)
    run.output_sha256 = output_sha
    run.storage_key = storage_key
    run.mime_type = restored.mime_type
    run.width = restored.width
    run.height = restored.height
    run.byte_size = len(restored.image_bytes)
    run.status = FaceRestorationStatus.COMPLETE
    run.finished_at = datetime.now(timezone.utc)
    run.error_message = None
    try:
        db.commit()
    except Exception:
        try:
            storage.delete_prefix(storage_key)
        except Exception:
            pass
        return _finish_run_failed(
            db, run, "Face restoration failed; try again later"
        )
    db.refresh(run)
    return run


def retry_restoration(
    db: Session,
    kind: str,
    photo,
    face_id: int,
    run_id: int,
    restored_key_for_sha,
    restorer: FaceRestorer | None = None,
) -> FaceRestorationRun:
    """Explicit re-run of a FAILED or stale PROCESSING run.

    Always appends a NEW run (history preserved, never
    deduplicated). Fresh PROCESSING runs conflict; COMPLETE runs
    cannot be "retried" -- trigger a fresh restoration instead.
    """
    ensure_ready(photo)
    face, _ = load_selected_face(db, kind, photo, face_id)
    existing = get_run(db, kind, photo, face.id, run_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face restoration run not found",
        )
    if existing.status == FaceRestorationStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration run already completed; "
            "start a new restoration instead",
        )
    if existing.status == FaceRestorationStatus.PROCESSING and (
        not is_run_stale(existing)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration is already running for this "
            "face; try again later",
        )
    return trigger_restoration(
        db, kind, photo, face.id, restored_key_for_sha, restorer
    )


def delete_runs_for_photo(db: Session, kind: str, photo_id: int) -> None:
    """Remove all restoration runs for one photo. Caller commits."""
    for run in _runs_query(db, kind, photo_id).all():
        db.delete(run)


def read_restoration_bytes(run: FaceRestorationRun) -> bytes:
    """Return artifact bytes when present and SHA-verified."""
    if not run.storage_key or not run.output_sha256:
        raise RestorationError(
            "Face restoration run is not complete"
        )
    try:
        data = storage.get_restored_face_bytes(run.storage_key)
    except Exception as exc:
        raise RestorationError(
            "Restored face is unavailable; try again later"
        ) from exc
    if hashlib.sha256(data).hexdigest() != run.output_sha256:
        raise RestorationError(
            "Restored face changed during processing"
        )
    return data


def load_valid_restoration(
    db: Session, kind: str, photo, face, run_id: int
) -> FaceRestorationRun:
    """Return the restoration run when the full chain verifies.

    Existence/ownership failures read as "not found" (no
    cross-face probing); wrong-state and stale-source failures
    read as conflicts. Raises HTTPException either way.
    """
    run = get_run(db, kind, photo, face.id, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face restoration run not found",
        )
    if run.status != FaceRestorationStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration run is not complete",
        )
    if run.face_detection_run_id != face.run_id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration run is stale for the current "
            "face detection; restore the face again",
        )
    if run.source_derived_sha256 != photo.derived_sha256:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Face restoration run is stale for the current "
            "derived image",
        )
    return run


def restored_face_geometry(run: FaceRestorationRun) -> FaceGeometry:
    """Restored-frame SFace geometry recorded on the run.

    Never the original YuNet coordinates: those belong to the
    source-photo frame. Raises RepresentationError when the
    record is missing or malformed.
    """
    try:
        record = json.loads(run.restored_geometry)
        landmarks = record["landmarks"]
        width = int(record["frame_width"])
        height = int(record["frame_height"])
    except (ValueError, KeyError, TypeError) as exc:
        raise RepresentationError(
            "Face representation failed: invalid restored geometry"
        ) from exc
    if width <= 0 or height <= 0:
        raise RepresentationError(
            "Face representation failed: invalid restored geometry"
        )
    return FaceGeometry(
        x_min=0,
        y_min=0,
        x_max=width,
        y_max=height,
        landmarks=landmarks,
    )


def get_existing_restored_embedding(
    db: Session,
    face_detection_id: int,
    identity: tuple[str, str, str, str],
    restoration_run_id: int,
) -> FaceEmbedding | None:
    """Newest restored embedding for one (face, restoration run)."""
    rep_name, rep_version, model_name, model_version = identity
    return (
        db.query(FaceEmbedding)
        .filter(
            FaceEmbedding.face_detection_id == face_detection_id,
            FaceEmbedding.representation_name == rep_name,
            FaceEmbedding.representation_version == rep_version,
            FaceEmbedding.model_name == model_name,
            FaceEmbedding.model_version == model_version,
            FaceEmbedding.face_restoration_run_id
            == restoration_run_id,
        )
        .order_by(FaceEmbedding.id.desc())
        .first()
    )


def get_restored_embedding_for_run(
    db: Session,
    face,
    photo,
    identity: tuple[str, str, str, str],
    run: FaceRestorationRun,
) -> FaceEmbedding | None:
    """Existing restored embedding usable for one restoration run.

    Prefers the row pinned to this exact run; falls back to a row
    for bit-identical artifact bytes produced by an earlier run
    (content-addressed storage keys make identical outputs share
    one SHA, so re-restoring a face can legitimately converge).
    The fallback row is returned only when its own chain still
    verifies through the same checks.
    """
    direct = get_existing_restored_embedding(
        db, face.id, identity, run.id
    )
    if direct is not None:
        return direct
    if not run.output_sha256:
        return None
    rep_name, rep_version, model_name, model_version = identity
    row = (
        db.query(FaceEmbedding)
        .filter(
            FaceEmbedding.face_detection_id == face.id,
            FaceEmbedding.representation_name == rep_name,
            FaceEmbedding.representation_version == rep_version,
            FaceEmbedding.model_name == model_name,
            FaceEmbedding.model_version == model_version,
            FaceEmbedding.source_sha256 == run.output_sha256,
            FaceEmbedding.face_restoration_run_id.is_not(None),
        )
        .order_by(FaceEmbedding.id.desc())
        .first()
    )
    if row is None:
        return None
    if validate_restoration_candidate(db, row, face, photo) is None:
        return None
    return row


def represent_restored_face(
    db: Session,
    face,
    photo,
    restoration_run_id: int,
    kind: str | None = None,
    representation=None,
) -> FaceEmbedding:
    """Generate (or reuse) the SFace embedding of a restored face.

    SFace consumes the restored artifact bytes plus the
    restored-frame geometry recorded on the run (never the
    original YuNet coordinates). Raises RepresentationError when
    no valid representation can be produced; creates no row then.
    Restored embeddings never enter whole-photo normal
    representation: callers address them only by explicit
    restoration run id.
    """
    if representation is None:
        from app.services.sface_representation import (
            get_face_representation,
        )

        representation = get_face_representation()
    if kind is None:
        kind = (
            "case" if face.case_photo_id is not None else "sighting"
        )
    run = load_valid_restoration(
        db, kind, photo, face, restoration_run_id
    )
    try:
        artifact_bytes = read_restoration_bytes(run)
    except RestorationError as exc:
        raise RepresentationError(
            "Face representation failed: %s" % exc
        ) from exc
    from app.services.face_representation_service import (
        active_representation_identity,
    )

    identity = active_representation_identity(representation)
    existing = get_restored_embedding_for_run(
        db, face, photo, identity, run
    )
    if existing is not None:
        return existing
    geometry = restored_face_geometry(run)
    vector = validate_vector(
        representation.represent(artifact_bytes, geometry)
    )
    row = FaceEmbedding(
        face_detection_id=face.id,
        source_derived_sha256=photo.derived_sha256,
        # Grandparent-compatible source marker: the exact artifact
        # consumed is identified by face_restoration_run_id plus
        # source_sha256 (the artifact output SHA). DERIVED/ENHANCED
        # whole-photo semantics are untouched.
        source_type=ImageSourceType.DERIVED,
        source_sha256=run.output_sha256,
        face_restoration_run_id=run.id,
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
    except IntegrityError:
        db.rollback()
        winner = get_restored_embedding_for_run(
            db, face, photo, identity, run
        )
        if winner is not None:
            return winner
        raise RepresentationError(
            "Face representation failed: "
            "a representation already exists for a different "
            "restored artifact"
        )
    db.refresh(row)
    return row


def validate_restoration_candidate_by_run(
    db: Session, run: FaceRestorationRun, face, photo
) -> FaceRestorationRun | None:
    """Validate a restoration run against its face and photo.

    None when any link fails (not COMPLETE, wrong face/photo,
    stale derived SHA, artifact missing or SHA-mismatched).
    Silent by design: similarity excludes without internals.
    """
    if run is None or run.status != FaceRestorationStatus.COMPLETE:
        return None
    if (
        run.face_detection_id != face.id
        or run.case_id != photo.case_id
    ):
        return None
    if face.case_photo_id is not None:
        if run.case_photo_id != face.case_photo_id:
            return None
    else:
        if run.sighting_photo_id != face.sighting_photo_id:
            return None
    if run.source_derived_sha256 != photo.derived_sha256:
        return None
    try:
        read_restoration_bytes(run)
    except RestorationError:
        return None
    return run


def validate_restoration_candidate(
    db: Session, embedding: FaceEmbedding, face, photo
) -> FaceRestorationRun | None:
    """Return the restoration run behind a valid restored candidate.

    None when any link of the chain fails (run missing, not
    COMPLETE, wrong face/photo, stale derived SHA, artifact
    missing or SHA-mismatched, embedding SHA mismatch). Silent
    by design: similarity excludes invalid candidates without
    leaking internals.
    """
    run_id = embedding.face_restoration_run_id
    if run_id is None:
        return None
    run = validate_restoration_candidate_by_run(
        db, db.get(FaceRestorationRun, run_id), face, photo
    )
    if run is None:
        return None
    if (
        embedding.source_derived_sha256 != photo.derived_sha256
        or embedding.source_sha256 != run.output_sha256
    ):
        return None
    return run
