"""Face-detection orchestration (Phase 4 + Phase 7 sources).

Shared by the case-photo and sighting-photo face routers so both
photo types detect identically. Runs synchronously inside the
triggering request; no queues or workers.

Flow per explicit trigger:
    READY photo -> reuse current COMPLETE run, else
    PROCESSING run -> resolved source bytes -> detector ->
    normalize/clamp/filter -> face rows -> COMPLETE run.
    Any failure -> FAILED run (the photo's Phase 3 status is never
    touched).

The detector stays source-agnostic (image bytes + dimensions in).
Source selection is centralized in image_source: the normal path
consumes the Phase 3 derived image, while an explicit
enhancement_run_id selects one COMPLETE enhancement output (never
the latest). Coordinates are recorded in source-image pixels.

This module performs persistence and storage reads on the caller's
behalf, but never authorization and never key construction: routers
enforce permissions; storage keys come from the photo row (derived)
or the enhancement run row (enhanced).
"""

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.case_photo import PhotoStatus
from app.models.enhancement import ImageSourceType
from app.models.face_detection import (
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.services import image_source
from app.services.face_detector import DetectedFace, FaceDetector
from app.services.image_source import ImageSource
from app.services.preprocessing import STALE_PROCESSING_THRESHOLD_SECONDS


def active_detector_identity() -> tuple[str, str, float]:
    """Currently valid (name, version, threshold) from configuration."""
    return (
        settings.FACE_DETECTOR_NAME,
        settings.FACE_DETECTOR_VERSION,
        settings.FACE_DETECTION_THRESHOLD,
    )


def _photo_parent(photo, kind: str) -> tuple[int | None, int | None]:
    if kind == "case":
        return photo.id, None
    return None, photo.id


def _runs_query(db: Session, kind: str, photo_id: int):
    column = (
        FaceDetectionRun.case_photo_id
        if kind == "case"
        else FaceDetectionRun.sighting_photo_id
    )
    return db.query(FaceDetectionRun).filter(column == photo_id)


def get_current_run(
    db: Session, kind: str, photo
) -> FaceDetectionRun | None:
    """Newest COMPLETE normal run matching active config + source.

    Only DERIVED runs qualify: enhanced faces are never silently
    chosen as "current". Callers wanting an enhanced result must
    resolve it explicitly via get_current_enhanced_run.
    """
    name, version, threshold = active_detector_identity()
    return (
        _runs_query(db, kind, photo.id)
        .filter(
            FaceDetectionRun.status == FaceDetectionStatus.COMPLETE,
            FaceDetectionRun.detector_name == name,
            FaceDetectionRun.detector_version == version,
            FaceDetectionRun.threshold == threshold,
            FaceDetectionRun.source_type == ImageSourceType.DERIVED,
            FaceDetectionRun.source_derived_sha == photo.derived_sha256,
            FaceDetectionRun.source_sha256 == photo.derived_sha256,
        )
        .order_by(FaceDetectionRun.id.desc())
        .first()
    )


def get_current_enhanced_run(
    db: Session, kind: str, photo, enhancement_run_id: int
) -> FaceDetectionRun | None:
    """Newest COMPLETE run on one explicitly selected enhancement."""
    name, version, threshold = active_detector_identity()
    return (
        _runs_query(db, kind, photo.id)
        .filter(
            FaceDetectionRun.status == FaceDetectionStatus.COMPLETE,
            FaceDetectionRun.detector_name == name,
            FaceDetectionRun.detector_version == version,
            FaceDetectionRun.threshold == threshold,
            FaceDetectionRun.source_type == ImageSourceType.ENHANCED,
            FaceDetectionRun.enhancement_run_id == enhancement_run_id,
        )
        .order_by(FaceDetectionRun.id.desc())
        .first()
    )


def _resolution_error_to_http(exc: Exception):
    """Map source-resolution failures to router-facing errors."""
    from fastapi import HTTPException, status as http_status

    from app.services.image_source import SourceResolutionError

    if isinstance(exc, SourceResolutionError) and (
        "not found" in str(exc).lower()
    ):
        return HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=str(exc),
        )
    return HTTPException(
        status_code=http_status.HTTP_409_CONFLICT,
        detail=str(exc)
        if isinstance(exc, SourceResolutionError)
        else "Face detection failed; try again later",
    )


def get_run_faces(
    db: Session, run_id: int
) -> list[FaceDetection]:
    return (
        db.query(FaceDetection)
        .filter(FaceDetection.run_id == run_id)
        .order_by(FaceDetection.ordinal.asc())
        .all()
    )


def ensure_ready(photo) -> None:
    """Reject detection unless Phase 3 produced a derived image."""
    if (
        photo.processing_status != PhotoStatus.READY
        or not photo.storage_key_derived
        or not photo.derived_sha256
        or photo.derived_width is None
        or photo.derived_height is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Photo is not ready for face detection",
        )


def ensure_no_active_run(db: Session, kind: str, photo_id: int) -> None:
    """Reject a trigger that would race an in-progress run."""
    active = (
        _runs_query(db, kind, photo_id)
        .filter(FaceDetectionRun.status == FaceDetectionStatus.PROCESSING)
        .order_by(FaceDetectionRun.id.desc())
        .first()
    )
    if active is None:
        return
    now = datetime.now(timezone.utc)
    started = active.started_at
    if started is not None and started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    if started is None or (
        now - started
    ).total_seconds() > STALE_PROCESSING_THRESHOLD_SECONDS:
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Face detection is already running; try again later",
    )


def normalize_faces(
    detected: list[DetectedFace], width: int, height: int
) -> list[DetectedFace]:
    """Clamp, validate, and deterministically order raw detections.

    Coordinates are kept in source-image pixels. Zero-area or
    out-of-frame boxes are dropped; survivors are ordered by
    confidence (desc), then coordinates, so ordinals are stable.
    """
    cleaned: list[DetectedFace] = []
    for face in detected:
        x_min = max(0, min(face.x_min, width))
        y_min = max(0, min(face.y_min, height))
        x_max = max(0, min(face.x_max, width))
        y_max = max(0, min(face.y_max, height))
        if x_max <= x_min or y_max <= y_min:
            continue
        cleaned.append(
            DetectedFace(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                confidence=face.confidence,
                landmarks=face.landmarks,
            )
        )
    cleaned.sort(
        key=lambda f: (
            -f.confidence,
            f.x_min,
            f.y_min,
            f.x_max,
            f.y_max,
        )
    )
    return cleaned


def _finish_run_failed(
    db: Session, run: FaceDetectionRun, message: str
) -> FaceDetectionRun:
    run.status = FaceDetectionStatus.FAILED
    run.finished_at = datetime.now(timezone.utc)
    run.error = message
    run.face_count = 0
    db.commit()
    db.refresh(run)
    return run


def _declared_source(photo, resolved=None) -> dict:
    """Provenance a new run row declares for its attempted source.

    For the normal path this is known before any bytes are read
    (the Phase 3 artifact); for the enhanced path it comes from
    the already-validated resolution. A run that later fails still
    records what it attempted, never what it consumed.
    """
    if resolved is not None:
        return {
            "source_type": resolved.source_type,
            "source_sha256": resolved.source_sha256,
            "enhancement_run_id": (
                resolved.enhancement_run.id
                if resolved.enhancement_run is not None
                else None
            ),
            "source_width": resolved.width,
            "source_height": resolved.height,
            "source_derived_sha": resolved.source_derived_sha,
        }
    return {
        "source_type": ImageSourceType.DERIVED,
        "source_sha256": photo.derived_sha256,
        "enhancement_run_id": None,
        "source_width": photo.derived_width,
        "source_height": photo.derived_height,
        "source_derived_sha": photo.derived_sha256,
    }


def _new_run(db: Session, kind: str, photo, declared: dict) -> (
    FaceDetectionRun
):
    name, version, threshold = active_detector_identity()
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    run = FaceDetectionRun(
        case_id=photo.case_id,
        case_photo_id=case_photo_id,
        sighting_photo_id=sighting_photo_id,
        sighting_id=getattr(photo, "sighting_id", None),
        status=FaceDetectionStatus.PROCESSING,
        detector_name=name,
        detector_version=version,
        threshold=threshold,
        source_derived_sha=declared["source_derived_sha"],
        source_derived_width=photo.derived_width,
        source_derived_height=photo.derived_height,
        source_type=declared["source_type"],
        source_sha256=declared["source_sha256"],
        enhancement_run_id=declared["enhancement_run_id"],
        source_width=declared["source_width"],
        source_height=declared["source_height"],
        face_count=0,
        error=None,
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _execute_run(
    db: Session,
    kind: str,
    photo,
    run: FaceDetectionRun,
    detector,
    source: ImageSource,
) -> FaceDetectionRun:
    """Run the detector on resolved source bytes, persist faces."""
    try:
        detected = detector.detect(
            source.image_bytes, source.width, source.height
        )
    except Exception as exc:
        from app.services.face_detector import DetectorError

        message = (
            str(exc)
            if isinstance(exc, DetectorError)
            else "Face detection failed; try again later"
        )
        return _finish_run_failed(db, run, message)
    faces = normalize_faces(
        detected or [], source.width, source.height
    )
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    for ordinal, face in enumerate(faces):
        db.add(
            FaceDetection(
                run_id=run.id,
                case_id=photo.case_id,
                case_photo_id=case_photo_id,
                sighting_photo_id=sighting_photo_id,
                ordinal=ordinal,
                x_min=face.x_min,
                y_min=face.y_min,
                x_max=face.x_max,
                y_max=face.y_max,
                confidence=face.confidence,
                frame_width=source.width,
                frame_height=source.height,
                landmarks=face.landmarks,
            )
        )
    run.status = FaceDetectionStatus.COMPLETE
    run.finished_at = datetime.now(timezone.utc)
    run.error = None
    run.face_count = len(faces)
    db.commit()
    db.refresh(run)
    return run


def _resolve_source_or_raise(
    db: Session, kind: str, photo, enhancement_run_id: int | None
) -> ImageSource:
    try:
        return image_source.resolve_image_source(
            db, kind, photo, enhancement_run_id
        )
    except Exception as exc:
        raise _resolution_error_to_http(exc)


def detect_faces(
    db: Session,
    kind: str,
    photo,
    detector: FaceDetector | None = None,
    enhancement_run_id: int | None = None,
) -> FaceDetectionRun:
    """Detect, reusing the current valid result when present.

    enhancement_run_id=None is the normal Phase 3 path; any other
    value explicitly selects that enhancement (never the latest).
    """
    ensure_ready(photo)
    if enhancement_run_id is None:
        # Normal path: preserve the Phase 4 contract exactly --
        # reuse the current run when valid, else create a
        # PROCESSING run FIRST so missing/tampered derived bytes
        # resolve to a FAILED run (HTTP 200), never a bare error.
        current = get_current_run(db, kind, photo)
        if current is not None:
            return current
        ensure_no_active_run(db, kind, photo.id)
        run = _new_run(db, kind, photo, _declared_source(photo))
        try:
            source = image_source.resolve_image_source(
                db, kind, photo, None
            )
        except image_source.SourceResolutionError as exc:
            return _finish_run_failed(db, run, str(exc))
    else:
        # Enhanced path: the explicit selection is validated BEFORE
        # any run is created (unknown selection -> 404, stale or
        # tampered source -> 409), so bad selections never pollute
        # detection history.
        source = _resolve_source_or_raise(
            db, kind, photo, enhancement_run_id
        )
        current = get_current_enhanced_run(
            db, kind, photo, enhancement_run_id
        )
        if current is not None:
            return current
        ensure_no_active_run(db, kind, photo.id)
        run = _new_run(
            db, kind, photo, _declared_source(photo, source)
        )
    if detector is None:
        from app.services.yunet_detector import get_face_detector

        detector = get_face_detector()
    return _execute_run(db, kind, photo, run, detector, source)


def redetect_faces(
    db: Session,
    kind: str,
    photo,
    detector: FaceDetector | None = None,
    enhancement_run_id: int | None = None,
) -> FaceDetectionRun:
    """Explicit re-detect: always create a new run (history kept).

    Enhanced redetect requires the explicit enhancement_run_id.
    """
    ensure_ready(photo)
    ensure_no_active_run(db, kind, photo.id)
    if enhancement_run_id is None:
        run = _new_run(db, kind, photo, _declared_source(photo))
        try:
            source = image_source.resolve_image_source(
                db, kind, photo, None
            )
        except image_source.SourceResolutionError as exc:
            return _finish_run_failed(db, run, str(exc))
    else:
        source = _resolve_source_or_raise(
            db, kind, photo, enhancement_run_id
        )
        run = _new_run(
            db, kind, photo, _declared_source(photo, source)
        )
    if detector is None:
        from app.services.yunet_detector import get_face_detector

        detector = get_face_detector()
    return _execute_run(db, kind, photo, run, detector, source)


def delete_runs_for_photo(db: Session, kind: str, photo_id: int) -> None:
    """Remove all runs (faces cascade) for one photo. Caller commits."""
    for run in _runs_query(db, kind, photo_id).all():
        db.delete(run)
