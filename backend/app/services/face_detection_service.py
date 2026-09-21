"""Face-detection orchestration (Phase 4).

Shared by the case-photo and sighting-photo face routers so both
photo types detect identically. Runs synchronously inside the
triggering request; no queues or workers.

Flow per explicit trigger:
    READY photo -> reuse current COMPLETE run, else
    PROCESSING run -> derived bytes -> SHA verify -> detector ->
    normalize/clamp/filter -> face rows -> COMPLETE run.
    Any failure -> FAILED run (the photo's Phase 3 status is never
    touched).

This module performs persistence and storage reads on the caller's
behalf, but never authorization and never key construction: routers
enforce permissions; storage keys come from the photo row.
"""

import hashlib
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.case_photo import PhotoStatus
from app.models.face_detection import (
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.services import storage
from app.services.face_detector import DetectedFace, FaceDetector
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
    """Newest COMPLETE run matching active config + derived SHA."""
    name, version, threshold = active_detector_identity()
    return (
        _runs_query(db, kind, photo.id)
        .filter(
            FaceDetectionRun.status == FaceDetectionStatus.COMPLETE,
            FaceDetectionRun.detector_name == name,
            FaceDetectionRun.detector_version == version,
            FaceDetectionRun.threshold == threshold,
            FaceDetectionRun.source_derived_sha == photo.derived_sha256,
        )
        .order_by(FaceDetectionRun.id.desc())
        .first()
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

    Coordinates are kept in derived-image pixels. Zero-area or
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


def _new_run(db: Session, kind: str, photo) -> FaceDetectionRun:
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
        source_derived_sha=photo.derived_sha256,
        source_derived_width=photo.derived_width,
        source_derived_height=photo.derived_height,
        face_count=0,
        error=None,
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _execute_run(
    db: Session, kind: str, photo, run: FaceDetectionRun, detector
) -> FaceDetectionRun:
    """Read derived bytes, run the detector, persist faces."""
    try:
        derived_bytes = storage.get_derived_bytes(
            photo.storage_key_derived
        )
    except Exception:
        return _finish_run_failed(
            db, run, "Derived image is unavailable; try again later"
        )
    if hashlib.sha256(derived_bytes).hexdigest() != photo.derived_sha256:
        return _finish_run_failed(
            db,
            run,
            "Derived image changed during processing; reprocess the photo",
        )
    try:
        detected = detector.detect(
            derived_bytes, photo.derived_width, photo.derived_height
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
        detected or [], photo.derived_width, photo.derived_height
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
                frame_width=photo.derived_width,
                frame_height=photo.derived_height,
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


def detect_faces(
    db: Session,
    kind: str,
    photo,
    detector: FaceDetector | None = None,
) -> FaceDetectionRun:
    """Normal detect: reuse the current COMPLETE run when valid."""
    ensure_ready(photo)
    current = get_current_run(db, kind, photo)
    if current is not None:
        return current
    ensure_no_active_run(db, kind, photo.id)
    if detector is None:
        from app.services.yunet_detector import get_face_detector

        detector = get_face_detector()
    run = _new_run(db, kind, photo)
    return _execute_run(db, kind, photo, run, detector)


def redetect_faces(
    db: Session,
    kind: str,
    photo,
    detector: FaceDetector | None = None,
) -> FaceDetectionRun:
    """Explicit re-detect: always create a new run (history kept)."""
    ensure_ready(photo)
    ensure_no_active_run(db, kind, photo.id)
    if detector is None:
        from app.services.yunet_detector import get_face_detector

        detector = get_face_detector()
    run = _new_run(db, kind, photo)
    return _execute_run(db, kind, photo, run, detector)


def delete_runs_for_photo(db: Session, kind: str, photo_id: int) -> None:
    """Remove all runs (faces cascade) for one photo. Caller commits."""
    for run in _runs_query(db, kind, photo_id).all():
        db.delete(run)
