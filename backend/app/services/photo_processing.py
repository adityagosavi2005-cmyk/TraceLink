"""Derived-processing orchestration (Phase 3).

Owns the photo-row state machine shared by the case-photo and
sighting-photo routers so both photo types transition identically:

    UPLOADED -> READY   (synchronous upload path, no PROCESSING persisted)
    UPLOADED -> FAILED
    UPLOADED/FAILED/stale-PROCESSING -> PROCESSING -> READY/FAILED (retry)

This module performs persistence and storage calls on the caller's
behalf, but never authorization and never key construction: routers
enforce permissions and build derived keys via the storage service.
"""

from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.models.case_photo import PhotoStatus
from app.services import preprocessing, storage
from app.services.preprocessing import (
    STALE_PROCESSING_THRESHOLD_SECONDS,
)


def is_processing_stale(photo, now: datetime | None = None) -> bool:
    """True when a PROCESSING row may be safely recovered."""
    if photo.processing_status != PhotoStatus.PROCESSING:
        return False
    if photo.processing_started_at is None:
        return True
    now = now or datetime.now(timezone.utc)
    started = photo.processing_started_at
    if started.tzinfo is None:
        started = started.replace(tzinfo=timezone.utc)
    return (now - started).total_seconds() > STALE_PROCESSING_THRESHOLD_SECONDS


def ensure_retry_allowed(photo) -> None:
    """Reject a retry that would race an in-progress attempt."""
    if photo.processing_status == PhotoStatus.PROCESSING and not (
        is_processing_stale(photo)
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Photo is already being processed; try again later",
        )


def _finish_success(photo, derived_key: str, meta: tuple) -> None:
    """Apply READY metadata in memory (caller commits)."""
    _, digest, width, height, mime_type, version = meta
    photo.storage_key_derived = derived_key
    photo.processing_status = PhotoStatus.READY
    photo.processing_started_at = None
    photo.processing_error = None
    photo.processing_version = version
    photo.derived_sha256 = digest
    photo.derived_width = width
    photo.derived_height = height
    photo.derived_mime_type = mime_type


def _finish_failure(photo, message: str) -> None:
    """Apply FAILED metadata in memory (caller commits)."""
    photo.processing_status = PhotoStatus.FAILED
    photo.processing_started_at = None
    photo.processing_error = message


def run_upload_processing(
    db: Session,
    photo,
    original_bytes: bytes,
    derived_key_for_sha,
) -> None:
    """Phase B of upload: derive from in-memory bytes, commit outcome.

    The original row is already committed by the caller. A derived
    failure never removes the valid original: it commits FAILED.
    """
    try:
        meta = preprocessing.preprocess(original_bytes)
    except preprocessing.PreprocessingError as exc:
        _finish_failure(photo, str(exc))
        db.commit()
        db.refresh(photo)
        return
    derived_bytes, digest = meta[0], meta[1]
    try:
        storage.put_derived(
            derived_key_for_sha(digest), derived_bytes, meta[4]
        )
    except Exception:
        _finish_failure(
            photo, "Derived image storage is unavailable; retry later"
        )
        db.commit()
        db.refresh(photo)
        return
    _finish_success(photo, derived_key_for_sha(digest), meta)
    db.commit()
    db.refresh(photo)


def run_retry_processing(
    db: Session,
    photo,
    derived_key_for_sha,
) -> None:
    """Explicit retry: PROCESSING is persisted, then resolved.

    Assumes the caller already enforced authorization and committed the
    PROCESSING marker via begin_retry(). Original bytes are re-read
    from immutable storage so historical UPLOADED rows use the exact
    same pipeline as new uploads.
    """
    try:
        original_bytes = storage.get_original_bytes(
            photo.storage_key_original
        )
    except Exception:
        _finish_failure(
            photo, "Original image is unavailable; retry later"
        )
        db.commit()
        db.refresh(photo)
        return
    run_upload_processing(db, photo, original_bytes, derived_key_for_sha)


def begin_retry(db: Session, photo) -> None:
    """Mark PROCESSING (+ timestamp, error cleared) and commit."""
    ensure_retry_allowed(photo)
    photo.processing_status = PhotoStatus.PROCESSING
    photo.processing_started_at = datetime.now(timezone.utc)
    photo.processing_error = None
    db.commit()
    db.refresh(photo)
