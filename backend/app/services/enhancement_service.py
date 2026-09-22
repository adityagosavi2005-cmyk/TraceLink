"""Image-enhancement orchestration (Phase 7).

Shared by the case-photo and sighting-photo enhancement routers so
both photo types enhance identically. Runs synchronously inside the
triggering request; no queues or workers (Phase 7 v1).

Flow per explicit trigger:
    READY photo -> PROCESSING run -> derived bytes -> SHA verify ->
    enhancer -> validate output -> SHA-256 -> enhanced/ object ->
    COMPLETE run. Any failure -> FAILED run (the photo's Phase 3
    status and the original/derived artifacts are never touched).

History is preserved: every trigger appends a new run, even for
identical model/parameters. At most one ACTIVE (fresh PROCESSING)
run per photo; FAILED and stale PROCESSING runs (600-second
philosophy, shared with Phase 3/4) may be retried explicitly, which
likewise appends a new run.

This module performs persistence and storage reads on the caller's
behalf, but never authorization and never key construction: routers
enforce permissions and build enhanced keys via the storage service.
"""

import hashlib
import json
from datetime import datetime, timezone

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.case_photo import PhotoStatus
from app.models.enhancement import EnhancementRun, EnhancementStatus
from app.services import image_source, storage
from app.services.enhancement import EnhancementError, ImageEnhancer
from app.services.preprocessing import (
    STALE_PROCESSING_THRESHOLD_SECONDS,
)


def active_enhancer_identity() -> tuple[str, str, str, str]:
    """Currently valid (enhancer, version, model, model version)."""
    from app.services.realesrgan_adapter import RealESRGANAdapter

    adapter = RealESRGANAdapter()
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
        EnhancementRun.case_photo_id
        if kind == "case"
        else EnhancementRun.sighting_photo_id
    )
    return db.query(EnhancementRun).filter(column == photo_id)


def list_runs(db: Session, kind: str, photo_id: int) -> list[EnhancementRun]:
    """All runs for one photo, newest first (history preserved)."""
    return (
        _runs_query(db, kind, photo_id)
        .order_by(EnhancementRun.id.desc())
        .all()
    )


def get_run(
    db: Session, kind: str, photo, run_id: int
) -> EnhancementRun | None:
    """One run scoped to its photo (None when it belongs elsewhere)."""
    run = db.get(EnhancementRun, run_id)
    if run is None:
        return None
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    if (
        run.case_id != photo.case_id
        or run.case_photo_id != case_photo_id
        or run.sighting_photo_id != sighting_photo_id
    ):
        return None
    return run


def ensure_ready(photo) -> None:
    """Reject enhancement unless Phase 3 produced a derived image."""
    if (
        photo.processing_status != PhotoStatus.READY
        or not photo.storage_key_derived
        or not photo.derived_sha256
        or photo.derived_width is None
        or photo.derived_height is None
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Photo is not ready for image enhancement",
        )
    pixels = photo.derived_width * photo.derived_height
    if pixels > settings.ENHANCEMENT_MAX_INPUT_PIXELS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Derived image is too large for image enhancement",
        )


def is_run_stale(run: EnhancementRun, now=None) -> bool:
    """True when a PROCESSING run may be safely recovered."""
    if run.status != EnhancementStatus.PROCESSING:
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


def ensure_no_active_run(db: Session, kind: str, photo_id: int) -> None:
    """Reject a trigger that would race an in-progress run."""
    active = (
        _runs_query(db, kind, photo_id)
        .filter(EnhancementRun.status == EnhancementStatus.PROCESSING)
        .order_by(EnhancementRun.id.desc())
        .first()
    )
    if active is None or is_run_stale(active):
        return
    raise HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail="Image enhancement is already running; try again later",
    )


def _finish_run_failed(
    db: Session, run: EnhancementRun, message: str
) -> EnhancementRun:
    db.rollback()
    run.status = EnhancementStatus.FAILED
    run.finished_at = datetime.now(timezone.utc)
    run.error_message = message
    db.commit()
    db.refresh(run)
    return run


def _new_run(
    db: Session,
    kind: str,
    photo,
    identity: tuple[str, str, str, str],
    model_sha256: str,
    parameters: str,
) -> EnhancementRun:
    name, version, model_name, model_version = identity
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    run = EnhancementRun(
        case_id=photo.case_id,
        case_photo_id=case_photo_id,
        sighting_photo_id=sighting_photo_id,
        sighting_id=getattr(photo, "sighting_id", None),
        status=EnhancementStatus.PROCESSING,
        source_derived_sha256=photo.derived_sha256,
        enhancer_name=name,
        enhancer_version=version,
        model_name=model_name,
        model_version=model_version,
        model_sha256=model_sha256,
        parameters=parameters,
        mime_type="image/jpeg",
        started_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def trigger_enhancement(
    db: Session,
    kind: str,
    photo,
    enhanced_key_for_sha,
    enhancer: ImageEnhancer | None = None,
) -> EnhancementRun:
    """Start one enhancement execution (always a new history row)."""
    ensure_ready(photo)
    ensure_no_active_run(db, kind, photo.id)
    if enhancer is None:
        from app.services.realesrgan_adapter import get_image_enhancer

        enhancer = get_image_enhancer()
    identity = (
        enhancer.name,
        enhancer.version,
        enhancer.model_name,
        enhancer.model_version,
    )
    parameters = json.dumps({"scale": enhancer.scale})
    try:
        model_sha256 = enhancer.model_sha256
    except EnhancementError as exc:
        # Weights/model unavailable before any run exists: record
        # the attempt as FAILED history so the failure is visible.
        return _record_stillborn(db, kind, photo, identity, parameters, str(exc))
    run = _new_run(db, kind, photo, identity, model_sha256, parameters)
    return _execute_run(db, photo, run, enhancer, enhanced_key_for_sha)


def _record_stillborn(
    db: Session,
    kind: str,
    photo,
    identity: tuple[str, str, str, str],
    parameters: str,
    message: str,
) -> EnhancementRun:
    """Persist a FAILED run when the model cannot even load."""
    name, version, model_name, model_version = identity
    case_photo_id, sighting_photo_id = _photo_parent(photo, kind)
    run = EnhancementRun(
        case_id=photo.case_id,
        case_photo_id=case_photo_id,
        sighting_photo_id=sighting_photo_id,
        sighting_id=getattr(photo, "sighting_id", None),
        status=EnhancementStatus.FAILED,
        source_derived_sha256=photo.derived_sha256,
        enhancer_name=name,
        enhancer_version=version,
        model_name=model_name,
        model_version=model_version,
        # Provenance unknown: the weights file could not be read.
        model_sha256="0" * 64,
        parameters=parameters,
        mime_type="image/jpeg",
        error_message=message,
        started_at=datetime.now(timezone.utc),
        finished_at=datetime.now(timezone.utc),
    )
    db.add(run)
    db.commit()
    db.refresh(run)
    return run


def _execute_run(
    db: Session, photo, run: EnhancementRun, enhancer, enhanced_key_for_sha
) -> EnhancementRun:
    """Read derived bytes, run the enhancer, persist the artifact."""
    try:
        derived_bytes = image_source.read_derived_bytes(photo)
    except image_source.SourceResolutionError as exc:
        return _finish_run_failed(db, run, str(exc))
    if run.source_derived_sha256 != photo.derived_sha256:
        return _finish_run_failed(
            db,
            run,
            "Derived image changed during processing; "
            "reprocess the photo",
        )
    try:
        enhanced = enhancer.enhance(derived_bytes)
    except EnhancementError as exc:
        return _finish_run_failed(db, run, str(exc))
    except Exception:
        return _finish_run_failed(
            db, run, "Image enhancement failed; try again later"
        )
    if (
        not enhanced.image_bytes
        or enhanced.width <= 0
        or enhanced.height <= 0
    ):
        return _finish_run_failed(
            db, run, "Image enhancement failed; try again later"
        )
    output_sha = hashlib.sha256(enhanced.image_bytes).hexdigest()
    storage_key = enhanced_key_for_sha(output_sha)
    try:
        storage.put_enhanced(
            storage_key, enhanced.image_bytes, enhanced.mime_type
        )
    except Exception:
        return _finish_run_failed(
            db, run, "Enhanced image storage is unavailable; retry later"
        )
    run.output_sha256 = output_sha
    run.storage_key = storage_key
    run.mime_type = enhanced.mime_type
    run.width = enhanced.width
    run.height = enhanced.height
    run.byte_size = len(enhanced.image_bytes)
    run.status = EnhancementStatus.COMPLETE
    run.finished_at = datetime.now(timezone.utc)
    run.error_message = None
    try:
        db.commit()
    except Exception:
        # Storage succeeded but persistence failed: remove the
        # orphan object (best effort), then record FAILED so the
        # run row reflects reality instead of dangling.
        try:
            storage.delete_prefix(storage_key)
        except Exception:
            pass
        return _finish_run_failed(
            db, run, "Image enhancement failed; try again later"
        )
    db.refresh(run)
    return run


def retry_enhancement(
    db: Session,
    kind: str,
    photo,
    run_id: int,
    enhanced_key_for_sha,
    enhancer: ImageEnhancer | None = None,
) -> EnhancementRun:
    """Explicit re-run of a FAILED or stale PROCESSING run.

    Always appends a NEW run (history preserved, never
    deduplicated). Fresh PROCESSING runs conflict; COMPLETE runs
    cannot be "retried" -- trigger a fresh enhancement instead.
    """
    ensure_ready(photo)
    existing = get_run(db, kind, photo, run_id)
    if existing is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement run not found",
        )
    if existing.status == EnhancementStatus.COMPLETE:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Enhancement run already completed; "
            "start a new enhancement instead",
        )
    if existing.status == EnhancementStatus.PROCESSING and not is_run_stale(
        existing
    ):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Image enhancement is already running; "
            "try again later",
        )
    ensure_no_active_run(db, kind, photo.id)
    if enhancer is None:
        from app.services.realesrgan_adapter import get_image_enhancer

        enhancer = get_image_enhancer()
    identity = (
        enhancer.name,
        enhancer.version,
        enhancer.model_name,
        enhancer.model_version,
    )
    parameters = json.dumps({"scale": enhancer.scale})
    try:
        model_sha256 = enhancer.model_sha256
    except EnhancementError as exc:
        return _record_stillborn(db, kind, photo, identity, parameters, str(exc))
    run = _new_run(db, kind, photo, identity, model_sha256, parameters)
    return _execute_run(db, photo, run, enhancer, enhanced_key_for_sha)


def delete_runs_for_photo(db: Session, kind: str, photo_id: int) -> None:
    """Remove all runs for one photo. Caller commits."""
    for run in _runs_query(db, kind, photo_id).all():
        db.delete(run)
