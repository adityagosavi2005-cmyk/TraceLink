"""Centralized image-source resolution (Phase 7).

Phase 4, Phase 5, and Phase 6 must never each invent their own
enhanced-source logic: every consumer resolves through this module
so SHA verification and enhancement-validity live in exactly one
place.

Two source types:

DERIVED  -> the photo's current Phase 3 artifact
            (photo.storage_key_derived).
ENHANCED -> the output of one explicitly selected COMPLETE
            EnhancementRun. Valid only when:
              1. the run exists and belongs to this photo,
              2. status == COMPLETE,
              3. the output object exists in storage,
              4. SHA(stored bytes) == output_sha256,
              5. run.source_derived_sha256 == photo.derived_sha256.

There is no implicit "latest enhancement": callers always pass an
explicit enhancement_run_id (or a run row that carries one).
"""

import hashlib
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.models.enhancement import (
    EnhancementRun,
    EnhancementStatus,
    ImageSourceType,
)
from app.services import storage


class SourceResolutionError(Exception):
    """The requested image source is missing, stale, or tampered.

    Carries only a safe, user-presentable message. Each caller maps
    it to its own error channel (HTTP 409/404, RepresentationError,
    or silent candidate exclusion).
    """


@dataclass(frozen=True)
class ImageSource:
    """Resolved bytes plus the provenance a run row must record."""

    image_bytes: bytes
    width: int
    height: int
    source_type: ImageSourceType
    # Digest of image_bytes (derived SHA for DERIVED, output SHA
    # for ENHANCED).
    source_sha256: str
    # Phase 3 grandparent SHA (always the photo's derived SHA).
    source_derived_sha: str
    enhancement_run: EnhancementRun | None


def _photo_parent_ids(kind: str, photo) -> tuple[int | None, int | None]:
    if kind == "case":
        return photo.id, None
    return None, photo.id


def load_valid_enhancement(
    db: Session, kind: str, photo, enhancement_run_id: int
) -> EnhancementRun:
    """Return the run when conditions 1, 2, and 5 hold.

    Existence/ownership failures read as "not found" (no
    cross-photo probing); wrong-state and stale-source failures
    read as conflicts.
    """
    run = db.get(EnhancementRun, enhancement_run_id)
    case_photo_id, sighting_photo_id = _photo_parent_ids(kind, photo)
    if (
        run is None
        or run.case_id != photo.case_id
        or run.case_photo_id != case_photo_id
        or run.sighting_photo_id != sighting_photo_id
    ):
        raise SourceResolutionError("Enhancement run not found")
    if run.status != EnhancementStatus.COMPLETE:
        raise SourceResolutionError(
            "Enhancement run is not complete"
        )
    if run.source_derived_sha256 != photo.derived_sha256:
        raise SourceResolutionError(
            "Enhancement run is stale for the current derived image"
        )
    return run


def read_enhanced_bytes(run: EnhancementRun) -> bytes:
    """Return output bytes when conditions 3 and 4 hold."""
    if not run.storage_key or not run.output_sha256:
        raise SourceResolutionError(
            "Enhancement run is not complete"
        )
    try:
        data = storage.get_enhanced_bytes(run.storage_key)
    except Exception as exc:
        raise SourceResolutionError(
            "Enhanced image is unavailable; try again later"
        ) from exc
    if hashlib.sha256(data).hexdigest() != run.output_sha256:
        raise SourceResolutionError(
            "Enhanced image changed during processing"
        )
    return data


def read_derived_bytes(photo) -> bytes:
    """Return derived bytes verified against the photo row."""
    if not photo.storage_key_derived or not photo.derived_sha256:
        raise SourceResolutionError(
            "Photo is not ready for face detection"
        )
    try:
        data = storage.get_derived_bytes(photo.storage_key_derived)
    except Exception as exc:
        raise SourceResolutionError(
            "Derived image is unavailable; try again later"
        ) from exc
    if hashlib.sha256(data).hexdigest() != photo.derived_sha256:
        raise SourceResolutionError(
            "Derived image changed during processing; "
            "reprocess the photo"
        )
    return data


def resolve_image_source(
    db: Session, kind: str, photo, enhancement_run_id: int | None = None
) -> ImageSource:
    """Resolve the exact bytes a new detection run must consume.

    enhancement_run_id=None selects the normal Phase 3 source; any
    other value selects that explicit enhancement (never the latest).
    """
    if enhancement_run_id is None:
        data = read_derived_bytes(photo)
        return ImageSource(
            image_bytes=data,
            width=photo.derived_width,
            height=photo.derived_height,
            source_type=ImageSourceType.DERIVED,
            source_sha256=photo.derived_sha256,
            source_derived_sha=photo.derived_sha256,
            enhancement_run=None,
        )
    run = load_valid_enhancement(db, kind, photo, enhancement_run_id)
    data = read_enhanced_bytes(run)
    return ImageSource(
        image_bytes=data,
        width=run.width,
        height=run.height,
        source_type=ImageSourceType.ENHANCED,
        source_sha256=run.output_sha256,
        source_derived_sha=run.source_derived_sha256,
        enhancement_run=run,
    )


def resolve_run_source(db: Session, run, photo) -> ImageSource:
    """Resolve the bytes an EXISTING detection run consumed.

    Used by representation (and similarity validation): the run row
    declares its source, and this re-verifies the artifacts still
    match. Raises SourceResolutionError for stale/tampered sources.
    """
    if run.source_type == ImageSourceType.ENHANCED:
        if run.enhancement_run_id is None:
            raise SourceResolutionError(
                "Enhanced detection has no enhancement run"
            )
        enhancement = load_valid_enhancement(
            db,
            "case" if run.case_photo_id is not None else "sighting",
            photo,
            run.enhancement_run_id,
        )
        if run.source_sha256 != enhancement.output_sha256:
            raise SourceResolutionError(
                "Enhanced image changed during processing"
            )
        data = read_enhanced_bytes(enhancement)
        return ImageSource(
            image_bytes=data,
            width=run.source_width,
            height=run.source_height,
            source_type=ImageSourceType.ENHANCED,
            source_sha256=enhancement.output_sha256,
            source_derived_sha=enhancement.source_derived_sha256,
            enhancement_run=enhancement,
        )
    data = read_derived_bytes(photo)
    if (
        run.source_derived_sha != photo.derived_sha256
        or run.source_sha256 != photo.derived_sha256
    ):
        raise SourceResolutionError(
            "Face detection is not current for the derived image"
        )
    return ImageSource(
        image_bytes=data,
        width=run.source_width,
        height=run.source_height,
        source_type=ImageSourceType.DERIVED,
        source_sha256=photo.derived_sha256,
        source_derived_sha=photo.derived_sha256,
        enhancement_run=None,
    )
