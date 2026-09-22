import hashlib

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_edit_case,
    can_view_case,
    get_current_user,
)
from app.models.case import Case
from app.models.case_photo import PhotoStatus
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User
from app.schemas.sighting_photo import SightingPhotoResponse
from app.services import storage
from app.services.image_validation import validate_image as _validate_image

router = APIRouter(
    prefix="/cases/{case_id}/sightings/{sighting_id}/photos",
    tags=["Sighting Photos"],
)


def _get_case_or_404(db: Session, case_id: int) -> Case:
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )
    return case


def _get_sighting_or_404(
    db: Session, case_id: int, sighting_id: int
) -> Sighting:
    sighting = db.get(Sighting, sighting_id)
    if sighting is None or sighting.case_id != case_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sighting not found",
        )
    return sighting


def _get_photo_or_404(
    db: Session, case_id: int, sighting_id: int, photo_id: int
) -> SightingPhoto:
    """Triple binding: photo.case_id == sighting.case_id == URL case_id."""
    photo = db.get(SightingPhoto, photo_id)
    if (
        photo is None
        or photo.sighting_id != sighting_id
        or photo.case_id != case_id
    ):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found",
        )
    return photo


def _can_contribute(
    user: User, sighting: Sighting, case: Case, db: Session
) -> bool:
    """Parent case edit rule, or the sighting's own reporter."""
    if sighting.reported_by == user.id:
        return True
    return can_edit_case(user, case, db)


def _face_summary(
    db: Session, photo: SightingPhoto
) -> tuple[str, int]:
    """Current face-detection status/count (NOT_RUN when none)."""
    from app.services import face_detection_service

    run = face_detection_service.get_current_run(db, "sighting", photo)
    if run is None:
        return "NOT_RUN", 0
    return run.status.value, run.face_count


def _respond(photo: SightingPhoto, db: Session) -> SightingPhotoResponse:
    view_url, expires_in = storage.presigned_get_url(
        photo.storage_key_original
    )
    derived_view_url: str | None = None
    derived_expires_in: int | None = None
    if (
        photo.processing_status == PhotoStatus.READY
        and photo.storage_key_derived
    ):
        derived_view_url, derived_expires_in = storage.presigned_get_url(
            photo.storage_key_derived
        )
    face_status, face_count = _face_summary(db, photo)
    return SightingPhotoResponse(
        id=photo.id,
        sighting_id=photo.sighting_id,
        case_id=photo.case_id,
        uploaded_by=photo.uploaded_by,
        mime_type=photo.mime_type,
        byte_size=photo.byte_size,
        width=photo.width,
        height=photo.height,
        sha256=photo.sha256,
        processing_status=photo.processing_status,
        processing_error=photo.processing_error,
        processing_version=photo.processing_version,
        derived_sha256=photo.derived_sha256,
        derived_width=photo.derived_width,
        derived_height=photo.derived_height,
        derived_mime_type=photo.derived_mime_type,
        created_at=photo.created_at,
        updated_at=photo.updated_at,
        face_detection_status=face_status,
        face_count=face_count,
        view_url=view_url,
        expires_in=expires_in,
        derived_view_url=derived_view_url,
        derived_expires_in=derived_expires_in,
    )


@router.post(
    "",
    response_model=SightingPhotoResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_sighting_photo(
    case_id: int,
    sighting_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.core.config import settings

    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_contribute(current_user, sighting, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to add photos to this sighting",
        )

    data = await file.read(settings.max_photo_size_bytes + 1)
    mime_type, ext, width, height = _validate_image(data)
    digest = hashlib.sha256(data).hexdigest()

    # Flush first so the storage key carries the real photo id.
    photo = SightingPhoto(
        sighting_id=sighting.id,
        case_id=case.id,
        uploaded_by=current_user.id,
        storage_key_original="pending",
        mime_type=mime_type,
        byte_size=len(data),
        width=width,
        height=height,
        sha256=digest,
        processing_status=PhotoStatus.UPLOADED,
    )
    db.add(photo)
    db.flush()

    key = storage.build_sighting_original_key(
        case.id, sighting.id, photo.id, digest, ext
    )
    try:
        storage.put_original(key, data, mime_type)
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Image storage is unavailable; nothing was saved",
        )

    photo.storage_key_original = key
    db.commit()
    db.refresh(photo)

    # PHASE B — derived processing (synchronous). The original is
    # committed above; a derived failure commits FAILED without
    # touching the valid original.
    from app.services import photo_processing

    photo_processing.run_upload_processing(
        db,
        photo,
        data,
        lambda digest: storage.build_sighting_derived_key(
            case.id, sighting.id, photo.id, digest
        ),
    )
    return _respond(photo, db)


@router.get("", response_model=list[SightingPhotoResponse])
def list_sighting_photos(
    case_id: int,
    sighting_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )
    photos = (
        db.query(SightingPhoto)
        .filter(SightingPhoto.sighting_id == sighting.id)
        .order_by(SightingPhoto.created_at.asc())
        .all()
    )
    return [_respond(photo, db) for photo in photos]


@router.get("/{photo_id}", response_model=SightingPhotoResponse)
def get_sighting_photo(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )
    photo = _get_photo_or_404(db, case.id, sighting_id, photo_id)
    return _respond(photo, db)


@router.post("/{photo_id}/retry", response_model=SightingPhotoResponse)
def retry_sighting_photo_processing(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run deterministic processing (UPLOADED/FAILED/stale)."""
    from app.services import photo_processing

    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_contribute(current_user, sighting, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to process photos in this sighting",
        )
    photo = _get_photo_or_404(db, case.id, sighting.id, photo_id)
    photo_processing.begin_retry(db, photo)
    photo_processing.run_retry_processing(
        db,
        photo,
        lambda digest: storage.build_sighting_derived_key(
            case.id, sighting.id, photo.id, digest
        ),
    )
    return _respond(photo, db)


@router.delete("/{photo_id}", status_code=status.HTTP_200_OK)
def delete_sighting_photo(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_contribute(current_user, sighting, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete photos from this sighting",
        )
    photo = _get_photo_or_404(db, case.id, sighting.id, photo_id)
    # Storage first (originals + derived + enhanced scopes), so a
    # storage failure aborts before any row is touched.
    try:
        storage.delete_prefix(
            storage.sighting_photo_prefix(
                case.id, sighting.id, photo.id
            )
        )
        storage.delete_prefix(
            storage.derived_sighting_photo_prefix(
                case.id, sighting.id, photo.id
            )
        )
        storage.delete_prefix(
            storage.enhanced_sighting_photo_prefix(
                case.id, sighting.id, photo.id
            )
        )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Photo storage is unavailable; nothing was deleted",
        )
    from app.services import face_detection_service
    from app.services import enhancement_service

    face_detection_service.delete_runs_for_photo(db, "sighting", photo.id)
    enhancement_service.delete_runs_for_photo(db, "sighting", photo.id)
    db.delete(photo)
    db.commit()
    return {"message": "Photo deleted successfully"}
