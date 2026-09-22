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
from app.models.case_photo import CasePhoto, PhotoStatus
from app.models.user import User
from app.schemas.case_photo import CasePhotoResponse
from app.services import storage
from app.services.image_validation import validate_image as _validate_image

router = APIRouter(
    prefix="/cases/{case_id}/photos",
    tags=["Case Photos"],
)


def _get_case_or_404(db: Session, case_id: int) -> Case:
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )
    return case


def _face_summary(
    db: Session, photo: CasePhoto
) -> tuple[str, int]:
    """Current face-detection status/count (NOT_RUN when none)."""
    from app.services import face_detection_service

    run = face_detection_service.get_current_run(db, "case", photo)
    if run is None:
        return "NOT_RUN", 0
    return run.status.value, run.face_count


def _respond(photo: CasePhoto, db: Session) -> CasePhotoResponse:
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
    return CasePhotoResponse(
        id=photo.id,
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
    response_model=CasePhotoResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_case_photo(
    case_id: int,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.core.config import settings

    case = _get_case_or_404(db, case_id)
    if not can_edit_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to add photos to this case",
        )

    data = await file.read(settings.max_photo_size_bytes + 1)
    mime_type, ext, width, height = _validate_image(data)
    digest = hashlib.sha256(data).hexdigest()

    # Flush first so the storage key carries the real photo id.
    photo = CasePhoto(
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

    key = storage.build_original_key(case.id, photo.id, digest, ext)
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
        lambda digest: storage.build_derived_key(
            case.id, photo.id, digest
        ),
    )
    return _respond(photo, db)


@router.get("", response_model=list[CasePhotoResponse])
def list_case_photos(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )
    photos = (
        db.query(CasePhoto)
        .filter(CasePhoto.case_id == case.id)
        .order_by(CasePhoto.created_at.asc())
        .all()
    )
    return [_respond(photo, db) for photo in photos]


@router.get("/{photo_id}", response_model=CasePhotoResponse)
def get_case_photo(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )
    photo = db.get(CasePhoto, photo_id)
    if photo is None or photo.case_id != case.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found",
        )
    return _respond(photo, db)


@router.post("/{photo_id}/retry", response_model=CasePhotoResponse)
def retry_case_photo_processing(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run deterministic processing (UPLOADED/FAILED/stale)."""
    from app.services import photo_processing

    case = _get_case_or_404(db, case_id)
    if not can_edit_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to process photos in this case",
        )
    photo = db.get(CasePhoto, photo_id)
    if photo is None or photo.case_id != case.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found",
        )
    photo_processing.begin_retry(db, photo)
    photo_processing.run_retry_processing(
        db,
        photo,
        lambda digest: storage.build_derived_key(
            case.id, photo.id, digest
        ),
    )
    return _respond(photo, db)


@router.delete("/{photo_id}", status_code=status.HTTP_200_OK)
def delete_case_photo(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    if not can_edit_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete photos from this case",
        )
    photo = db.get(CasePhoto, photo_id)
    if photo is None or photo.case_id != case.id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found",
        )
    # Storage first (originals + derived + enhanced scopes), so a
    # storage failure aborts before any row is touched.
    try:
        storage.delete_prefix(storage.photo_prefix(case.id, photo.id))
        storage.delete_prefix(
            storage.derived_photo_prefix(case.id, photo.id)
        )
        storage.delete_prefix(
            storage.enhanced_photo_prefix(case.id, photo.id)
        )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Photo storage is unavailable; nothing was deleted",
        )
    from app.services import face_detection_service
    from app.services import enhancement_service

    face_detection_service.delete_runs_for_photo(db, "case", photo.id)
    enhancement_service.delete_runs_for_photo(db, "case", photo.id)
    db.delete(photo)
    db.commit()
    return {"message": "Photo deleted successfully"}
