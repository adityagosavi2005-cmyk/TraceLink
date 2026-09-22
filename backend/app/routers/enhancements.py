from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_trigger_enhancement,
    can_view_case,
    get_current_user,
)
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User
from app.schemas.enhancement import (
    EnhancementRunDetailResponse,
    EnhancementRunResponse,
)
from app.services import enhancement_service, storage

case_router = APIRouter(
    prefix="/cases/{case_id}/photos/{photo_id}/enhancements",
    tags=["Image Enhancement"],
)

sighting_router = APIRouter(
    prefix="/cases/{case_id}/sightings/{sighting_id}/photos/{photo_id}/enhancements",
    tags=["Image Enhancement"],
)


def _get_case_or_404(db: Session, case_id: int) -> Case:
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )
    return case


def _get_case_photo_or_404(
    db: Session, case_id: int, photo_id: int
) -> CasePhoto:
    photo = db.get(CasePhoto, photo_id)
    if photo is None or photo.case_id != case_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Photo not found",
        )
    return photo


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


def _get_sighting_photo_or_404(
    db: Session, case_id: int, sighting_id: int, photo_id: int
) -> SightingPhoto:
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


def _require_trigger(user: User, case: Case, db: Session) -> None:
    if not can_trigger_enhancement(user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run image enhancement",
        )


def _require_view(user: User, case: Case, db: Session) -> None:
    if not can_view_case(user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )


def _detail(run) -> EnhancementRunDetailResponse:
    """Run row plus a presigned view URL for COMPLETE artifacts."""
    view_url: str | None = None
    expires_in: int | None = None
    if run.status.value == "COMPLETE" and run.storage_key:
        view_url, expires_in = storage.presigned_get_url(run.storage_key)
    return EnhancementRunDetailResponse(
        id=run.id,
        case_id=run.case_id,
        case_photo_id=run.case_photo_id,
        sighting_photo_id=run.sighting_photo_id,
        sighting_id=run.sighting_id,
        status=run.status,
        source_derived_sha256=run.source_derived_sha256,
        enhancer_name=run.enhancer_name,
        enhancer_version=run.enhancer_version,
        model_name=run.model_name,
        model_version=run.model_version,
        model_sha256=run.model_sha256,
        parameters=run.parameters,
        output_sha256=run.output_sha256,
        mime_type=run.mime_type,
        width=run.width,
        height=run.height,
        byte_size=run.byte_size,
        error_message=run.error_message,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        view_url=view_url,
        expires_in=expires_in,
    )


@case_router.post(
    "",
    response_model=EnhancementRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def trigger_case_photo_enhancement(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Explicitly start enhancement (never automatic on upload)."""
    case = _get_case_or_404(db, case_id)
    _require_trigger(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    return enhancement_service.trigger_enhancement(
        db,
        "case",
        photo,
        lambda digest: storage.build_enhanced_key(
            case.id, photo.id, digest
        ),
    )


@case_router.get("", response_model=list[EnhancementRunResponse])
def list_case_photo_enhancements(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _require_view(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    return enhancement_service.list_runs(db, "case", photo.id)


@case_router.get(
    "/{run_id}", response_model=EnhancementRunDetailResponse
)
def get_case_photo_enhancement(
    case_id: int,
    photo_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _require_view(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    run = enhancement_service.get_run(db, "case", photo, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement run not found",
        )
    return _detail(run)


@case_router.post(
    "/{run_id}/retry", response_model=EnhancementRunResponse
)
def retry_case_photo_enhancement(
    case_id: int,
    photo_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run a FAILED or stale PROCESSING run (new history row)."""
    case = _get_case_or_404(db, case_id)
    _require_trigger(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    return enhancement_service.retry_enhancement(
        db,
        "case",
        photo,
        run_id,
        lambda digest: storage.build_enhanced_key(
            case.id, photo.id, digest
        ),
    )


@sighting_router.post(
    "",
    response_model=EnhancementRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def trigger_sighting_photo_enhancement(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Explicitly start enhancement (never automatic on upload)."""
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_trigger(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    return enhancement_service.trigger_enhancement(
        db,
        "sighting",
        photo,
        lambda digest: storage.build_sighting_enhanced_key(
            case.id, sighting_id, photo.id, digest
        ),
    )


@sighting_router.get("", response_model=list[EnhancementRunResponse])
def list_sighting_photo_enhancements(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_view(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    return enhancement_service.list_runs(db, "sighting", photo.id)


@sighting_router.get(
    "/{run_id}", response_model=EnhancementRunDetailResponse
)
def get_sighting_photo_enhancement(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_view(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    run = enhancement_service.get_run(db, "sighting", photo, run_id)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Enhancement run not found",
        )
    return _detail(run)


@sighting_router.post(
    "/{run_id}/retry", response_model=EnhancementRunResponse
)
def retry_sighting_photo_enhancement(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run a FAILED or stale PROCESSING run (new history row)."""
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_trigger(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    return enhancement_service.retry_enhancement(
        db,
        "sighting",
        photo,
        run_id,
        lambda digest: storage.build_sighting_enhanced_key(
            case.id, sighting_id, photo.id, digest
        ),
    )
