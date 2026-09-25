from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_edit_case,
    can_view_case,
    get_current_user,
    get_user_organization_ids,
    require_role,
)
from app.models.case import Case
from app.models.sighting import Sighting
from app.models.user import User, UserRole
from app.schemas.sighting import (
    SightingCreate,
    SightingResponse,
    SightingUpdate,
)
from app.services import storage


router = APIRouter(
    prefix="/cases/{case_id}/sightings",
    tags=["Sightings"],
)

# Cross-case index (unpaginated in Phase 2; same scoping as GET /cases).
index_router = APIRouter(
    prefix="/sightings",
    tags=["Sightings"],
)


def _get_case_or_404(db: Session, case_id: int) -> Case:
    case = db.get(Case, case_id)
    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found",
        )
    return case


def _get_sighting_or_404(db: Session, case_id: int, sighting_id: int) -> Sighting:
    sighting = db.get(Sighting, sighting_id)
    if sighting is None or sighting.case_id != case_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Sighting not found",
        )
    return sighting


def _can_edit_sighting(user: User, sighting: Sighting, case: Case, db: Session) -> bool:
    """Reporter-own-row OR the parent case's edit rule."""
    if sighting.reported_by == user.id:
        return True
    return can_edit_case(user, case, db)


@router.post(
    "",
    response_model=SightingResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_sighting(
    case_id: int,
    sighting_data: SightingCreate,
    current_user: User = Depends(
        require_role(
            UserRole.REPORTER,
            UserRole.ORGANIZATION_MEMBER,
            UserRole.ADMIN,
        )
    ),
    db: Session = Depends(get_db),
):
    # REVIEWER cannot create (role gate above, mirroring POST /cases).
    # Organization VIEWERs hold a global member-class role, so they pass
    # the gate and are authorized by case visibility below.
    case = _get_case_or_404(db, case_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to report sightings for this case",
        )

    sighting = Sighting(
        case_id=case.id,
        reported_by=current_user.id,
        sighting_at=sighting_data.sighting_at,
        location_text=sighting_data.location_text,
        latitude=sighting_data.latitude,
        longitude=sighting_data.longitude,
        description=sighting_data.description,
        contact_info=sighting_data.contact_info,
    )
    db.add(sighting)
    db.commit()
    db.refresh(sighting)
    return sighting


@router.get("", response_model=list[SightingResponse])
def list_sightings(
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
    return (
        db.query(Sighting)
        .filter(Sighting.case_id == case.id)
        .order_by(Sighting.sighting_at.desc(), Sighting.created_at.desc())
        .all()
    )


@index_router.get("", response_model=list[SightingResponse])
def list_all_sightings(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Cross-case index scoped exactly like GET /cases (unpaginated)."""
    query = (
        db.query(Sighting)
        .join(Case, Sighting.case_id == Case.id)
        .order_by(Sighting.sighting_at.desc(), Sighting.created_at.desc())
    )
    if current_user.role in (UserRole.ADMIN, UserRole.REVIEWER):
        return query.all()
    org_ids = get_user_organization_ids(db, current_user.id)
    if org_ids:
        query = query.filter(
            (Case.created_by == current_user.id)
            | (Case.organization_id.in_(org_ids))
        )
    else:
        query = query.filter(Case.created_by == current_user.id)
    return query.all()


@router.get("/{sighting_id}", response_model=SightingResponse)
def get_sighting(
    case_id: int,
    sighting_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )
    return _get_sighting_or_404(db, case.id, sighting_id)


@router.patch("/{sighting_id}", response_model=SightingResponse)
def update_sighting(
    case_id: int,
    sighting_id: int,
    sighting_data: SightingUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_edit_sighting(current_user, sighting, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this sighting",
        )

    for field, value in sighting_data.model_dump(exclude_unset=True).items():
        setattr(sighting, field, value)
    db.commit()
    db.refresh(sighting)
    return sighting


@router.delete("/{sighting_id}", status_code=status.HTTP_200_OK)
def delete_sighting(
    case_id: int,
    sighting_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    from app.models.sighting_photo import SightingPhoto

    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_edit_sighting(current_user, sighting, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this sighting",
        )

    photos = (
        db.query(SightingPhoto)
        .filter(SightingPhoto.sighting_id == sighting.id)
        .all()
    )
    # Storage first (originals + reserved derived scope per photo), so
    # a storage failure aborts before any row is touched.
    try:
        for photo in photos:
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
            storage.delete_prefix(
                storage.restored_sighting_photo_prefix(
                    case.id, sighting.id, photo.id
                )
            )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Sighting storage is unavailable; nothing was deleted",
        )

    from app.services import face_detection_service
    from app.services import enhancement_service
    from app.services import face_restoration_service

    for photo in photos:
        face_detection_service.delete_runs_for_photo(
            db, "sighting", photo.id
        )
        enhancement_service.delete_runs_for_photo(
            db, "sighting", photo.id
        )
    for photo in photos:
        db.delete(photo)
    for _photo in photos:
        face_restoration_service.delete_runs_for_photo(
            db, "sighting", _photo.id
        )  # per-photo restoration history
    db.delete(sighting)
    db.commit()
    return {"message": "Sighting deleted successfully"}
