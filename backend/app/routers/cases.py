from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    get_current_user,
    require_role,
    can_view_case,
    can_edit_case,
    get_user_organization_ids,
    require_case_organization_access,
)
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User, UserRole
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate
from app.services import storage


router = APIRouter(
    prefix="/cases",
    tags=["Cases"]
)

@router.post(
    "",
    response_model=CaseResponse,
    status_code=status.HTTP_201_CREATED
)
def create_case(
    case_data: CaseCreate,
    current_user: User = Depends(
        require_role(
            UserRole.REPORTER,
            UserRole.ORGANIZATION_MEMBER,
            UserRole.ADMIN
        )
    ),
    db: Session = Depends(get_db)
):
    # Personal cases (organization_id=None) always allowed.
    # Organization cases require membership; enforced centrally.
    require_case_organization_access(db, current_user, case_data.organization_id)

    new_case = Case(
        title=case_data.title,
        description=case_data.description,
        created_by=current_user.id,
        organization_id=case_data.organization_id,
        full_name=case_data.full_name,
        age_years=case_data.age_years,
        age_estimate_note=case_data.age_estimate_note,
        last_seen_at=case_data.last_seen_at,
        last_seen_location=case_data.last_seen_location,
        clothing_description=case_data.clothing_description,
        distinguishing_marks=case_data.distinguishing_marks,
        contact_info=case_data.contact_info,
    )

    db.add(new_case)
    db.commit()
    db.refresh(new_case)

    return new_case

@router.get(
    "",
    response_model=list[CaseResponse]
)
def get_cases(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    if current_user.role in (UserRole.ADMIN, UserRole.REVIEWER):
        cases = (
            db.query(Case)
            .order_by(Case.created_at.desc())
            .all()
        )
    else:
        org_ids = get_user_organization_ids(db, current_user.id)
        if org_ids:
            query = db.query(Case).filter(
                (Case.created_by == current_user.id)
                | (Case.organization_id.in_(org_ids))
            )
        else:
            query = db.query(Case).filter(
                Case.created_by == current_user.id
            )
        cases = query.order_by(Case.created_at.desc()).all()

    return cases


@router.get(
    "/{case_id}",
    response_model=CaseResponse
)
def get_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = db.get(Case, case_id)

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )

    if not can_view_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case"
        )

    return case


@router.patch(
    "/{case_id}",
    response_model=CaseResponse
)
def update_case(
    case_id: int,
    case_data: CaseUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = db.get(Case, case_id)

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )

    if not can_edit_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this case"
        )

    update_data = case_data.model_dump(exclude_unset=True)

    # Moving a case into (or between) organizations re-checks the
    # attach rule against the NEW organization_id.
    if "organization_id" in update_data:
        require_case_organization_access(
            db, current_user, update_data["organization_id"]
        )

    for field, value in update_data.items():
        setattr(case, field, value)

    db.commit()
    db.refresh(case)

    return case


@router.delete(
    "/{case_id}",
    status_code=status.HTTP_200_OK
)
def delete_case(
    case_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    case = db.get(Case, case_id)

    if case is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found"
        )

    if not can_edit_case(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this case"
        )

    # ---- Phase 2: ordered cascade over ALL case evidence ----
    # Storage objects first (originals + reserved derived scope, per
    # photo), so a storage failure aborts before any row is touched.
    # Rows second, FK-leaf-first, in a single commit, so no orphaned
    # MinIO objects or database rows can remain.
    case_photos = (
        db.query(CasePhoto)
        .filter(CasePhoto.case_id == case.id)
        .all()
    )
    sightings = (
        db.query(Sighting)
        .filter(Sighting.case_id == case.id)
        .all()
    )
    sighting_photos = (
        db.query(SightingPhoto)
        .filter(SightingPhoto.case_id == case.id)
        .all()
    )
    try:
        for photo in case_photos:
            storage.delete_prefix(storage.photo_prefix(case.id, photo.id))
            storage.delete_prefix(
                storage.derived_photo_prefix(case.id, photo.id)
            )
        for photo in sighting_photos:
            storage.delete_prefix(
                storage.sighting_photo_prefix(
                    case.id, photo.sighting_id, photo.id
                )
            )
            storage.delete_prefix(
                storage.derived_sighting_photo_prefix(
                    case.id, photo.sighting_id, photo.id
                )
            )
    except Exception:
        db.rollback()
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Case storage is unavailable; nothing was deleted",
        )

    from app.services import face_detection_service

    for photo in case_photos:
        face_detection_service.delete_runs_for_photo(
            db, "case", photo.id
        )
    for photo in sighting_photos:
        face_detection_service.delete_runs_for_photo(
            db, "sighting", photo.id
        )
    for photo in sighting_photos:
        db.delete(photo)
    for sighting in sightings:
        db.delete(sighting)
    for photo in case_photos:
        db.delete(photo)
    db.delete(case)
    db.commit()

    return {
        "message": "Case deleted successfully"
    }
