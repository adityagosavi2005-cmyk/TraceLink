from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import require_role
from app.models.case import Case
from app.models.user import User, UserRole
from app.schemas.case import CaseCreate, CaseResponse, CaseUpdate
from app.core.dependencies import get_current_user, require_role


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
    new_case = Case(
        title=case_data.title,
        description=case_data.description,
        created_by=current_user.id
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
    if current_user.role == UserRole.ADMIN:
        cases = (
            db.query(Case)
            .order_by(Case.created_at.desc())
            .all()
        )
    else:
        cases = (
            db.query(Case)
            .filter(Case.created_by == current_user.id)
            .order_by(Case.created_at.desc())
            .all()
        )

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

    if (
        current_user.role != UserRole.ADMIN
        and case.created_by != current_user.id
    ):
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

    if (
        current_user.role != UserRole.ADMIN
        and case.created_by != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to update this case"
        )

    update_data = case_data.model_dump(exclude_unset=True)

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

    if (
        current_user.role != UserRole.ADMIN
        and case.created_by != current_user.id
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to delete this case"
        )

    db.delete(case)
    db.commit()

    return {
        "message": "Case deleted successfully"
    }