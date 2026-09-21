from fastapi import APIRouter, Depends, status, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    get_current_user,
    get_membership,
    get_user_organization_ids,
    is_org_manager,
    require_role,
)
from app.models.organization import Membership, Organization, OrgRole
from app.models.user import User, UserRole
from app.schemas.organization import (
    MembershipAdd,
    MembershipResponse,
    MembershipUpdate,
    OrganizationCreate,
    OrganizationResponse,
)


router = APIRouter(
    prefix="/organizations",
    tags=["Organizations"]
)


def _get_organization_or_404(db: Session, organization_id: int) -> Organization:
    organization = db.get(Organization, organization_id)
    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )
    return organization


def _require_org_visibility(
    db: Session, user: User, organization_id: int
) -> Organization:
    """ADMIN sees all; other users only organizations they belong to."""
    organization = _get_organization_or_404(db, organization_id)
    if user.role == UserRole.ADMIN:
        return organization
    if get_membership(db, user.id, organization_id) is None:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this organization",
        )
    return organization


@router.post(
    "",
    response_model=OrganizationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_organization(
    org_data: OrganizationCreate,
    current_user: User = Depends(require_role(UserRole.ADMIN)),
    db: Session = Depends(get_db),
):
    organization = Organization(
        name=org_data.name,
        description=org_data.description,
        created_by=current_user.id,
    )
    db.add(organization)
    db.commit()
    db.refresh(organization)
    return organization


@router.get("", response_model=list[OrganizationResponse])
def list_organizations(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if current_user.role == UserRole.ADMIN:
        return (
            db.query(Organization)
            .order_by(Organization.created_at.desc())
            .all()
        )
    org_ids = get_user_organization_ids(db, current_user.id)
    if not org_ids:
        return []
    return (
        db.query(Organization)
        .filter(Organization.id.in_(org_ids))
        .order_by(Organization.created_at.desc())
        .all()
    )


@router.get("/{organization_id}", response_model=OrganizationResponse)
def get_organization(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _require_org_visibility(db, current_user, organization_id)


@router.get(
    "/{organization_id}/members",
    response_model=list[MembershipResponse],
)
def list_members(
    organization_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _require_org_visibility(db, current_user, organization_id)
    return (
        db.query(Membership)
        .filter(Membership.organization_id == organization_id)
        .order_by(Membership.created_at.asc())
        .all()
    )


@router.post(
    "/{organization_id}/members",
    response_model=MembershipResponse,
    status_code=status.HTTP_201_CREATED,
)
def add_member(
    organization_id: int,
    member_data: MembershipAdd,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_organization_or_404(db, organization_id)
    if not is_org_manager(db, current_user.id, organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to manage this organization",
        )

    member_user = db.get(User, member_data.user_id)
    if member_user is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    existing = get_membership(db, member_data.user_id, organization_id)
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="User is already a member of this organization",
        )

    membership = Membership(
        user_id=member_data.user_id,
        organization_id=organization_id,
        role=member_data.role,
    )
    db.add(membership)
    db.commit()
    db.refresh(membership)
    return membership


@router.patch(
    "/{organization_id}/members/{user_id}",
    response_model=MembershipResponse,
)
def update_member_role(
    organization_id: int,
    user_id: int,
    member_data: MembershipUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_organization_or_404(db, organization_id)
    if not is_org_manager(db, current_user.id, organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to manage this organization",
        )

    membership = get_membership(db, user_id, organization_id)
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Membership not found",
        )

    # Never demote the last ORG_ADMIN: an organization must keep
    # at least one administrator.
    if (
        membership.role == OrgRole.ORG_ADMIN
        and member_data.role != OrgRole.ORG_ADMIN
    ):
        remaining_admins = (
            db.query(Membership)
            .filter(
                Membership.organization_id == organization_id,
                Membership.role == OrgRole.ORG_ADMIN,
                Membership.user_id != user_id,
            )
            .count()
        )
        if remaining_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot demote the last administrator of this organization",
            )

    membership.role = member_data.role
    db.commit()
    db.refresh(membership)
    return membership


@router.delete(
    "/{organization_id}/members/{user_id}",
    status_code=status.HTTP_200_OK,
)
def remove_member(
    organization_id: int,
    user_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    _get_organization_or_404(db, organization_id)
    if not is_org_manager(db, current_user.id, organization_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to manage this organization",
        )

    membership = get_membership(db, user_id, organization_id)
    if membership is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Membership not found",
        )

    # Never remove the last ORG_ADMIN.
    if membership.role == OrgRole.ORG_ADMIN:
        remaining_admins = (
            db.query(Membership)
            .filter(
                Membership.organization_id == organization_id,
                Membership.role == OrgRole.ORG_ADMIN,
                Membership.user_id != user_id,
            )
            .count()
        )
        if remaining_admins == 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Cannot remove the last administrator of this organization",
            )

    db.delete(membership)
    db.commit()
    return {"message": "Member removed successfully"}
