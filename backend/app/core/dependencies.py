from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import decode_access_token
from app.models.user import User, UserRole
from app.models.organization import Membership, OrgRole
from typing import Callable


bearer_scheme = HTTPBearer()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(
        bearer_scheme
    ),
    db: Session = Depends(get_db)
) -> User:

    token = credentials.credentials

    try:
        payload = decode_access_token(token)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user_id = payload.get("sub")

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    try:
        user_id = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"}
        )

    user = db.get(User, user_id)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"}
        )

    return user


def require_role(*allowed_roles: UserRole) -> Callable:
    def role_checker(
        current_user: User = Depends(get_current_user)
    ) -> User:

        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )

        return current_user

    return role_checker


# ---- Phase 0: organization / case visibility foundation ----
#
# Intended long-term rule for Case access:
#     owner OR authorized organization member OR REVIEWER OR ADMIN
# These helpers implement the reusable pieces. Routers compose them;
# no second authorization system is introduced.


def get_membership(
    db: Session,
    user_id: int,
    organization_id: int,
) -> Membership | None:
    """Return the user's membership in an organization, if any."""
    return (
        db.query(Membership)
        .filter(
            Membership.user_id == user_id,
            Membership.organization_id == organization_id,
        )
        .first()
    )


def get_user_organization_ids(db: Session, user_id: int) -> set[int]:
    """Return IDs of all organizations the user belongs to."""
    rows = (
        db.query(Membership.organization_id)
        .filter(Membership.user_id == user_id)
        .all()
    )
    return {row[0] for row in rows}


def is_org_manager(
    db: Session,
    user_id: int,
    organization_id: int,
) -> bool:
    """True when the user may manage an organization (ADMIN or ORG_ADMIN)."""
    user = db.get(User, user_id)
    if user is not None and user.role == UserRole.ADMIN:
        return True
    membership = get_membership(db, user_id, organization_id)
    return membership is not None and membership.role == OrgRole.ORG_ADMIN


def can_view_case(user: User, case, db: Session) -> bool:
    """Read rule: owner, org member, REVIEWER, or ADMIN."""
    if user.role == UserRole.ADMIN:
        return True
    if case.created_by == user.id:
        return True
    if user.role == UserRole.REVIEWER:
        return True
    if case.organization_id is not None:
        if get_membership(db, user.id, case.organization_id) is not None:
            return True
    return False


def can_edit_case(user: User, case, db: Session) -> bool:
    """Write rule: owner, ADMIN, or org INVESTIGATOR/ORG_ADMIN.

    Org VIEWERs and REVIEWERs are read-only: they may view cases in
    scope but may not modify or delete them.
    """
    if user.role == UserRole.ADMIN:
        return True
    if case.created_by == user.id:
        return True
    if case.organization_id is not None:
        membership = get_membership(db, user.id, case.organization_id)
        if membership is not None and membership.role in (
            OrgRole.ORG_ADMIN,
            OrgRole.INVESTIGATOR,
        ):
            return True
    return False


def can_trigger_face_detection(user: User, case, db: Session) -> bool:
    """Narrow AI-analysis permission for Phase 4 face detection.

    ONLY ADMIN and REVIEWER may trigger detection. Case-edit
    permission deliberately does NOT imply detection permission:
    case owners and organization case editors (INVESTIGATOR /
    ORG_ADMIN) are denied. REVIEWERs gain no general case/photo /
    sighting edit or delete rights; every other router keeps using
    can_edit_case / can_view_case unchanged.
    """

    return user.role in (UserRole.ADMIN, UserRole.REVIEWER)


def can_trigger_enhancement(user: User, case, db: Session) -> bool:
    """Narrow AI-enhancement permission for Phase 7.

    ONLY ADMIN and REVIEWER may trigger enhancement. Case-edit
    permission deliberately does NOT imply enhancement permission:
    case owners and organization case editors (INVESTIGATOR /
    ORG_ADMIN) are denied, exactly like face detection. REVIEWERs
    gain no general case/photo / sighting edit or delete rights;
    every other router keeps using can_edit_case / can_view_case
    unchanged.
    """

    return user.role in (UserRole.ADMIN, UserRole.REVIEWER)


def can_search_similarity(user: User, db: Session) -> bool:
    """Phase 6 similarity-retrieval permission.

    Separate from modification permission: can_edit_case NEVER
    implies search and search NEVER implies edit. ADMIN and
    REVIEWER may always search. Other users may search only when
    they hold an INVESTIGATOR or ORG_ADMIN membership somewhere;
    REPORTERs and organization VIEWERs are denied. Organization
    scoping itself is applied per candidate by the similarity
    service, not here.
    """

    if user.role in (UserRole.ADMIN, UserRole.REVIEWER):
        return True
    if user.role != UserRole.ORGANIZATION_MEMBER:
        return False
    memberships = (
        db.query(Membership)
        .filter(Membership.user_id == user.id)
        .all()
    )
    return any(
        membership.role
        in (OrgRole.ORG_ADMIN, OrgRole.INVESTIGATOR)
        for membership in memberships
    )


def require_case_organization_access(
    db: Session,
    user: User,
    organization_id: int | None,
) -> None:
    """Enforce attach rule when a case is placed in an organization.

    organization_id=None (personal case) always passes. Otherwise the
    user must be ADMIN or hold an INVESTIGATOR/ORG_ADMIN membership in
    that organization. Raises 403 on violation, 404 on unknown org.
    """
    if organization_id is None:
        return

    from app.models.organization import Organization

    organization = db.get(Organization, organization_id)
    if organization is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Organization not found",
        )

    if user.role == UserRole.ADMIN:
        return

    membership = get_membership(db, user.id, organization_id)
    if membership is None or membership.role not in (
        OrgRole.ORG_ADMIN,
        OrgRole.INVESTIGATOR,
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You are not authorized to file cases under this organization",
        )
