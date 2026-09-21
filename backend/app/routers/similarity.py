from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_search_similarity,
    get_current_user,
    get_user_organization_ids,
)
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.face_detection import FaceDetection
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User, UserRole
from app.schemas.similarity import (
    SimilarityRequest,
    SimilarityResponse,
)
from app.services import similarity_service


router = APIRouter(tags=["Similarity"])


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


def _get_face_or_404(
    db: Session, kind: str, photo, face_id: int
) -> FaceDetection:
    """Face must belong to the URL photo and the expected type."""
    face = db.get(FaceDetection, face_id)
    if face is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face not found",
        )
    if kind == "case":
        if (
            face.case_photo_id != photo.id
            or face.sighting_photo_id is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
    else:
        if (
            face.sighting_photo_id != photo.id
            or face.case_photo_id is not None
        ):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Face not found",
            )
    if face.case_id != photo.case_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face not found",
        )
    return face


def _authorize_search(
    db: Session, user: User, case: Case
) -> set[int]:
    """Role gate plus query-case scope. Returns org scope for Admin bypass."""
    if not can_search_similarity(user, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to perform similarity retrieval",
        )
    if user.role == UserRole.ADMIN:
        return set()
    scope = get_user_organization_ids(db, user.id)
    if case.organization_id is not None:
        if case.organization_id not in scope:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to search from this case",
            )
    elif case.created_by != user.id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to search from this case",
        )
    return scope


def _respond(
    db: Session,
    kind: str,
    photo,
    face: FaceDetection,
    body: SimilarityRequest,
    scope: set[int],
    user: User,
) -> SimilarityResponse:
    _, items = similarity_service.search_similar_faces(
        db,
        kind,
        face,
        photo,
        scope,
        user.id,
        top_k=body.top_k,
        threshold=body.threshold,
        is_admin=user.role == UserRole.ADMIN,
    )
    return SimilarityResponse(
        query_face_id=face.id,
        query_photo_id=photo.id,
        query_photo_type=kind,
        top_k=body.top_k,
        threshold=body.threshold,
        results=items,
    )


@router.post(
    "/cases/{case_id}/photos/{photo_id}/faces/{face_id}/similar",
    response_model=SimilarityResponse,
)
def search_case_face_similar(
    case_id: int,
    photo_id: int,
    face_id: int,
    body: SimilarityRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Case face queries sighting faces (never case faces)."""
    case = _get_case_or_404(db, case_id)
    scope = _authorize_search(db, current_user, case)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    return _respond(
        db, "case", photo, face, body or SimilarityRequest(), scope,
        current_user,
    )


@router.post(
    "/cases/{case_id}/sightings/{sighting_id}/photos/{photo_id}"
    "/faces/{face_id}/similar",
    response_model=SimilarityResponse,
)
def search_sighting_face_similar(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
    body: SimilarityRequest | None = None,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Sighting face queries case faces (never sighting faces)."""
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    scope = _authorize_search(db, current_user, case)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    face = _get_face_or_404(db, "sighting", photo, face_id)
    return _respond(
        db, "sighting", photo, face, body or SimilarityRequest(),
        scope, current_user,
    )
