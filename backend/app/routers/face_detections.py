from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_trigger_face_detection,
    can_view_case,
    get_current_user,
)
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User
from app.schemas.face_detection import FaceDetectionResultResponse
from app.services import face_detection_service


case_router = APIRouter(
    prefix="/cases/{case_id}/photos/{photo_id}/faces",
    tags=["Face Detection"],
)

sighting_router = APIRouter(
    prefix="/cases/{case_id}/sightings/{sighting_id}/photos/{photo_id}/faces",
    tags=["Face Detection"],
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


def _can_trigger_sighting_detection(
    user: User, sighting: Sighting, case: Case, db: Session
) -> bool:
    """Same narrow rule as case photos: ADMIN or REVIEWER only.

    A REPORTER who reported the sighting gains no detection
    permission from that relationship.
    """
    return can_trigger_face_detection(user, case, db)


def _result(
    db: Session, kind: str, photo
) -> FaceDetectionResultResponse:
    run = face_detection_service.get_current_run(db, kind, photo)
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No face detection has been performed for this photo",
        )
    faces = face_detection_service.get_run_faces(db, run.id)
    return FaceDetectionResultResponse(
        status=run.status,
        face_count=run.face_count,
        run=run,
        faces=faces,
    )


def _run_result(db: Session, kind: str, photo, run) -> (
    FaceDetectionResultResponse
):
    faces = face_detection_service.get_run_faces(db, run.id)
    return FaceDetectionResultResponse(
        status=run.status,
        face_count=run.face_count,
        run=run,
        faces=faces,
    )


@case_router.post("/detect", response_model=FaceDetectionResultResponse)
def detect_case_photo_faces(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Detect faces, reusing the current valid result when present."""
    case = _get_case_or_404(db, case_id)
    if not can_trigger_face_detection(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run face detection",
        )
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    run = face_detection_service.detect_faces(db, "case", photo)
    return _run_result(db, "case", photo, run)


@case_router.post(
    "/redetect", response_model=FaceDetectionResultResponse
)
def redetect_case_photo_faces(
    case_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Always create a new detection run; history is preserved."""
    case = _get_case_or_404(db, case_id)
    if not can_trigger_face_detection(current_user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run face detection",
        )
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    run = face_detection_service.redetect_faces(db, "case", photo)
    return _run_result(db, "case", photo, run)


@case_router.get("", response_model=FaceDetectionResultResponse)
def get_case_photo_faces(
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
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    return _result(db, "case", photo)


@sighting_router.post(
    "/detect", response_model=FaceDetectionResultResponse
)
def detect_sighting_photo_faces(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Detect faces, reusing the current valid result when present."""
    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_trigger_sighting_detection(
        current_user, sighting, case, db
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run face detection",
        )
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting.id, photo_id
    )
    run = face_detection_service.detect_faces(db, "sighting", photo)
    return _run_result(db, "sighting", photo, run)


@sighting_router.post(
    "/redetect", response_model=FaceDetectionResultResponse
)
def redetect_sighting_photo_faces(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Always create a new detection run; history is preserved."""
    case = _get_case_or_404(db, case_id)
    sighting = _get_sighting_or_404(db, case.id, sighting_id)
    if not _can_trigger_sighting_detection(
        current_user, sighting, case, db
    ):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run face detection",
        )
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting.id, photo_id
    )
    run = face_detection_service.redetect_faces(db, "sighting", photo)
    return _run_result(db, "sighting", photo, run)


@sighting_router.get("", response_model=FaceDetectionResultResponse)
def get_sighting_photo_faces(
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
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    return _result(db, "sighting", photo)
