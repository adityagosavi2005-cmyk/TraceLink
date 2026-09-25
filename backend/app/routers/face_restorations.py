from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.dependencies import (
    can_trigger_face_restoration,
    can_view_case,
    get_current_user,
)
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.face_detection import FaceDetection
from app.models.sighting import Sighting
from app.models.sighting_photo import SightingPhoto
from app.models.user import User
from app.schemas.face_restoration import (
    FaceRestorationRunDetailResponse,
    FaceRestorationRunResponse,
    RestoredEmbeddingResponse,
)
from app.services import face_restoration_service, storage
from app.services.face_representation import RepresentationError

case_router = APIRouter(
    prefix="/cases/{case_id}/photos/{photo_id}/faces/{face_id}/restorations",
    tags=["Face Restoration"],
)

sighting_router = APIRouter(
    prefix="/cases/{case_id}/sightings/{sighting_id}/photos/{photo_id}/faces/{face_id}/restorations",
    tags=["Face Restoration"],
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


def _require_trigger(user: User, case: Case, db: Session) -> None:
    if not can_trigger_face_restoration(user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to run face restoration",
        )


def _require_view(user: User, case: Case, db: Session) -> None:
    if not can_view_case(user, case, db):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to view this case",
        )


def _detail(run) -> FaceRestorationRunDetailResponse:
    """Run row plus a presigned view URL for COMPLETE artifacts."""
    view_url: str | None = None
    expires_in: int | None = None
    if run.status.value == "COMPLETE" and run.storage_key:
        view_url, expires_in = storage.presigned_get_url(run.storage_key)
    return FaceRestorationRunDetailResponse(
        id=run.id,
        case_id=run.case_id,
        case_photo_id=run.case_photo_id,
        sighting_photo_id=run.sighting_photo_id,
        sighting_id=run.sighting_id,
        face_detection_id=run.face_detection_id,
        face_detection_run_id=run.face_detection_run_id,
        status=run.status,
        source_derived_sha256=run.source_derived_sha256,
        prep_version=run.prep_version,
        prepared_input_sha256=run.prepared_input_sha256,
        restorer_name=run.restorer_name,
        restorer_version=run.restorer_version,
        model_name=run.model_name,
        model_version=run.model_version,
        model_sha256=run.model_sha256,
        parameters=run.parameters,
        restored_geometry_kind=run.restored_geometry_kind,
        restored_geometry_version=run.restored_geometry_version,
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


def _embedding_response(row) -> RestoredEmbeddingResponse:
    source_type = row.source_type
    if hasattr(source_type, "value"):
        source_type = source_type.value
    return RestoredEmbeddingResponse(
        id=row.id,
        face_detection_id=row.face_detection_id,
        face_restoration_run_id=row.face_restoration_run_id,
        source_derived_sha256=row.source_derived_sha256,
        source_type=source_type,
        source_sha256=row.source_sha256,
        representation_name=row.representation_name,
        representation_version=row.representation_version,
        model_name=row.model_name,
        model_version=row.model_version,
        dimension=row.dimension,
        created_at=row.created_at,
    )


def _represent_or_409(db: Session, kind: str, photo, face, run_id: int):
    try:
        return face_restoration_service.represent_restored_face(
            db, face, photo, run_id, kind
        )
    except RepresentationError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        )


@case_router.post(
    "",
    response_model=FaceRestorationRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def trigger_case_face_restoration(
    case_id: int,
    photo_id: int,
    face_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Restore one explicitly selected face (never automatic)."""
    case = _get_case_or_404(db, case_id)
    _require_trigger(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    return face_restoration_service.trigger_restoration(
        db,
        "case",
        photo,
        face.id,
        lambda digest: storage.build_restored_face_key(
            case.id, photo.id, face.id, digest
        ),
    )


@case_router.get("", response_model=list[FaceRestorationRunResponse])
def list_case_face_restorations(
    case_id: int,
    photo_id: int,
    face_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _require_view(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    return face_restoration_service.list_runs(
        db, "case", photo.id, face.id
    )


@case_router.get(
    "/{run_id}", response_model=FaceRestorationRunDetailResponse
)
def get_case_face_restoration(
    case_id: int,
    photo_id: int,
    face_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _require_view(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    run = face_restoration_service.get_run(
        db, "case", photo, face.id, run_id
    )
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face restoration run not found",
        )
    return _detail(run)


@case_router.post(
    "/{run_id}/retry", response_model=FaceRestorationRunResponse
)
def retry_case_face_restoration(
    case_id: int,
    photo_id: int,
    face_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Re-run a FAILED or stale PROCESSING run (new history row)."""
    case = _get_case_or_404(db, case_id)
    _require_trigger(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    return face_restoration_service.retry_restoration(
        db,
        "case",
        photo,
        face.id,
        run_id,
        lambda digest: storage.build_restored_face_key(
            case.id, photo.id, face.id, digest
        ),
    )


@case_router.post(
    "/{run_id}/embedding", response_model=RestoredEmbeddingResponse
)
def embed_case_restored_face(
    case_id: int,
    photo_id: int,
    face_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate (or reuse) the SFace embedding of a restored face."""
    case = _get_case_or_404(db, case_id)
    _require_trigger(current_user, case, db)
    photo = _get_case_photo_or_404(db, case.id, photo_id)
    face = _get_face_or_404(db, "case", photo, face_id)
    return _embedding_response(
        _represent_or_409(db, "case", photo, face, run_id)
    )


@sighting_router.post(
    "",
    response_model=FaceRestorationRunResponse,
    status_code=status.HTTP_201_CREATED,
)
def trigger_sighting_face_restoration(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Restore one explicitly selected face (never automatic)."""
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_trigger(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    face = _get_face_or_404(db, "sighting", photo, face_id)
    return face_restoration_service.trigger_restoration(
        db,
        "sighting",
        photo,
        face.id,
        lambda digest: storage.build_sighting_restored_face_key(
            case.id, sighting_id, photo.id, face.id, digest
        ),
    )


@sighting_router.get("", response_model=list[FaceRestorationRunResponse])
def list_sighting_face_restorations(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_view(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    face = _get_face_or_404(db, "sighting", photo, face_id)
    return face_restoration_service.list_runs(
        db, "sighting", photo.id, face.id
    )


@sighting_router.get(
    "/{run_id}", response_model=FaceRestorationRunDetailResponse
)
def get_sighting_face_restoration(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
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
    face = _get_face_or_404(db, "sighting", photo, face_id)
    run = face_restoration_service.get_run(
        db, "sighting", photo, face.id, run_id
    )
    if run is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Face restoration run not found",
        )
    return _detail(run)


@sighting_router.post(
    "/{run_id}/retry", response_model=FaceRestorationRunResponse
)
def retry_sighting_face_restoration(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
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
    face = _get_face_or_404(db, "sighting", photo, face_id)
    return face_restoration_service.retry_restoration(
        db,
        "sighting",
        photo,
        face.id,
        run_id,
        lambda digest: storage.build_sighting_restored_face_key(
            case.id, sighting_id, photo.id, face.id, digest
        ),
    )


@sighting_router.post(
    "/{run_id}/embedding", response_model=RestoredEmbeddingResponse
)
def embed_sighting_restored_face(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
    run_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Generate (or reuse) the SFace embedding of a restored face."""
    case = _get_case_or_404(db, case_id)
    _get_sighting_or_404(db, case.id, sighting_id)
    _require_trigger(current_user, case, db)
    photo = _get_sighting_photo_or_404(
        db, case.id, sighting_id, photo_id
    )
    face = _get_face_or_404(db, "sighting", photo, face_id)
    return _embedding_response(
        _represent_or_409(db, "sighting", photo, face, run_id)
    )
