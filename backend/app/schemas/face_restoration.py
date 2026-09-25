from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.face_restoration import FaceRestorationStatus


class FaceRestorationRunResponse(BaseModel):
    """One face-restoration execution: provenance, status, artifact."""

    id: int
    case_id: int
    case_photo_id: int | None = None
    sighting_photo_id: int | None = None
    sighting_id: int | None = None
    face_detection_id: int
    face_detection_run_id: int
    status: FaceRestorationStatus
    source_derived_sha256: str
    prep_version: str
    prepared_input_sha256: str
    restorer_name: str
    restorer_version: str
    model_name: str
    model_version: str
    model_sha256: str
    parameters: str | None = None
    restored_geometry_kind: str
    restored_geometry_version: str
    output_sha256: str | None = None
    mime_type: str
    width: int | None = None
    height: int | None = None
    byte_size: int | None = None
    error_message: str | None = None
    created_at: datetime
    started_at: datetime
    finished_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class FaceRestorationRunDetailResponse(FaceRestorationRunResponse):
    """Run detail plus a short-lived view URL for COMPLETE runs."""

    view_url: str | None = None
    expires_in: int | None = None


class RestoredEmbeddingResponse(BaseModel):
    """The SFace embedding generated from a restored face."""

    id: int
    face_detection_id: int
    face_restoration_run_id: int | None = None
    source_derived_sha256: str
    source_type: str
    source_sha256: str
    representation_name: str
    representation_version: str
    model_name: str
    model_version: str
    dimension: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)
