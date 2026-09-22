from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enhancement import ImageSourceType
from app.models.face_detection import FaceDetectionStatus


class FaceDetectionResponse(BaseModel):
    id: int
    run_id: int
    case_id: int
    case_photo_id: int | None = None
    sighting_photo_id: int | None = None
    ordinal: int
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    confidence: float
    frame_width: int
    frame_height: int
    landmarks: dict | None = None

    model_config = ConfigDict(from_attributes=True)


class FaceDetectionRunResponse(BaseModel):
    id: int
    case_id: int
    case_photo_id: int | None = None
    sighting_photo_id: int | None = None
    sighting_id: int | None = None
    status: FaceDetectionStatus
    detector_name: str
    detector_version: str
    threshold: float
    source_derived_sha: str
    # Phase 7 source provenance: which artifact was consumed.
    source_type: ImageSourceType
    source_sha256: str
    enhancement_run_id: int | None = None
    source_width: int
    source_height: int
    face_count: int
    error: str | None = None
    started_at: datetime
    finished_at: datetime | None = None

    model_config = ConfigDict(from_attributes=True)


class FaceDetectionResultResponse(BaseModel):
    """Current (or just-produced) run plus its faces."""

    status: FaceDetectionStatus
    face_count: int
    run: FaceDetectionRunResponse
    faces: list[FaceDetectionResponse]
