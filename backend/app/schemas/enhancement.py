from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.enhancement import EnhancementStatus, ImageSourceType


class EnhancementRunResponse(BaseModel):
    """One enhancement execution: provenance, status, and artifact."""

    id: int
    case_id: int
    case_photo_id: int | None = None
    sighting_photo_id: int | None = None
    sighting_id: int | None = None
    status: EnhancementStatus
    source_derived_sha256: str
    enhancer_name: str
    enhancer_version: str
    model_name: str
    model_version: str
    model_sha256: str
    parameters: str | None = None
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


class EnhancementRunDetailResponse(EnhancementRunResponse):
    """Run detail plus a short-lived view URL for COMPLETE runs."""

    view_url: str | None = None
    expires_in: int | None = None


class FaceDetectionSourceResponse(BaseModel):
    """Source provenance shared by detection results (Phase 7)."""

    source_type: ImageSourceType
    source_sha256: str
    enhancement_run_id: int | None = None
    source_width: int
    source_height: int

    model_config = ConfigDict(from_attributes=True)
