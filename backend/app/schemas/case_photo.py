from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.case_photo import PhotoStatus


class CasePhotoResponse(BaseModel):
    id: int
    case_id: int
    uploaded_by: int
    mime_type: str
    byte_size: int
    width: int
    height: int
    sha256: str
    processing_status: PhotoStatus
    # Phase 4 face-detection summary (NOT_RUN until a current run exists).
    face_detection_status: str = "NOT_RUN"
    face_count: int = 0
    created_at: datetime
    updated_at: datetime

    # Phase 3 processing metadata (null until/unless processed).
    processing_error: str | None = None
    processing_version: str | None = None
    derived_sha256: str | None = None
    derived_width: int | None = None
    derived_height: int | None = None
    derived_mime_type: str | None = None

    # Freshly minted private access URL; short-lived (see
    # PRESIGNED_URL_TTL_SECONDS). Clients must not store it.
    view_url: str
    expires_in: int

    # Derived rendition URL; present only when READY with a stored
    # derived object. The original view_url above always keeps working.
    derived_view_url: str | None = None
    derived_expires_in: int | None = None

    model_config = ConfigDict(
        from_attributes=True
    )
