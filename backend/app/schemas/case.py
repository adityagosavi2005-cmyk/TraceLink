from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.case import CaseStatus


class CaseCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    description: str = Field(min_length=1)

    # Phase 0 structured fields — all optional so existing clients
    # sending only title/description keep working unchanged.
    full_name: str | None = Field(default=None, max_length=200)
    age_years: int | None = Field(default=None, ge=0, le=150)
    age_estimate_note: str | None = Field(default=None, max_length=255)
    last_seen_at: datetime | None = None
    last_seen_location: str | None = Field(default=None, max_length=500)
    clothing_description: str | None = None
    distinguishing_marks: str | None = None
    contact_info: str | None = Field(default=None, max_length=500)

    # NULL = personal case (see Case model). Attaching to an organization
    # requires membership; enforced in the router, not here.
    organization_id: int | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class CaseUpdate(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    description: str | None = Field(default=None, min_length=1)
    status: CaseStatus | None = None

    full_name: str | None = Field(default=None, max_length=200)
    age_years: int | None = Field(default=None, ge=0, le=150)
    age_estimate_note: str | None = Field(default=None, max_length=255)
    last_seen_at: datetime | None = None
    last_seen_location: str | None = Field(default=None, max_length=500)
    clothing_description: str | None = None
    distinguishing_marks: str | None = None
    contact_info: str | None = Field(default=None, max_length=500)
    organization_id: int | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class CaseResponse(BaseModel):
    id: int
    title: str
    description: str
    status: CaseStatus
    created_by: int
    organization_id: int | None
    full_name: str | None
    age_years: int | None
    age_estimate_note: str | None
    last_seen_at: datetime | None
    last_seen_location: str | None
    clothing_description: str | None
    distinguishing_marks: str | None
    contact_info: str | None
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
