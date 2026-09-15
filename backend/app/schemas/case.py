from datetime import datetime

from pydantic import BaseModel, ConfigDict

from app.models.case import CaseStatus


class CaseCreate(BaseModel):
    title: str
    description: str


class CaseUpdate(BaseModel):
    title: str | None = None
    description: str | None = None
    status: CaseStatus | None = None


class CaseResponse(BaseModel):
    id: int
    title: str
    description: str
    status: CaseStatus
    created_by: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )