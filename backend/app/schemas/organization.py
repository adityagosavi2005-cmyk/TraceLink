from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.models.organization import OrgRole


class OrganizationCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = None

    model_config = ConfigDict(str_strip_whitespace=True)


class OrganizationResponse(BaseModel):
    id: int
    name: str
    description: str | None
    created_by: int
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class MembershipAdd(BaseModel):
    user_id: int
    role: OrgRole = OrgRole.INVESTIGATOR


class MembershipUpdate(BaseModel):
    role: OrgRole


class MembershipResponse(BaseModel):
    id: int
    user_id: int
    organization_id: int
    role: OrgRole
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)
