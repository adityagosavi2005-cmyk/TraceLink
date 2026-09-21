from pydantic import BaseModel, ConfigDict, EmailStr, Field
from app.models.user import UserRole

# Server-side request-shape validation.
# Uniqueness stays a database concern (checked in routers);
# business rules stay in routers/services. These schemas only
# enforce shape, types, and sane lengths.


class UserCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)


class UserLogin(BaseModel):
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=1, max_length=128)


class AdminCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    email: EmailStr = Field(max_length=255)
    password: str = Field(min_length=8, max_length=128)


class UserResponse(BaseModel):
    id: int
    name: str
    email: str
    role: UserRole

    model_config = ConfigDict(from_attributes=True)
