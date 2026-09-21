from datetime import datetime, timezone

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from app.models.sighting import SightingStatus


# Clock-skew tolerance for sightings reported "right now".
_FUTURE_TOLERANCE_SECONDS = 5 * 60

_EARLIEST_SIGHTING = datetime(1900, 1, 1, tzinfo=timezone.utc)


def _ensure_aware(value: datetime) -> datetime:
    """Treat naive datetimes as UTC so comparisons stay well-defined."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class SightingCreate(BaseModel):
    sighting_at: datetime
    location_text: str = Field(min_length=1, max_length=500)
    description: str = Field(min_length=1, max_length=5000)

    # Optional coordinate pair: both or neither (no map/geo-index in
    # Phase 2; free text carries the location).
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)

    contact_info: str | None = Field(default=None, max_length=500)

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("sighting_at")
    @classmethod
    def _sighting_at_in_range(cls, value: datetime) -> datetime:
        moment = _ensure_aware(value)
        now = datetime.now(timezone.utc)
        if moment < _EARLIEST_SIGHTING:
            raise ValueError("sighting_at is implausibly far in the past")
        if (moment - now).total_seconds() > _FUTURE_TOLERANCE_SECONDS:
            raise ValueError("sighting_at cannot be in the future")
        return value

    @model_validator(mode="after")
    def _coordinates_as_pair(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError(
                "latitude and longitude must be provided together"
            )
        return self


class SightingUpdate(BaseModel):
    sighting_at: datetime | None = None
    location_text: str | None = Field(
        default=None, min_length=1, max_length=500
    )
    description: str | None = Field(
        default=None, min_length=1, max_length=5000
    )
    latitude: float | None = Field(default=None, ge=-90, le=90)
    longitude: float | None = Field(default=None, ge=-180, le=180)
    contact_info: str | None = Field(default=None, max_length=500)
    status: SightingStatus | None = None

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("sighting_at")
    @classmethod
    def _sighting_at_in_range(cls, value: datetime | None):
        if value is None:
            return value
        moment = _ensure_aware(value)
        now = datetime.now(timezone.utc)
        if moment < _EARLIEST_SIGHTING:
            raise ValueError("sighting_at is implausibly far in the past")
        if (moment - now).total_seconds() > _FUTURE_TOLERANCE_SECONDS:
            raise ValueError("sighting_at cannot be in the future")
        return value

    @model_validator(mode="after")
    def _coordinates_as_pair(self):
        if (self.latitude is None) != (self.longitude is None):
            raise ValueError(
                "latitude and longitude must be provided together"
            )
        return self


class SightingResponse(BaseModel):
    id: int
    case_id: int
    reported_by: int
    sighting_at: datetime
    location_text: str
    latitude: float | None
    longitude: float | None
    description: str
    contact_info: str | None
    status: SightingStatus
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(
        from_attributes=True
    )
