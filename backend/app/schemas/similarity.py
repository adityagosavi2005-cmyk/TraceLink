from pydantic import BaseModel, ConfigDict, Field

from app.services.similarity_service import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOP_K,
    MAX_TOP_K,
)


class SimilarityRequest(BaseModel):
    top_k: int = Field(
        default=DEFAULT_TOP_K, ge=1, le=MAX_TOP_K
    )
    threshold: float = Field(
        default=DEFAULT_THRESHOLD, ge=-1.0, le=1.0
    )


    restoration_run_id: int | None = None


class SimilarityCandidate(BaseModel):
    """One retrieval candidate. No embedding vector is exposed."""

    face_id: int
    photo_id: int
    photo_type: str
    sighting_id: int | None = None
    case_id: int
    similarity: float
    # Phase 7 source provenance: DERIVED (normal Phase 3 image) or
    # ENHANCED, plus the enhancement run behind enhanced results.
    # Phase 8 restored-face provenance (face_restoration_run_id,
    # is_restored, synthesized_detail_warning) follows below.
    source_type: str = "DERIVED"
    enhancement_run_id: int | None = None
    face_restoration_run_id: int | None = None
    is_restored: bool = False
    synthesized_detail_warning: str | None = None

    model_config = ConfigDict(from_attributes=True)


class SimilarityResponse(BaseModel):
    """Dynamic retrieval result for one query face."""

    query_face_id: int
    query_photo_id: int
    query_photo_type: str
    top_k: int
    threshold: float
    results: list[SimilarityCandidate]
