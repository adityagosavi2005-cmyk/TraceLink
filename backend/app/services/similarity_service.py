"""Face-similarity retrieval (Phase 6 + Phase 7 sources).

Case faces search sighting faces and sighting faces search case
faces (never same-type). Retrieval is technical similarity only:
it never determines identity and never persists results.

Validity model (shared with Phase 4/5/7, not redefined here): a
candidate embedding participates only when it matches the active
representation identity and its source still verifies through the
centralized image_source resolver (artifact present, SHAs match).
Normal (DERIVED) candidates additionally require the current valid
COMPLETE detection run for their photo; enhanced (ENHANCED)
candidates require a valid enhancement chain instead, so several
enhancements of one photo may each contribute candidates. Scores
are never merged and no source is preferred: every result carries
its source provenance. History rows are never treated as current.

Metric: cosine similarity via pgvector cosine distance
(``embedding <=> query``) with ``similarity = 1 - distance``.
Higher API similarity means more similar; pgvector orders by
distance ascending. On PostgreSQL the ordering/threshold runs
in SQL; on other dialects (SQLite test runs) the same filter is
evaluated in Python with the shared ``cosine_distance`` helper
so behavior stays identical.
"""

import math

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.models.case import Case
from app.models.case_photo import CasePhoto
from app.models.enhancement import ImageSourceType
from app.models.face_detection import (
    FaceDetection,
    FaceDetectionRun,
    FaceDetectionStatus,
)
from app.models.face_embedding import FaceEmbedding
from app.models.sighting_photo import SightingPhoto
from app.services import face_detection_service, image_source
from app.services.face_representation import REPRESENTATION_DIMENSION
from app.services.face_representation_service import get_existing_embedding

# Request bounds: Top-K stays small and reviewable; similarity is
# the cosine range extended to the full [-1, 1] domain.
DEFAULT_TOP_K = 10
MAX_TOP_K = 100
DEFAULT_THRESHOLD = 0.70


def active_retrieval_identity() -> tuple[str, str, str, str, int]:
    """Representation identity retrieval accepts, read live.

    Built from the same settings + adapter defaults the
    production SFace engine uses, so the candidate filter can
    never drift from what Phase 5 actually stores. Instantiating
    the adapter performs no I/O (model weights load lazily).
    """
    from app.services.sface_representation import SFaceRepresentation

    adapter = SFaceRepresentation()
    return (
        adapter.representation_name,
        adapter.representation_version,
        adapter.model_name,
        adapter.model_version,
        adapter.dimension,
    )


def cosine_distance(a, b) -> float:
    """Cosine distance (1 - cosine similarity) between two vectors.

    Pure helper shared by the SQLite fallback path and the test
    oracle. Returns 0.0 for identical directions, 1.0 for
    orthogonal, 2.0 for opposite. A zero-norm input carries no
    direction and is treated as maximally dissimilar.
    """
    va = [float(v) for v in a]
    vb = [float(v) for v in b]
    if len(va) != len(vb):
        raise ValueError("cosine_distance requires equal-length vectors")
    dot = sum(x * y for x, y in zip(va, vb))
    na = math.sqrt(sum(x * x for x in va))
    nb = math.sqrt(sum(y * y for y in vb))
    if na == 0.0 or nb == 0.0 or not math.isfinite(dot):
        return 1.0
    return 1.0 - dot / (na * nb)


def cosine_similarity(a, b) -> float:
    """Cosine similarity, derived from cosine distance."""
    return 1.0 - cosine_distance(a, b)


def _not_found(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND, detail=detail
    )


def _conflict(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT, detail=detail
    )


def get_query_embedding(
    db: Session, face: FaceDetection, photo
) -> FaceEmbedding:
    """Current valid embedding for the query face, or raise.

    404-class problems (missing face/photo binding) are checked
    by the router; here a face with no usable current embedding
    is a 409 so callers can distinguish "valid query, no
    candidates" (200 + empty) from "unusable query" (error).
    """
    from app.services.sface_representation import SFaceRepresentation

    representation = SFaceRepresentation()
    identity = (
        representation.representation_name,
        representation.representation_version,
        representation.model_name,
        representation.model_version,
    )
    run = db.get(FaceDetectionRun, face.run_id)
    if run is None or run.status != FaceDetectionStatus.COMPLETE:
        raise _conflict(
            "Query face is not on a current face-detection run; "
            "run face detection first"
        )
    try:
        source = image_source.resolve_run_source(db, run, photo)
    except image_source.SourceResolutionError:
        raise _conflict(
            "Query face is not on a current face-detection run; "
            "run face detection first"
        )
    row = get_existing_embedding(
        db, face.id, identity, source.source_type, source.source_sha256
    )
    if row is None:
        raise _conflict(
            "Query face has no face representation yet; "
            "generate its embedding first"
        )
    if row.source_derived_sha256 != photo.derived_sha256:
        raise _conflict(
            "Query face representation is stale for the current "
            "derived image"
        )
    return row


def _photo_for_kind(db: Session, kind: str, photo_id: int):
    photo = db.get(
        CasePhoto if kind == "case" else SightingPhoto, photo_id
    )
    return photo


def _candidate_photo(db: Session, kind: str, face: FaceDetection):
    """Photo row owning a candidate face of the opposite kind."""
    if kind == "case":
        return db.get(CasePhoto, face.case_photo_id)
    return db.get(SightingPhoto, face.sighting_photo_id)


def _run_is_current_for_photo(
    db: Session, kind: str, photo, run: FaceDetectionRun
) -> bool:
    """True when run is the current valid COMPLETE run of photo."""
    current = face_detection_service.get_current_run(db, kind, photo)
    return current is not None and current.id == run.id


def _case_in_scope(case: Case, scope_org_ids: set[int], user_id: int) -> bool:
    """Organization scope for one candidate/query case.

    Organization cases qualify by membership; personal cases
    (NULL organization) qualify only for their owner. Callers
    bypass this entirely for ADMIN.
    """
    if case.organization_id is not None:
        return case.organization_id in scope_org_ids
    return case.created_by == user_id


def collect_candidates(
    db: Session,
    kind: str,
    query_face_id: int,
    scope_org_ids: set[int],
    user_id: int,
    is_admin: bool = False,
) -> list[dict]:
    """Valid opposite-type candidates with their provenance.

    Applies every Phase 6/7 validity rule (compatible identity,
    verified source, opposite evidence type, query-face exclusion,
    organization scope) BEFORE any distance is computed, so
    unauthorized or stale rows can never leak into ranking. Normal
    candidates additionally require the current COMPLETE run;
    enhanced candidates require a valid enhancement chain instead
    (several enhancements of one photo may each qualify). Returns
    plain dicts (never ORM rows with raw vectors attached beyond
    the embedding values needed for scoring).
    """
    rep_name, rep_version, model_name, model_version, dimension = (
        active_retrieval_identity()
    )
    opposite = "sighting" if kind == "case" else "case"
    if opposite == "sighting":
        type_column = FaceDetection.sighting_photo_id
        null_column = FaceDetection.case_photo_id
    else:
        type_column = FaceDetection.case_photo_id
        null_column = FaceDetection.sighting_photo_id

    rows = (
        db.query(FaceEmbedding, FaceDetection)
        .join(
            FaceDetection,
            FaceEmbedding.face_detection_id == FaceDetection.id,
        )
        .filter(
            FaceEmbedding.representation_name == rep_name,
            FaceEmbedding.representation_version == rep_version,
            FaceEmbedding.model_name == model_name,
            FaceEmbedding.model_version == model_version,
            FaceEmbedding.dimension == dimension,
            FaceEmbedding.face_detection_id != query_face_id,
            type_column.is_not(None),
            null_column.is_(None),
        )
        .all()
    )
    candidates: list[dict] = []
    run_cache: dict[int, FaceDetectionRun | None] = {}
    photo_cache: dict[tuple[str, int], object] = {}
    case_cache: dict[int, Case | None] = {}
    source_cache: dict[int, object] = {}
    for embedding, face in rows:
        run_id = face.run_id
        if run_id not in run_cache:
            run_cache[run_id] = db.get(FaceDetectionRun, run_id)
        run = run_cache[run_id]
        if run is None or run.status != FaceDetectionStatus.COMPLETE:
            continue
        photo_id = (
            face.sighting_photo_id
            if opposite == "sighting"
            else face.case_photo_id
        )
        cache_key = (opposite, photo_id)
        if cache_key not in photo_cache:
            photo_cache[cache_key] = _candidate_photo(db, opposite, face)
        photo = photo_cache[cache_key]
        if photo is None:
            continue
        # Centralized source verification (artifact present, SHAs
        # match, enhancement chain valid when ENHANCED). Stale,
        # deleted, or tampered sources never reach ranking.
        if run_id not in source_cache:
            try:
                source_cache[run_id] = image_source.resolve_run_source(
                    db, run, photo
                )
            except image_source.SourceResolutionError:
                source_cache[run_id] = None
        source = source_cache[run_id]
        if source is None:
            continue
        if (
            embedding.source_type != run.source_type
            or embedding.source_sha256 != source.source_sha256
        ):
            continue
        if embedding.source_derived_sha256 != photo.derived_sha256:
            continue
        if run.source_type == ImageSourceType.DERIVED:
            if not _run_is_current_for_photo(db, opposite, photo, run):
                continue
        elif run.enhancement_run_id is None:
            continue
        if face.case_id not in case_cache:
            case_cache[face.case_id] = db.get(Case, face.case_id)
        case = case_cache[face.case_id]
        if case is None:
            continue
        if photo.case_id != case.id:
            continue
        if not is_admin and not _case_in_scope(
            case, scope_org_ids, user_id
        ):
            continue
        candidates.append(
            {
                "embedding": embedding,
                "face": face,
                "run": run,
                "photo": photo,
            }
        )
    return candidates


def _build_item(candidate: dict, photo_type: str, similarity: float) -> dict:
    face = candidate["face"]
    photo = candidate["photo"]
    run = candidate["run"]
    return {
        "face_id": face.id,
        "photo_id": photo.id,
        "photo_type": photo_type,
        "sighting_id": run.sighting_id,
        "case_id": face.case_id,
        "similarity": similarity,
        # Phase 7 provenance: DERIVED (normal) vs ENHANCED plus
        # the enhancement run behind an enhanced candidate.
        "source_type": run.source_type.value
        if isinstance(run.source_type, ImageSourceType)
        else run.source_type,
        "enhancement_run_id": run.enhancement_run_id,
    }


def _rank_python(
    candidates: list[dict],
    query_vector: list[float],
    photo_type: str,
    threshold: float,
    top_k: int,
) -> list[dict]:
    """Threshold + rank + Top-K over cosine similarity (fallback)."""
    scored = [
        (cosine_distance(query_vector, list(c["embedding"].embedding)), c)
        for c in candidates
    ]
    scored.sort(key=lambda pair: pair[0])
    items: list[dict] = []
    for distance, candidate in scored:
        similarity = 1.0 - distance
        if similarity < threshold:
            continue
        items.append(_build_item(candidate, photo_type, similarity))
        if len(items) >= top_k:
            break
    return items


def _rank_pgvector(
    db: Session,
    candidates: list[dict],
    query_vector: list[float],
    photo_type: str,
    threshold: float,
    top_k: int,
) -> list[dict]:
    """Threshold + rank + Top-K using pgvector cosine distance.

    Only the pre-filtered valid candidate ids enter the SQL
    query, so the database orders exactly the authorized set:
    ``ORDER BY embedding <=> :query`` with ``similarity =
    1 - distance``. Raw vectors never leave the database.
    """
    if not candidates:
        return []
    ids = [c["embedding"].id for c in candidates]
    distance = FaceEmbedding.embedding.cosine_distance(query_vector)
    rows = (
        db.query(FaceEmbedding, distance.label("distance"))
        .filter(
            FaceEmbedding.id.in_(ids),
            (1.0 - distance) >= threshold,
        )
        .order_by(distance.asc())
        .limit(top_k)
        .all()
    )
    by_id = {c["embedding"].id: c for c in candidates}
    return [
        _build_item(by_id[row.id], photo_type, 1.0 - float(dist))
        for row, dist in rows
    ]


def search_similar_faces(
    db: Session,
    kind: str,
    query_face: FaceDetection,
    query_photo,
    scope_org_ids: set[int],
    user_id: int,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_THRESHOLD,
    is_admin: bool = False,
) -> tuple[FaceEmbedding, list[dict]]:
    """Run retrieval for one query face.

    Returns (query embedding row, candidate items). Raises
    HTTPException for unusable queries; a valid query with no
    candidates yields an empty list (never an error).
    """
    if top_k < 1 or top_k > MAX_TOP_K:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="top_k must be between 1 and %d" % MAX_TOP_K,
        )
    if not math.isfinite(threshold) or threshold < -1.0 or threshold > 1.0:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="threshold must be between -1.0 and 1.0",
        )
    query_row = get_query_embedding(db, query_face, query_photo)
    query_vector = list(query_row.embedding)
    candidates = collect_candidates(
        db, kind, query_face.id, scope_org_ids, user_id, is_admin
    )
    photo_type = "sighting" if kind == "case" else "case"
    bind = db.get_bind() if hasattr(db, "get_bind") else db.bind
    dialect = getattr(getattr(bind, "dialect", None), "name", "")
    if dialect == "postgresql":
        items = _rank_pgvector(
            db, candidates, query_vector, photo_type, threshold, top_k
        )
    else:
        items = _rank_python(
            candidates, query_vector, photo_type, threshold, top_k
        )
    return query_row, items


def expected_settings_identity() -> dict:
    """Active retrieval identity for diagnostics/tests."""
    rep_name, rep_version, model_name, model_version, dimension = (
        active_retrieval_identity()
    )
    return {
        "representation_name": rep_name,
        "representation_version": rep_version,
        "model_name": model_name,
        "model_version": model_version,
        "dimension": dimension,
        "detector_name": settings.FACE_DETECTOR_NAME,
        "detector_version": settings.FACE_DETECTOR_VERSION,
        "threshold": settings.FACE_DETECTION_THRESHOLD,
    }
