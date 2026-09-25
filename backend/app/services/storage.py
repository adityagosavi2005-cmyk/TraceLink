"""S3-compatible object-storage boundary (Phase 1).

This is the ONLY module that imports or calls boto3. Routers and
services work with storage keys and bytes through the functions below
and must never construct S3 clients themselves.

Layout:
    originals/{case_id}/{photo_id}/{sha256}.{ext}   (write-once)
    derived/{case_id}/{photo_id}/{derived_sha}.jpg  (Phase 3 rendition)
    enhanced/{case_id}/{photo_id}/{output_sha}.jpg  (Phase 7 artifact)
    originals/sightings/{case_id}/{sighting_id}/{photo_id}/{sha256}.{ext}
                                                    (Phase 2, write-once)
    derived/sightings/{case_id}/{sighting_id}/{photo_id}/{derived_sha}.jpg
                                                    (Phase 3 rendition)
    enhanced/sightings/{case_id}/{sighting_id}/{photo_id}/{output_sha}.jpg
                                                    (Phase 7 artifact)

All reads are served as short-lived presigned GET URLs minted after
the caller has passed the normal authorization checks.
"""

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings

ORIGINALS_PREFIX = "originals"
DERIVED_PREFIX = "derived"
ENHANCED_PREFIX = "enhanced"
RESTORED_PREFIX = "restored-faces"
SIGHTINGS_SEGMENT = "sightings"


def _client():
    kwargs: dict = {"region_name": settings.S3_REGION}
    if settings.S3_ENDPOINT_URL:
        kwargs["endpoint_url"] = settings.S3_ENDPOINT_URL
    if settings.S3_ACCESS_KEY:
        kwargs["aws_access_key_id"] = settings.S3_ACCESS_KEY
    if settings.S3_SECRET_KEY:
        kwargs["aws_secret_access_key"] = settings.S3_SECRET_KEY
    return boto3.client("s3", **kwargs)


def ensure_bucket() -> None:
    """Create the bucket when missing (local dev convenience)."""
    client = _client()
    try:
        client.head_bucket(Bucket=settings.S3_BUCKET)
    except ClientError as exc:
        code = exc.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchBucket", "Not Found"):
            client.create_bucket(Bucket=settings.S3_BUCKET)
        else:
            raise


def build_original_key(
    case_id: int, photo_id: int, sha256: str, ext: str
) -> str:
    return "%s/%d/%d/%s.%s" % (
        ORIGINALS_PREFIX,
        case_id,
        photo_id,
        sha256,
        ext,
    )


def put_original(key: str, data: bytes, content_type: str) -> None:
    _client().put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def put_derived(key: str, data: bytes, content_type: str) -> None:
    """Store one Phase 3 derived rendition (derived/ scope only).

    Separate from put_original so preprocessing can never address the
    immutable originals/ evidence scope. Callers must build the key
    with build_derived_key / build_sighting_derived_key below.
    """
    if not key.startswith(DERIVED_PREFIX + "/"):
        raise ValueError("Derived objects must live under derived/")
    _client().put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def get_original_bytes(key: str) -> bytes:
    """Read immutable original bytes back for (re)processing."""
    response = _client().get_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
    )
    return response["Body"].read()


def get_derived_bytes(key: str) -> bytes:
    """Read Phase 3 derived bytes back for AI processing.

    Separated from get_original_bytes so callers state explicitly
    which rendition they consume. Guards the derived/ scope the same
    way put_derived guards writes.
    """
    if not key.startswith(DERIVED_PREFIX + "/"):
        raise ValueError("Derived objects must live under derived/")
    response = _client().get_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
    )
    return response["Body"].read()


def build_derived_key(case_id: int, photo_id: int, derived_sha: str) -> str:
    """Deterministic derived key for one case photo (Phase 3)."""
    return "%s/%d/%d/%s.jpg" % (
        DERIVED_PREFIX,
        case_id,
        photo_id,
        derived_sha,
    )


def build_sighting_derived_key(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    derived_sha: str,
) -> str:
    """Deterministic derived key for one sighting photo (Phase 3)."""
    return "%s/%s/%d/%d/%d/%s.jpg" % (
        DERIVED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
        derived_sha,
    )


def presigned_get_url(key: str) -> tuple[str, int]:
    """Mint a short-lived private read URL for one stored object."""
    expires_in = settings.PRESIGNED_URL_TTL_SECONDS
    url = _client().generate_presigned_url(
        "get_object",
        Params={"Bucket": settings.S3_BUCKET, "Key": key},
        ExpiresIn=expires_in,
    )
    return url, expires_in


def delete_prefix(prefix: str) -> None:
    """Remove every object under a key prefix (one photo's scope)."""
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(
        Bucket=settings.S3_BUCKET, Prefix=prefix
    ):
        for obj in page.get("Contents", []):
            keys.append({"Key": obj["Key"]})
    for i in range(0, len(keys), 1000):
        client.delete_objects(
            Bucket=settings.S3_BUCKET,
            Delete={"Objects": keys[i:i + 1000]},
        )


def photo_prefix(case_id: int, photo_id: int) -> str:
    return "%s/%d/%d/" % (ORIGINALS_PREFIX, case_id, photo_id)


def derived_photo_prefix(case_id: int, photo_id: int) -> str:
    """Phase 4+ rendition scope for one case photo (empty in Phase 2)."""
    return "%s/%d/%d/" % (DERIVED_PREFIX, case_id, photo_id)


def build_sighting_original_key(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    sha256: str,
    ext: str,
) -> str:
    return "%s/%s/%d/%d/%d/%s.%s" % (
        ORIGINALS_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
        sha256,
        ext,
    )


def sighting_photo_prefix(
    case_id: int, sighting_id: int, photo_id: int
) -> str:
    return "%s/%s/%d/%d/%d/" % (
        ORIGINALS_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
    )


def derived_sighting_photo_prefix(
    case_id: int, sighting_id: int, photo_id: int
) -> str:
    """Phase 4+ rendition scope for one sighting photo (empty now)."""
    return "%s/%s/%d/%d/%d/" % (
        DERIVED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
    )


def build_enhanced_key(
    case_id: int, photo_id: int, output_sha: str
) -> str:
    """Deterministic enhanced key for one case photo (Phase 7).

    Layout: enhanced/{case_id}/{photo_id}/{output_sha}.jpg
    """
    return "%s/%d/%d/%s.jpg" % (
        ENHANCED_PREFIX,
        case_id,
        photo_id,
        output_sha,
    )


def build_sighting_enhanced_key(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    output_sha: str,
) -> str:
    """Deterministic enhanced key for one sighting photo (Phase 7).

    Layout: enhanced/sightings/{case_id}/{sighting_id}/{photo_id}/
    {output_sha}.jpg
    """
    return "%s/%s/%d/%d/%d/%s.jpg" % (
        ENHANCED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
        output_sha,
    )


def put_enhanced(key: str, data: bytes, content_type: str) -> None:
    """Store one Phase 7 enhanced artifact (enhanced/ scope only).

    Separate from put_original/put_derived so enhancement output can
    never address the immutable originals/ evidence scope or the
    Phase 3 derived scope. Callers must build the key with
    build_enhanced_key / build_sighting_enhanced_key above.
    """
    if not key.startswith(ENHANCED_PREFIX + "/"):
        raise ValueError("Enhanced objects must live under enhanced/")
    _client().put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def get_enhanced_bytes(key: str) -> bytes:
    """Read Phase 7 enhanced bytes back for AI processing.

    Guards the enhanced/ scope the same way put_enhanced guards
    writes.
    """
    if not key.startswith(ENHANCED_PREFIX + "/"):
        raise ValueError("Enhanced objects must live under enhanced/")
    response = _client().get_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
    )
    return response["Body"].read()


def enhanced_photo_prefix(case_id: int, photo_id: int) -> str:
    """Phase 7 artifact scope for one case photo."""
    return "%s/%d/%d/" % (ENHANCED_PREFIX, case_id, photo_id)


def enhanced_sighting_photo_prefix(
    case_id: int, sighting_id: int, photo_id: int
) -> str:
    """Phase 7 artifact scope for one sighting photo."""
    return "%s/%s/%d/%d/%d/" % (
        ENHANCED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
    )


def build_restored_face_key(
    case_id: int, photo_id: int, face_id: int, output_sha: str
) -> str:
    """Deterministic restored-face key for one case photo face.

    Layout: restored-faces/{case_id}/{photo_id}/{face_id}/
    {output_sha}.jpg (content-addressed; the run id is not in
    the key because one run maps to exactly one artifact).
    """
    return "%s/%d/%d/%d/%s.jpg" % (
        RESTORED_PREFIX,
        case_id,
        photo_id,
        face_id,
        output_sha,
    )


def build_sighting_restored_face_key(
    case_id: int,
    sighting_id: int,
    photo_id: int,
    face_id: int,
    output_sha: str,
) -> str:
    """Deterministic restored-face key for one sighting photo face.

    Layout: restored-faces/sightings/{case_id}/{sighting_id}/
    {photo_id}/{face_id}/{output_sha}.jpg
    """
    return "%s/%s/%d/%d/%d/%d/%s.jpg" % (
        RESTORED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
        face_id,
        output_sha,
    )


def put_restored_face(key: str, data: bytes, content_type: str) -> None:
    """Store one Phase 8 restored-face artifact (restored scope).

    Separate from put_original/put_derived/put_enhanced so face
    restoration output can never address the immutable
    originals/ evidence scope, the Phase 3 derived scope, or the
    Phase 7 enhanced scope. Callers must build the key with
    build_restored_face_key / build_sighting_restored_face_key.
    """
    if not key.startswith(RESTORED_PREFIX + "/"):
        raise ValueError("Restored faces must live under restored-faces/")
    _client().put_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
        Body=data,
        ContentType=content_type,
    )


def get_restored_face_bytes(key: str) -> bytes:
    """Read Phase 8 restored-face bytes back for AI processing.

    Guards the restored-faces/ scope the same way put_restored_face
    guards writes.
    """
    if not key.startswith(RESTORED_PREFIX + "/"):
        raise ValueError("Restored faces must live under restored-faces/")
    response = _client().get_object(
        Bucket=settings.S3_BUCKET,
        Key=key,
    )
    return response["Body"].read()


def restored_photo_prefix(case_id: int, photo_id: int) -> str:
    """Phase 8 artifact scope for one case photo (all its faces)."""
    return "%s/%d/%d/" % (RESTORED_PREFIX, case_id, photo_id)


def restored_sighting_photo_prefix(
    case_id: int, sighting_id: int, photo_id: int
) -> str:
    """Phase 8 artifact scope for one sighting photo (all faces)."""
    return "%s/%s/%d/%d/%d/" % (
        RESTORED_PREFIX,
        SIGHTINGS_SEGMENT,
        case_id,
        sighting_id,
        photo_id,
    )
