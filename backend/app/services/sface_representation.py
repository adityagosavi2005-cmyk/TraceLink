"""SFace production face representation (Phase 5).

Real SFace inference behind the FaceRepresentation protocol,
backed by OpenCV FaceRecognizerSF. OpenCV, NumPy, and the model
weights are loaded lazily so importing this module never requires
any of them: a missing runtime or model file surfaces as
RepresentationError at representation time.

Model weights are resolved from configuration (SFACE_MODEL_PATH
or the bundled assets directory) and must NEVER live in the
evidence bucket. See backend/app/assets/README.md for provenance.

Workflow per face (OpenCV Zoo YuNet -> SFace):
    derived image bytes + Phase 4 bbox/landmarks
        -> 14-value SFace row (build_sface_face_row, landmarks kept)
        -> recognizer.alignCrop()  (in memory only, never persisted)
        -> recognizer.feature()
        -> validated 128-dim vector (validate_vector, raw, unmodified)
"""

import hashlib
import os

from app.services.face_representation import (
    FaceGeometry,
    REPRESENTATION_DIMENSION,
    RepresentationError,
    build_sface_face_row,
    validate_vector,
)

DEFAULT_MODEL_FILENAME = "face_recognition_sface_2021dec.onnx"


def default_model_path() -> str:
    """Filesystem path where the SFace weights are expected."""
    override = os.environ.get("SFACE_MODEL_PATH")
    if override:
        return override
    try:
        from app.core.config import settings

        if settings.SFACE_MODEL_PATH:
            return settings.SFACE_MODEL_PATH
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(here), "assets", DEFAULT_MODEL_FILENAME
    )


def hash_model_file(model_path: str) -> str:
    """SHA-256 of the exact model file used (provenance)."""
    if not os.path.exists(model_path):
        raise RepresentationError(
            "Face representation model is unavailable; try again later"
        )
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SFaceRepresentation:
    """Production FaceRepresentation backed by OpenCV SFace."""

    representation_name = "sface"
    model_name = "sface"

    def __init__(
        self,
        model_path: str | None = None,
        version: str | None = None,
    ) -> None:
        self.model_path = model_path or default_model_path()
        if version is None:
            try:
                from app.core.config import settings

                version = settings.FACE_REPRESENTATION_VERSION
            except Exception:
                version = "2021dec"
        # Representation pipeline version and weights version move
        # together for SFace today; stored as separate provenance
        # columns so they can diverge later without a schema change.
        self.representation_version = version
        self.model_version = version
        self.dimension = REPRESENTATION_DIMENSION
        self._recognizer = None
        self._model_sha256: str | None = None

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the exact ONNX file, hashed at load time."""
        if self._model_sha256 is None:
            self._model_sha256 = hash_model_file(self.model_path)
        return self._model_sha256

    def _load(self):
        """Create the OpenCV recognizer, or raise RepresentationError."""
        if self._recognizer is not None:
            return self._recognizer
        # Hash first: provenance is captured for the exact file used,
        # and a missing file fails here before OpenCV is imported.
        self._model_sha256 = hash_model_file(self.model_path)
        try:
            import cv2
        except ImportError as exc:
            raise RepresentationError(
                "Face representation model is unavailable; "
                "try again later"
            ) from exc
        try:
            recognizer = cv2.FaceRecognizerSF.create(
                self.model_path,
                "",
            )
        except Exception as exc:
            raise RepresentationError(
                "Face representation model is unavailable; "
                "try again later"
            ) from exc
        self._recognizer = recognizer
        return recognizer

    def represent(
        self, image_bytes: bytes, geometry: FaceGeometry
    ) -> list[float]:
        if not image_bytes:
            raise RepresentationError(
                "Face representation failed: empty image"
            )
        face_row = build_sface_face_row(geometry)
        recognizer = self._load()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise RepresentationError(
                "Face representation model is unavailable; "
                "try again later"
            ) from exc
        try:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise RepresentationError(
                "Face representation failed: unreadable image"
            ) from exc
        if img is None:
            raise RepresentationError(
                "Face representation failed: unreadable image"
            )
        try:
            face = np.asarray(face_row, dtype=np.float32)
            aligned = recognizer.alignCrop(img, face)
            feature = recognizer.feature(aligned)
        except RepresentationError:
            raise
        except Exception as exc:
            raise RepresentationError(
                "Face representation failed during inference"
            ) from exc
        if feature is None:
            raise RepresentationError(
                "Face representation failed during inference"
            )
        try:
            values = np.ravel(feature).tolist()
        except Exception as exc:
            raise RepresentationError(
                "Face representation failed during inference"
            ) from exc
        # Raw SFace feature, stored unmodified. The aligned crop is
        # a local variable only: never persisted anywhere.
        return validate_vector(values)


def get_face_representation() -> SFaceRepresentation:
    """Production representation factory used by the service."""
    return SFaceRepresentation()
