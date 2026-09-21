"""YuNet production face detector (Phase 4).

Real YuNet inference behind the FaceDetector protocol, backed by the
OpenCV FaceDetectorYN (YuNet) implementation. OpenCV and the model
weights are loaded lazily so importing this module never requires
either: a missing runtime or model file surfaces as DetectorError
at detection time, which the service records as a FAILED run.

Model weights are resolved from configuration (YUNET_MODEL_PATH or
the bundled assets directory) and must NEVER live in the evidence
bucket. See backend/app/assets/README.md for how to obtain them.
"""

import os

from app.services.face_detector import (
    DetectedFace,
    DetectorError,
    FaceDetector,
)

# Landmark order emitted by OpenCV FaceDetectorYN rows:
# right eye, left eye, nose tip, right mouth corner, left mouth.
_YUNET_LANDMARK_NAMES = (
    "right_eye",
    "left_eye",
    "nose_tip",
    "mouth_right",
    "mouth_left",
)

DEFAULT_MODEL_FILENAME = "face_detection_yunet_2023mar.onnx"


def default_model_path() -> str:
    """Filesystem path where the YuNet weights are expected."""
    override = os.environ.get("YUNET_MODEL_PATH")
    if override:
        return override
    try:
        from app.core.config import settings

        if settings.YUNET_MODEL_PATH:
            return settings.YUNET_MODEL_PATH
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(here), "assets", DEFAULT_MODEL_FILENAME
    )


class YuNetDetector:
    """Production FaceDetector backed by OpenCV FaceDetectorYN."""

    name = "yunet"

    def __init__(
        self,
        model_path: str | None = None,
        version: str | None = None,
        score_threshold: float | None = None,
    ) -> None:
        self.model_path = model_path or default_model_path()
        if version is None or score_threshold is None:
            try:
                from app.core.config import settings

                version = version or settings.FACE_DETECTOR_VERSION
                score_threshold = (
                    score_threshold
                    if score_threshold is not None
                    else settings.FACE_DETECTION_THRESHOLD
                )
            except Exception:
                version = version or "2023mar"
                score_threshold = (
                    score_threshold
                    if score_threshold is not None
                    else 0.6
                )
        self.version = version
        self.score_threshold = score_threshold
        self._detector = None

    def _load(self):
        """Create the OpenCV detector, or raise DetectorError."""
        if self._detector is not None:
            return self._detector
        if not os.path.exists(self.model_path):
            raise DetectorError(
                "Face detection model is unavailable; try again later"
            )
        try:
            import cv2
        except ImportError as exc:
            raise DetectorError(
                "Face detection model is unavailable; try again later"
            ) from exc
        try:
            detector = cv2.FaceDetectorYN.create(
                self.model_path,
                "",
                (320, 320),
                self.score_threshold,
                0.3,
                5000,
                cv2.dnn.DNN_BACKEND_OPENCV,
                cv2.dnn.DNN_TARGET_CPU,
            )
        except Exception as exc:
            raise DetectorError(
                "Face detection model is unavailable; try again later"
            ) from exc
        self._detector = detector
        return detector

    def detect(
        self, image_bytes: bytes, width: int, height: int
    ) -> list[DetectedFace]:
        if not image_bytes:
            raise DetectorError("Face detection failed: empty image")
        detector = self._load()
        try:
            import cv2
            import numpy as np
        except ImportError as exc:
            raise DetectorError(
                "Face detection model is unavailable; try again later"
            ) from exc
        try:
            buf = np.frombuffer(image_bytes, dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        except Exception as exc:
            raise DetectorError(
                "Face detection failed: unreadable image"
            ) from exc
        if img is None:
            raise DetectorError(
                "Face detection failed: unreadable image"
            )
        try:
            detector.setInputSize((width, height))
            _, faces = detector.detect(img)
        except Exception as exc:
            raise DetectorError(
                "Face detection failed during inference"
            ) from exc
        if faces is None or len(faces) == 0:
            return []
        results: list[DetectedFace] = []
        for row in faces:
            x, y, w, h = (
                int(row[0]),
                int(row[1]),
                int(row[2]),
                int(row[3]),
            )
            score = float(row[14])
            landmarks = {
                name: [float(row[4 + 2 * i]), float(row[5 + 2 * i])]
                for i, name in enumerate(_YUNET_LANDMARK_NAMES)
            }
            results.append(
                DetectedFace(
                    x_min=x,
                    y_min=y,
                    x_max=x + w,
                    y_max=y + h,
                    confidence=score,
                    landmarks=landmarks,
                )
            )
        return results


def get_face_detector() -> FaceDetector:
    """Production detector factory used by the service by default."""
    return YuNetDetector()
