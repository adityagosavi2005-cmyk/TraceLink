from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    DATABASE_URL: str
    JWT_SECRET_KEY: str
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # ---- Phase 1: S3-compatible object storage ----
    # Local dev points at MinIO; production points at a real
    # S3-compatible endpoint. Only these values change, never code.
    S3_ENDPOINT_URL: str | None = None
    S3_BUCKET: str = "tracelink-dev"
    S3_ACCESS_KEY: str | None = None
    S3_SECRET_KEY: str | None = None
    S3_REGION: str = "us-east-1"
    PRESIGNED_URL_TTL_SECONDS: int = 600
    max_photo_size_bytes: int = 10 * 1024 * 1024

    # ---- Phase 4: face detection ----
    # Active detector identity recorded on every detection run; a
    # change marks previous runs stale (history preserved).
    FACE_DETECTOR_NAME: str = "yunet"
    FACE_DETECTOR_VERSION: str = "2023mar"
    FACE_DETECTION_THRESHOLD: float = 0.6
    # Filesystem path to the YuNet ONNX weights. Never inside the
    # evidence bucket. None falls back to backend/app/assets/.
    YUNET_MODEL_PATH: str | None = None

    # ---- Phase 5: face representation ----
    # Active representation identity recorded on every embedding; a
    # change starts new rows (history preserved, never overwritten).
    FACE_REPRESENTATION_NAME: str = "sface"
    FACE_REPRESENTATION_VERSION: str = "2021dec"
    # Filesystem path to the SFace ONNX weights. Never inside the
    # evidence bucket. None falls back to backend/app/assets/.
    SFACE_MODEL_PATH: str | None = None

    # ---- Phase 7: image enhancement (Real-ESRGAN) ----
    # Active enhancer identity recorded on every enhancement run.
    # The adapter loads weights lazily from ENHANCER_MODEL_PATH
    # (never inside the evidence bucket); None falls back to
    # backend/app/assets/. A missing file/dependency fails
    # enhancement with a controlled error, never at import time.
    ENHANCER_NAME: str = "realesrgan"
    ENHANCER_VERSION: str = "v1"
    ENHANCER_MODEL_NAME: str = "RealESRGAN_x4plus"
    ENHANCER_MODEL_VERSION: str = "v0.1.0"
    ENHANCER_MODEL_PATH: str | None = None
    # Inference device for enhancement ("cpu" default; "cuda" when
    # a CUDA torch build is installed). "auto" prefers CUDA.
    ENHANCER_DEVICE: str = "cpu"
    # Upscale factor of the configured model (informational; the
    # adapter asserts the output matches it).
    ENHANCEMENT_SCALE: int = 4
    # Resource boundary: derived inputs larger than this (in pixels)
    # are rejected before inference. Phase 3 caps the long edge at
    # 1024 px, so legitimate inputs stay far below this limit.
    ENHANCEMENT_MAX_INPUT_PIXELS: int = 2 * 1024 * 1024

    # Backward-compatible alias: routers/tests still read PHOTO_MAX_BYTES.
    @property
    def PHOTO_MAX_BYTES(self) -> int:
        return self.max_photo_size_bytes

    @PHOTO_MAX_BYTES.setter
    def PHOTO_MAX_BYTES(self, value: int) -> None:
        self.max_photo_size_bytes = value

    model_config = SettingsConfigDict(
        env_file=".env"
    )


settings = Settings()
