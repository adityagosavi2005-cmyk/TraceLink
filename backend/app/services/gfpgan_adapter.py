"""GFPGAN production face restorer (Phase 8).

Face-level restoration behind the FaceRestorer protocol, using the
GFPGAN face-restoration weights. The ``gfpgan`` package, its
runtime dependencies, and the model weights are loaded lazily so
importing this module never requires any of them: a missing
runtime or model file surfaces as RestorationError at restoration
time, which the service records as a FAILED run.

Model weights are resolved from configuration (GFPGAN_MODEL_PATH
or the bundled assets directory) and must NEVER live in the
evidence bucket nor be committed to Git. See
backend/app/assets/README.md for provenance. Because the GFPGAN
inference pipeline performs its own internal face detection and
alignment, restored outputs are reported with ``realigned=True``
so the service uses versioned canonical restored-frame geometry
(see face_preparation) rather than mapped source landmarks.

Compatibility: the gfpgan/basicsr runtime predates the pinned
torchvision and is loaded through the local shim in
gfpgan_compat (imported lazily inside _load, never at app
startup). Auxiliary facexlib weights (RetinaFace detection,
face parsing) are pre-flighted: GFPGANer downloads them from
the network when absent from its ``gfpgan/weights`` directory,
so a missing file fails here with RestorationError instead of
triggering a request-time download. Inference device comes from
RESTORER_DEVICE ("cpu" default; "cuda"/"auto" like Phase 7).

Flow per face (all in memory, originals/derived never touched):
    prepared face bytes -> GFPGAN -> restored face ->
    deterministic JPEG -> RestoredFace.
"""

import hashlib
import io
import json
import os

from app.services.face_preparation import GFPGAN_INPUT_SIZE
from app.services.face_restoration import (
    RestoredFace,
    RestorationError,
)

DEFAULT_MODEL_FILENAME = "GFPGANv1.3.pth"

# Auxiliary facexlib weights GFPGANer needs at construction
# (internal RetinaFace detection + face parsing). GFPGANer
# hardcodes their directory as ``gfpgan/weights`` relative to the
# process working directory and DOWNLOADS them from these URLs
# when absent -- the pre-flight below fails closed instead so no
# request ever triggers network egress. (Pre-seed procedure and
# verified hashes: backend/app/assets/README.md.)
AUX_MODEL_DIRNAME = os.path.join("gfpgan", "weights")
AUX_MODELS = (
    (
        "detection_Resnet50_Final.pth",
        "https://github.com/xinntao/facexlib/releases/"
        "download/v0.1.0/detection_Resnet50_Final.pth",
    ),
    (
        "parsing_parsenet.pth",
        "https://github.com/xinntao/facexlib/releases/"
        "download/v0.2.2/parsing_parsenet.pth",
    ),
)

# Deterministic output encoding (mirrors the Phase 3/7
# convention: pinned flags, no timestamps). Note the encoding is
# deterministic but the upstream GFPGAN forward is NOT
# bit-deterministic run to run on CPU (measured max pixel diff
# ~17/255 between consecutive enhance() calls on identical
# input), so the output SHA-256 identifies the exact artifact
# bytes produced -- never a canonical result. Re-triggering a
# restoration legitimately produces a new artifact (the history
# design already appends a new run per trigger).
OUTPUT_MIME_TYPE = "image/jpeg"
OUTPUT_JPEG_QUALITY = 95


def default_model_path() -> str:
    """Filesystem path where the GFPGAN weights are expected."""
    override = os.environ.get("GFPGAN_MODEL_PATH")
    if override:
        return override
    try:
        from app.core.config import settings

        if settings.GFPGAN_MODEL_PATH:
            return settings.GFPGAN_MODEL_PATH
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(here), "assets", DEFAULT_MODEL_FILENAME
    )


def hash_model_file(model_path: str) -> str:
    """SHA-256 of the exact weights file used (provenance)."""
    if not os.path.exists(model_path):
        raise RestorationError(
            "Face restoration model is unavailable; try again later"
        )
    digest = hashlib.sha256()
    with open(model_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _torch():
    try:
        import torch

        return torch
    except ImportError as exc:
        raise RestorationError(
            "Face restoration model is unavailable; "
            "try again later"
        ) from exc


def _resolve_device():
    """Inference device from RESTORER_DEVICE (mirrors Phase 7).

    "cpu" default; "cuda" requires a CUDA torch build; "auto"
    prefers CUDA. Anything else fails closed: a meaningless
    device string must never silently run on an unintended
    device.
    """
    torch = _torch()
    try:
        from app.core.config import settings

        want = (settings.RESTORER_DEVICE or "cpu").lower()
    except Exception:
        want = "cpu"
    if want == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if want == "cuda" and not torch.cuda.is_available():
        raise RestorationError(
            "Face restoration model is unavailable; "
            "try again later"
        )
    if want not in ("cpu", "cuda"):
        raise RestorationError(
            "Face restoration model is unavailable; "
            "try again later"
        )
    return torch.device(want)


def expected_aux_paths() -> list:
    """Absolute aux-weight paths facexlib will resolve.

    facexlib computes ``abspath(join('gfpgan/weights', name))``
    against the process working directory and downloads when the
    file is absent, so these exact paths are pre-flighted before
    GFPGANer is constructed. Run/serve from backend/ with
    backend/gfpgan/weights/ pre-seeded (see assets README).
    """
    return [
        os.path.abspath(os.path.join(AUX_MODEL_DIRNAME, name))
        for name, _ in AUX_MODELS
    ]


def _check_aux_models() -> None:
    """Fail closed when an auxiliary weight file is missing.

    Called before GFPGANer construction so a missing file raises
    RestorationError here instead of triggering a network
    download inside facexlib during a request.
    """
    for path in expected_aux_paths():
        if not os.path.isfile(path):
            raise RestorationError(
                "Face restoration model is unavailable; "
                "try again later"
            )


class GFPGANAdapter:
    """Production FaceRestorer backed by GFPGAN weights."""

    name = "gfpgan"
    model_name = "GFPGANv1.3"

    def __init__(
        self,
        model_path: str | None = None,
        version: str | None = None,
        model_version: str | None = None,
    ) -> None:
        self.model_path = model_path or default_model_path()
        if version is None or model_version is None:
            try:
                from app.core.config import settings

                version = version or settings.RESTORER_VERSION
                model_version = (
                    model_version or settings.GFPGAN_MODEL_VERSION
                )
            except Exception:
                version = version or "v1"
                model_version = model_version or "v1.3"
        self.version = version
        self.model_version = model_version
        self._restorer = None
        self._model_sha256: str | None = None
        self._device = None
        self._aux_model_info: str | None = None

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the exact weights file, hashed at load time."""
        if self._model_sha256 is None:
            self._model_sha256 = hash_model_file(self.model_path)
        return self._model_sha256

    @property
    def aux_model_info(self) -> str | None:
        """Auxiliary-model provenance JSON (None until loaded).

        Records the exact detection/parsing weight files the
        loaded GFPGANer was constructed with (file name, source
        URL, SHA-256). The service persists this in the run's
        aux_model_info provenance field when present.
        """
        return self._aux_model_info

    def _load(self):
        """Load the GFPGAN restorer, or raise RestorationError."""
        if self._restorer is not None:
            return self._restorer
        # Hash first: provenance is captured for the exact file
        # used, and a missing file fails here before any heavy
        # runtime is imported.
        self._model_sha256 = hash_model_file(self.model_path)
        # Pre-flight the auxiliary weights BEFORE the runtime can
        # download them: facexlib fetches missing files from the
        # network during GFPGANer construction, which must never
        # happen inside a request.
        _check_aux_models()
        device = _resolve_device()
        # Local torchvision compatibility shim (see
        # gfpgan_compat): registered immediately before the
        # gfpgan import, never at application startup.
        from app.services.gfpgan_compat import (
            ensure_gfpgan_compat,
        )

        ensure_gfpgan_compat()
        try:
            from gfpgan import GFPGANer  # type: ignore
        except ImportError as exc:
            raise RestorationError(
                "Face restoration model is unavailable; "
                "try again later"
            ) from exc
        try:
            restorer = GFPGANer(
                model_path=self.model_path,
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=device,
            )
        except Exception as exc:
            raise RestorationError(
                "Face restoration failed: unreadable model"
            ) from exc
        self._restorer = restorer
        self._device = device
        self._aux_model_info = json.dumps(
            {
                "device": str(device),
                "aux_models": [
                    {
                        "file": name,
                        "url": url,
                        "sha256": hash_model_file(path),
                    }
                    for (name, url), path in zip(
                        AUX_MODELS, expected_aux_paths()
                    )
                ],
            },
            sort_keys=True,
        )
        return restorer

    def restore(self, prepared_bytes: bytes) -> RestoredFace:
        if not prepared_bytes:
            raise RestorationError(
                "Face restoration failed: empty image"
            )
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise RestorationError(
                "Face restoration model is unavailable; "
                "try again later"
            ) from exc
        try:
            with Image.open(io.BytesIO(prepared_bytes)) as src:
                image = src.convert("RGB")
                array = np.asarray(image)
        except RestorationError:
            raise
        except Exception as exc:
            raise RestorationError(
                "Face restoration failed: unreadable image"
            ) from exc
        if (
            array.shape[0] != GFPGAN_INPUT_SIZE
            or array.shape[1] != GFPGAN_INPUT_SIZE
        ):
            raise RestorationError(
                "Face restoration failed: "
                "invalid prepared-image dimensions"
            )
        restorer = self._load()
        try:
            import cv2

            bgr = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
            _, restored_faces, _ = restorer.enhance(
                bgr,
                has_aligned=False,
                only_center_face=True,
                paste_back=False,
            )
            if not restored_faces:
                raise RestorationError(
                    "Face restoration failed during inference"
                )
            restored_bgr = restored_faces[0]
            restored_rgb = cv2.cvtColor(
                restored_bgr, cv2.COLOR_BGR2RGB
            )
        except RestorationError:
            raise
        except Exception as exc:
            raise RestorationError(
                "Face restoration failed during inference"
            ) from exc
        out_height, out_width = (
            int(restored_rgb.shape[0]),
            int(restored_rgb.shape[1]),
        )
        if out_width <= 0 or out_height <= 0:
            raise RestorationError(
                "Face restoration failed during inference"
            )
        try:
            from PIL import Image

            out = io.BytesIO()
            Image.fromarray(restored_rgb).save(
                out,
                format="JPEG",
                quality=OUTPUT_JPEG_QUALITY,
                optimize=False,
                progressive=False,
            )
            restored = out.getvalue()
        except Exception as exc:
            raise RestorationError(
                "Face restoration failed during inference"
            ) from exc
        # The GFPGAN pipeline detects/aligns internally, so the
        # output frame is not the prepared-input frame: mapped
        # source landmarks are invalid for these pixels.
        return RestoredFace(
            image_bytes=restored,
            width=out_width,
            height=out_height,
            mime_type=OUTPUT_MIME_TYPE,
            realigned=True,
        )


def get_face_restorer():
    """Production restorer factory used by the service by default."""
    return GFPGANAdapter()
