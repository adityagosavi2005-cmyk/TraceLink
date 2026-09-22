"""Real-ESRGAN production image enhancer (Phase 7).

Real x4 super-resolution behind the ImageEnhancer protocol, using the
official RealESRGAN_x4plus weights (RRDBNet: scale 4, 64 features,
23 blocks) loaded directly with torch. torch and the model weights
are loaded lazily so importing this module never requires either: a
missing runtime or model file surfaces as EnhancementError at
enhancement time, which the service records as a FAILED run.

No realesrgan/basicsr/gfpgan packages are used: basicsr pins no
torchvision upper bound yet imports an API removed from modern
torchvision, and realesrgan drags GFPGAN (an explicit Phase 7
non-goal). The ~60-line RRDBNet below is the standard architecture
consuming the official weights strictly (strict load), keeping the
only third-party runtime to torch itself (CPU build by default).

Model weights are resolved from configuration (ENHANCER_MODEL_PATH
or the bundled assets directory) and must NEVER live in the evidence
bucket nor be committed to Git. See backend/app/assets/README.md for
provenance (SHA-256 recorded there and stored per-run).

Flow per image (all in memory, originals/derived never touched):
    derived JPEG bytes -> RGB float tensor -> RRDBNet x4 ->
    clamped uint8 -> deterministic JPEG -> EnhancedImage.
"""

import hashlib
import io
import os

from app.services.enhancement import (
    EnhancedImage,
    EnhancementError,
    ImageEnhancer,
)

DEFAULT_MODEL_FILENAME = "RealESRGAN_x4plus.pth"

# RRDBNet hyper-parameters of the official RealESRGAN_x4plus
# weights (verified against the release file's state dict:
# conv_first (64,3,3,3), 23 body blocks, conv_up1/conv_up2 tails).
RRDB_NUM_FEAT = 64
RRDB_NUM_BLOCK = 23
RRDB_GROWTH = 32
RRDB_SCALE = 4

# Deterministic output encoding (mirrors the Phase 3 convention:
# pinned flags, no timestamps) so the same input + weights yield
# bit-identical bytes and therefore a stable output SHA-256.
OUTPUT_MIME_TYPE = "image/jpeg"
OUTPUT_JPEG_QUALITY = 90

# Whole-image inference only (v1 boundary): derived inputs are
# capped at ENHANCEMENT_MAX_INPUT_PIXELS by the service, so the x4
# output fits comfortably in memory. Tiled inference is future work.


def default_model_path() -> str:
    """Filesystem path where the Real-ESRGAN weights are expected."""
    override = os.environ.get("ENHANCER_MODEL_PATH")
    if override:
        return override
    try:
        from app.core.config import settings

        if settings.ENHANCER_MODEL_PATH:
            return settings.ENHANCER_MODEL_PATH
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(
        os.path.dirname(here), "assets", DEFAULT_MODEL_FILENAME
    )


def hash_model_file(model_path: str) -> str:
    """SHA-256 of the exact weights file used (provenance)."""
    if not os.path.exists(model_path):
        raise EnhancementError(
            "Image enhancement model is unavailable; try again later"
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
        raise EnhancementError(
            "Image enhancement model is unavailable; try again later"
        ) from exc


def _resolve_device():
    torch = _torch()
    try:
        from app.core.config import settings

        want = (settings.ENHANCER_DEVICE or "cpu").lower()
    except Exception:
        want = "cpu"
    if want == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    if want == "cuda" and not torch.cuda.is_available():
        raise EnhancementError(
            "Image enhancement model is unavailable; try again later"
        )
    if want not in ("cpu", "cuda"):
        raise EnhancementError(
            "Image enhancement model is unavailable; try again later"
        )
    return torch.device(want)


def _rrdb_net_class():
    """RRDBNet architecture (standard Real-ESRGAN definition)."""
    torch = _torch()
    nn = torch.nn
    import torch.nn.functional as F

    class ResidualDenseBlock(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
            self.conv2 = nn.Conv2d(
                num_feat + num_grow_ch, num_grow_ch, 3, 1, 1
            )
            self.conv3 = nn.Conv2d(
                num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1
            )
            self.conv4 = nn.Conv2d(
                num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1
            )
            self.conv5 = nn.Conv2d(
                num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1
            )
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        def forward(self, x):
            x1 = self.lrelu(self.conv1(x))
            x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
            x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
            x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
            x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
            return x5 * 0.2 + x

    class RRDB(nn.Module):
        def __init__(self, num_feat=64, num_grow_ch=32):
            super().__init__()
            self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
            self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

        def forward(self, x):
            out = self.rdb1(x)
            out = self.rdb2(out)
            out = self.rdb3(out)
            return out * 0.2 + x

    class RRDBNet(nn.Module):
        def __init__(
            self,
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        ):
            super().__init__()
            self.scale = scale
            self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
            self.body = nn.Sequential(
                *[
                    RRDB(num_feat, num_grow_ch)
                    for _ in range(num_block)
                ]
            )
            self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(
                negative_slope=0.2, inplace=True
            )

        def forward(self, x):
            feat = self.conv_first(x)
            body_feat = self.conv_body(self.body(feat))
            feat = feat + body_feat
            feat = self.lrelu(
                self.conv_up1(
                    F.interpolate(feat, scale_factor=2, mode="nearest")
                )
            )
            feat = self.lrelu(
                self.conv_up2(
                    F.interpolate(feat, scale_factor=2, mode="nearest")
                )
            )
            out = self.conv_last(self.lrelu(self.conv_hr(feat)))
            return out

    return RRDBNet


class RealESRGANAdapter:
    """Production ImageEnhancer backed by Real-ESRGAN x4 weights."""

    name = "realesrgan"
    model_name = "RealESRGAN_x4plus"
    scale = RRDB_SCALE

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

                version = version or settings.ENHANCER_VERSION
                model_version = (
                    model_version or settings.ENHANCER_MODEL_VERSION
                )
            except Exception:
                version = version or "v1"
                model_version = model_version or "v0.1.0"
        self.version = version
        self.model_version = model_version
        self._net = None
        self._device = None
        self._model_sha256: str | None = None

    @property
    def model_sha256(self) -> str:
        """SHA-256 of the exact weights file, hashed at load time."""
        if self._model_sha256 is None:
            self._model_sha256 = hash_model_file(self.model_path)
        return self._model_sha256

    def _load(self):
        """Load weights strictly, or raise EnhancementError."""
        if self._net is not None:
            return self._net
        torch = _torch()
        # Hash first: provenance is captured for the exact file used,
        # and a missing file fails here before torch is exercised.
        self._model_sha256 = hash_model_file(self.model_path)
        device = _resolve_device()
        try:
            raw = torch.load(
                self.model_path,
                map_location="cpu",
                weights_only=True,
            )
        except Exception as exc:
            raise EnhancementError(
                "Image enhancement failed: unreadable model"
            ) from exc
        params = raw.get("params_ema", raw) if isinstance(raw, dict) else None
        if not isinstance(params, dict):
            raise EnhancementError(
                "Image enhancement failed: unreadable model"
            )
        RRDBNet = _rrdb_net_class()
        try:
            net = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                scale=RRDB_SCALE,
                num_feat=RRDB_NUM_FEAT,
                num_block=RRDB_NUM_BLOCK,
                num_grow_ch=RRDB_GROWTH,
            )
            # Strict: the file must match the x4plus architecture
            # exactly; anything else is a controlled failure.
            net.load_state_dict(params, strict=True)
            net.eval()
            net.to(device)
        except EnhancementError:
            raise
        except Exception as exc:
            raise EnhancementError(
                "Image enhancement failed: unreadable model"
            ) from exc
        self._net = net
        self._device = device
        return net

    def enhance(self, image_bytes: bytes) -> EnhancedImage:
        if not image_bytes:
            raise EnhancementError(
                "Image enhancement failed: empty image"
            )
        try:
            from PIL import Image
            import numpy as np
        except ImportError as exc:
            raise EnhancementError(
                "Image enhancement model is unavailable; "
                "try again later"
            ) from exc
        try:
            with Image.open(io.BytesIO(image_bytes)) as src:
                image = src.convert("RGB")
                width, height = image.size
                pixels = np.asarray(image, dtype=np.float32) / 255.0
        except EnhancementError:
            raise
        except Exception as exc:
            raise EnhancementError(
                "Image enhancement failed: unreadable image"
            ) from exc
        if width <= 0 or height <= 0:
            raise EnhancementError(
                "Image enhancement failed: unreadable image"
            )
        net = self._load()
        torch = _torch()
        try:
            with torch.no_grad():
                import numpy as np

                tensor = (
                    torch.from_numpy(pixels)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .to(self._device)
                )
                output = net(tensor).squeeze(0).clamp_(0, 1).cpu()
                array = (
                    (output.permute(1, 2, 0).numpy() * 255.0)
                    .round()
                    .astype("uint8")
                )
        except EnhancementError:
            raise
        except Exception as exc:
            raise EnhancementError(
                "Image enhancement failed during inference"
            ) from exc
        out_height, out_width = array.shape[0], array.shape[1]
        if out_width != width * self.scale or (
            out_height != height * self.scale
        ):
            raise EnhancementError(
                "Image enhancement failed during inference"
            )
        try:
            from PIL import Image

            out = io.BytesIO()
            Image.fromarray(array, mode="RGB").save(
                out,
                format="JPEG",
                quality=OUTPUT_JPEG_QUALITY,
                optimize=False,
                progressive=False,
            )
            enhanced = out.getvalue()
        except Exception as exc:
            raise EnhancementError(
                "Image enhancement failed during inference"
            ) from exc
        return EnhancedImage(
            image_bytes=enhanced,
            width=out_width,
            height=out_height,
            mime_type=OUTPUT_MIME_TYPE,
        )


def get_image_enhancer() -> ImageEnhancer:
    """Production enhancer factory used by the service by default."""
    return RealESRGANAdapter()
