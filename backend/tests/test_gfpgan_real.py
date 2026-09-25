"""Phase 8 real-model test: prepared real-face crop -> GFPGAN.

Isolated from the deterministic suite: skipped automatically when
the gfpgan runtime, the GFPGAN weights, the auxiliary facexlib
weights, or the real-face fixture are absent, and never required
for the normal test run (see tests/fake_restorer.py).

Fixture (backend/tests/fixtures/real_face.jpg): a real portrait
photograph containing two faces detectable by the RetinaFace
auxiliary model. The test prepares a 512x512 crop around the
center face and proves the production path end to end:

- model loads (main weights + pre-seeded auxiliary weights, no
  network egress),
- the prepared crop is accepted and a restored face is produced,
- output is 512x512 JPEG bytes (repeat inference is
  contract-stable; the upstream forward is not
  bit-deterministic, so no byte-equality is asserted),
- the adapter reports realigned=True, so the run takes the
  CANONICAL restored-geometry strategy,
- the canonical geometry satisfies SFace input validation
  (build_sface_face_row) without claiming recognition accuracy.

No image-quality assertions: the test never claims the restored
face is "better", only that the contract holds.

Run from backend/:  python -m pytest tests/test_gfpgan_real.py -v
"""

import io
import json
import os

import pytest
from PIL import Image

from app.services.face_preparation import (
    GFPGAN_INPUT_SIZE,
    canonical_restored_landmarks,
    prepare_face,
)
from app.services.face_representation import (
    REPRESENTATION_LANDMARK_NAMES,
    build_sface_face_row,
    FaceGeometry,
)
from app.services.face_restoration import RestorationError
from app.services.gfpgan_adapter import (
    GFPGANAdapter,
    default_model_path,
    expected_aux_paths,
)
from tests.fake_representation import yunet_landmarks

FIXTURE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixtures",
    "real_face.jpg",
)

# Center-face box in fixture pixels (fixture is 700x464). The
# RetinaFace auxiliary model detects two faces here; the box below
# frames the higher-scoring center face (approx 419,74,592,320)
# with margin so the prepared crop contains exactly one full face.
FACE_X_MIN = 380
FACE_Y_MIN = 30
FACE_X_MAX = 630
FACE_Y_MAX = 360


def _assets_available() -> bool:
    return (
        os.path.exists(default_model_path())
        and all(
            os.path.isfile(path)
            for path in expected_aux_paths()
        )
        and os.path.exists(FIXTURE_PATH)
    )


def _runtime_available() -> bool:
    try:
        from app.services.gfpgan_compat import (
            ensure_gfpgan_compat,
        )

        ensure_gfpgan_compat()
        from gfpgan import GFPGANer  # noqa: F401

        return True
    except ImportError:
        return False


def _prepared_crop():
    with open(FIXTURE_PATH, "rb") as handle:
        fixture_bytes = handle.read()
    with Image.open(io.BytesIO(fixture_bytes)) as img:
        width, height = img.size
    prepared, _, prep_w, prep_h, _, _ = prepare_face(
        fixture_bytes,
        width,
        height,
        FACE_X_MIN,
        FACE_Y_MIN,
        FACE_X_MAX,
        FACE_Y_MAX,
        yunet_landmarks(450, 150),
    )
    return prepared, prep_w, prep_h


@pytest.mark.skipif(
    not (_assets_available() and _runtime_available()),
    reason="GFPGAN weights/runtime/fixture absent "
    "(see backend/app/assets/README.md)",
)
def test_gfpgan_restores_real_face():
    prepared, prep_w, prep_h = _prepared_crop()
    assert (prep_w, prep_h) == (
        GFPGAN_INPUT_SIZE,
        GFPGAN_INPUT_SIZE,
    )

    adapter = GFPGANAdapter()
    first = adapter.restore(prepared)
    # The real GFPGAN path emits its native 512x512 frame.
    assert (first.width, first.height) == (
        GFPGAN_INPUT_SIZE,
        GFPGAN_INPUT_SIZE,
    )
    assert first.mime_type == "image/jpeg"
    assert len(first.image_bytes) > 0
    # The actual GFPGAN pipeline detects/aligns internally, so
    # the output frame is never the prepared-input frame.
    assert first.realigned is True
    # Runtime provenance is recorded for the exact files used.
    assert len(adapter.model_sha256) == 64
    aux_info = json.loads(adapter.aux_model_info)
    assert len(aux_info["aux_models"]) == 2
    # Repeat inference is contract-stable. The upstream GFPGAN
    # forward is NOT bit-deterministic run to run on CPU
    # (consecutive enhance() calls on identical input differ by
    # up to ~17/255 per pixel), so no byte-equality is asserted:
    # only dimensions, format, and realignment on every call.
    second = adapter.restore(prepared)
    assert (second.width, second.height) == (
        GFPGAN_INPUT_SIZE,
        GFPGAN_INPUT_SIZE,
    )
    assert second.mime_type == "image/jpeg"
    assert len(second.image_bytes) > 0
    assert second.realigned is True
    # The reported CANONICAL strategy must yield SFace-valid
    # geometry over the restored frame.
    landmarks = canonical_restored_landmarks(
        first.width, first.height
    )
    assert set(landmarks) == set(REPRESENTATION_LANDMARK_NAMES)
    row = build_sface_face_row(
        FaceGeometry(
            x_min=0,
            y_min=0,
            x_max=first.width,
            y_max=first.height,
            landmarks=landmarks,
        )
    )
    assert len(row) == 14


def test_gfpgan_rejects_unreadable_input():
    adapter = GFPGANAdapter()
    with pytest.raises(RestorationError):
        adapter.restore(b"not-an-image")
    with pytest.raises(RestorationError):
        adapter.restore(b"")


def test_gfpgan_fails_closed_without_aux_weights(tmp_path, monkeypatch):
    """Missing auxiliary weights fail closed (never download).

    Needs no weights or runtime: with an existing (dummy) main
    weights file but no gfpgan/weights/ directory under the
    working directory, _load must raise RestorationError before
    facexlib can fetch anything from the network.
    """
    dummy = tmp_path / "GFPGANv1.3.pth"
    dummy.write_bytes(b"dummy-weights")
    monkeypatch.chdir(tmp_path)
    adapter = GFPGANAdapter(model_path=str(dummy))
    with pytest.raises(RestorationError):
        adapter.restore(
            _prepared_crop_if_available() or _dummy_prepared()
        )


def _prepared_crop_if_available():
    if not os.path.exists(FIXTURE_PATH):
        return None
    prepared, _, _ = _prepared_crop()
    return prepared


def _dummy_prepared():
    raw = io.BytesIO()
    Image.new("RGB", (GFPGAN_INPUT_SIZE, GFPGAN_INPUT_SIZE)).save(
        raw, format="PNG"
    )
    return raw.getvalue()


def test_gfpgan_rejects_invalid_device(monkeypatch):
    """An unknown RESTORER_DEVICE fails closed (never silent)."""
    from app.core.config import settings

    monkeypatch.setattr(settings, "RESTORER_DEVICE", "bogus")
    from app.services.gfpgan_adapter import _resolve_device

    with pytest.raises(RestorationError):
        _resolve_device()


def test_gfpgan_compat_shim_registers_namespace():
    """The local shim exposes rgb_to_grayscale (no site-packages)."""
    torchvision = pytest.importorskip("torchvision")
    from app.services.gfpgan_compat import ensure_gfpgan_compat

    ensure_gfpgan_compat()
    import sys

    shim = sys.modules[
        "torchvision.transforms.functional_tensor"
    ]
    assert (
        shim.rgb_to_grayscale
        is torchvision.transforms.functional.rgb_to_grayscale
    )


def test_gfpgan_rejects_wrong_dimensions():
    adapter = GFPGANAdapter()
    raw = io.BytesIO()
    Image.new("RGB", (64, 48), "white").save(raw, format="PNG")
    with pytest.raises(RestorationError) as excinfo:
        adapter.restore(raw.getvalue())
    assert "invalid prepared-image dimensions" in str(
        excinfo.value
    )
