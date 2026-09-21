"""Phase 3 unit tests: shared deterministic preprocessor.

Pure Pillow byte transformation; no database, no S3, no app imports
beyond the preprocessor service.

Run from backend/:  python -m pytest tests/test_preprocessing.py -v
"""

import hashlib
import io

import pytest
from PIL import Image

from app.services.preprocessing import (
    DERIVED_JPEG_QUALITY,
    DERIVED_MAX_LONG_EDGE,
    DERIVED_MIME_TYPE,
    PROCESSOR_VERSION,
    PreprocessingError,
    preprocess,
)


def _png_bytes(mode="RGB", size=(8, 6), color="red"):
    img = Image.new(mode, size, color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_same_input_same_derived_bytes_and_sha():
    data = _png_bytes(size=(64, 48))
    first = preprocess(data)
    second = preprocess(data)
    assert first[0] == second[0]
    assert first[1] == second[1]
    assert first[1] == hashlib.sha256(first[0]).hexdigest()


def test_processor_version_pinned():
    assert PROCESSOR_VERSION == "phase3-v1"
    _, _, _, _, _, version = preprocess(_png_bytes())
    assert version == PROCESSOR_VERSION


def test_output_is_jpeg_rgb_quality_contract():
    derived, _, width, height, mime_type, _ = preprocess(
        _png_bytes(size=(64, 48))
    )
    assert derived[:3] == b"\xff\xd8\xff"
    assert mime_type == DERIVED_MIME_TYPE == "image/jpeg"
    with Image.open(io.BytesIO(derived)) as probe:
        assert probe.format == "JPEG"
        assert probe.mode == "RGB"
        assert probe.size == (width, height) == (64, 48)


def test_exif_orientation_corrected():
    img = Image.new("RGB", (20, 10), "blue")
    exif = Image.Exif()
    exif[274] = 6  # stored sideways; display orientation is 10x20
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    _, _, width, height, _, _ = preprocess(buf.getvalue())
    assert (width, height) == (10, 20)


def test_metadata_and_exif_stripped():
    img = Image.new("RGB", (16, 12), "green")
    exif = Image.Exif()
    exif[271] = "TraceLink-Test-Camera"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    derived, _, _, _, _, _ = preprocess(buf.getvalue())
    with Image.open(io.BytesIO(derived)) as probe:
        assert len(probe.getexif()) == 0
        assert "exif" not in probe.info


def test_palette_and_cmyk_normalized_to_rgb():
    for mode in ("P", "CMYK"):
        color = 0 if mode == "P" else (10, 20, 30, 40)
        img = Image.new(mode, (16, 12), color)
        buf = io.BytesIO()
        img.save(buf, format="PNG" if mode == "P" else "JPEG")
        derived, _, _, _, mime_type, _ = preprocess(buf.getvalue())
        assert mime_type == "image/jpeg"
        with Image.open(io.BytesIO(derived)) as probe:
            assert probe.mode == "RGB"


def test_rgba_transparency_flattened_onto_white():
    img = Image.new("RGBA", (8, 6), (255, 0, 0, 255))
    for x in range(4, 8):
        for y in range(6):
            img.putpixel((x, y), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    derived, _, _, _, _, _ = preprocess(buf.getvalue())
    with Image.open(io.BytesIO(derived)) as probe:
        assert probe.mode == "RGB"
        opaque = probe.getpixel((1, 3))
        assert opaque[0] > 200 and opaque[1] < 80 and opaque[2] < 80
        flattened = probe.getpixel((6, 3))
        assert all(channel > 250 for channel in flattened)


def test_large_image_scaled_to_long_edge_1024():
    data = _png_bytes(size=(2000, 1000))
    _, _, width, height, _, _ = preprocess(data)
    assert (width, height) == (1024, 512)
    assert max(width, height) == DERIVED_MAX_LONG_EDGE


def test_odd_dimensions_scale_proportionally():
    _, _, width, height, _, _ = preprocess(_png_bytes(size=(2001, 1001)))
    assert width == 1024
    assert height == round(1001 * 1024 / 2001) == 512


def test_small_image_never_upscaled():
    _, _, width, height, _, _ = preprocess(_png_bytes(size=(100, 80)))
    assert (width, height) == (100, 80)


def test_aspect_ratio_preserved_portrait():
    _, _, width, height, _, _ = preprocess(_png_bytes(size=(500, 2000)))
    assert (width, height) == (256, 1024)


def test_jpeg_quality_contract_constant():
    assert DERIVED_JPEG_QUALITY == 85


def test_corrupt_image_fails_safely():
    with pytest.raises(PreprocessingError):
        preprocess(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    with pytest.raises(PreprocessingError):
        preprocess(b"hello text, not an image")
    with pytest.raises(PreprocessingError):
        preprocess(b"")


def test_derived_metadata_matches_actual_bytes():
    data = _png_bytes(size=(300, 200))
    derived, digest, width, height, mime_type, _ = preprocess(data)
    assert digest == hashlib.sha256(derived).hexdigest()
    with Image.open(io.BytesIO(derived)) as probe:
        assert probe.size == (width, height)
    assert mime_type == "image/jpeg"
