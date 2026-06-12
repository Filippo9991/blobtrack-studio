"""Test unitari della pipeline di elaborazione immagine."""
import pytest

from services.image_processing import normalize_settings, process_image


def test_process_detects_blobs(sample_png):
    png, n = process_image(sample_png, {"threshold": 100, "min_size": 200})
    assert n == 3
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_process_styled_output(sample_png):
    png, n = process_image(
        sample_png,
        {"blob_shape": "circular", "blob_style": "dotted", "wf_type": "linear",
         "inner_style": "acid", "bg_mode": "black", "show_center": True},
    )
    assert n == 3
    assert png[:4] == b"\x89PNG"


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        process_image(b"not-an-image")


def test_normalize_clamps_and_defaults():
    s = normalize_settings(
        {"threshold": "999", "blob_style": "bogus", "max_blobs": -5, "opacity": "x"}
    )
    assert s["threshold"] == 255
    assert s["blob_style"] == "solid"   # valore sconosciuto -> default
    assert s["max_blobs"] >= 1
    assert s["opacity"] == 1.0
