"""Test del wrapper di elaborazione immagine (services.frame_engine)."""
import pytest

from app.services.frame_engine import render_image


def test_render_color_returns_png(sample_png):
    png = render_image(
        sample_png, {"detection_engine": "color", "threshold": 120, "inner_style": "acid"}
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_render_with_partial_config_uses_defaults(sample_png):
    # Config parziale: ProcessingConfig riempie i default mancanti
    png = render_image(sample_png, {"blob_shape": "rectangular"})
    assert png[:4] == b"\x89PNG"


def test_invalid_image_raises():
    with pytest.raises(ValueError):
        render_image(b"not-an-image", {})
