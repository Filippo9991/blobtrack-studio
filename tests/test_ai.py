"""Test dell'AI Preset Generator (gestione errori + salvataggio)."""
import json

import requests
from conftest import register_and_login

import app.blueprints.assistant as assistant_bp
import app.services.ai_presets as ai
from app.models import Preset


def test_consent_banner_flow(client):
    # Banner mostrato all'anonimo, poi nascosto dopo la scelta
    assert b"cookie-banner" in client.get("/").data
    client.post("/consent", data={"action": "accept"})
    assert b"cookie-banner" not in client.get("/").data


def test_missing_api_key_flashes_config_error(client):
    register_and_login(client, "aiuser")
    r = client.post("/assistant", data={"prompt": "neon look"}, follow_redirects=True)
    assert "non è configurato".encode() in r.data


def test_generate_preset_timeout_returns_none(app, monkeypatch):
    with app.app_context():
        app.config["GROQ_API_KEY"] = "fake"

        def boom(*args, **kwargs):
            raise requests.exceptions.Timeout()

        monkeypatch.setattr(requests, "post", boom)
        assert ai.generate_preset("anything") is None


def test_generate_and_save_ai_preset(client, app, monkeypatch):
    register_and_login(client, "aiuser2")
    fake = {"blob_shape": "circular", "inner_style": "acid", "blob_color": "#ff00ff"}
    monkeypatch.setattr(assistant_bp, "generate_preset", lambda prompt: dict(fake))

    r = client.post("/assistant", data={"prompt": "neon"})
    assert b"Preset generato" in r.data

    client.post("/assistant/save", data={"name": "Neon AI", "config": json.dumps(fake)})
    with app.app_context():
        p = Preset.query.filter_by(name="Neon AI").first()
        assert p is not None and p.source == "ai"
