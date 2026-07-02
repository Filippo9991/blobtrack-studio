"""Test delle rifiniture: export/import preset (JSON) e ricerca/paginazione galleria."""
import io
import json

from conftest import register_and_login
from test_studio import BASE, _post_studio

from app.models import Creation, Preset


def _make_preset(client, name):
    _post_studio(client, None, "save_preset", preset_name=name)


# --- Export / import preset ------------------------------------------------

def test_export_single_preset_is_json_attachment(client):
    register_and_login(client, "exp1")
    _make_preset(client, "Neon")
    with client.application.app_context():
        pid = Preset.query.filter_by(name="Neon").first().id

    r = client.get(f"/preset/{pid}/export")
    assert r.status_code == 200
    assert r.mimetype == "application/json"
    assert "attachment" in r.headers.get("Content-Disposition", "")
    payload = json.loads(r.data)
    assert payload["type"] == "presets"
    assert payload["presets"][0]["name"] == "Neon"
    assert "inner_style" in payload["presets"][0]["config"]


def test_export_all_presets(client):
    register_and_login(client, "expall")
    _make_preset(client, "Uno")
    _make_preset(client, "Due")
    r = client.get("/presets/export")
    assert r.status_code == 200
    names = {p["name"] for p in json.loads(r.data)["presets"]}
    assert {"Uno", "Due"} <= names


def test_export_all_empty_redirects_with_warning(client):
    register_and_login(client, "expempty")
    r = client.get("/presets/export", follow_redirects=True)
    assert "Non hai preset da esportare".encode() in r.data


def test_import_roundtrip(client, app):
    """Esporto un preset, lo reimporto: viene ricreato con la stessa config."""
    register_and_login(client, "roundtrip")
    _make_preset(client, "Originale")
    exported = client.get("/presets/export").data

    r = client.post(
        "/presets/import",
        data={"file": (io.BytesIO(exported), "presets.json")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Importati 1 preset".encode() in r.data
    with app.app_context():
        # ora esistono due preset con lo stesso nome (originale + importato)
        assert Preset.query.filter_by(name="Originale").count() == 2


def test_import_bare_config_object(client, app):
    """Accetta anche un singolo oggetto con 'config' (non solo il wrapper)."""
    register_and_login(client, "bareimport")
    blob = json.dumps({"name": "Custom", "config": {"inner_style": "thermal", "blob_color": "#ff0000"}})
    r = client.post(
        "/presets/import",
        data={"file": (io.BytesIO(blob.encode()), "p.json")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "Importati 1 preset".encode() in r.data
    with app.app_context():
        p = Preset.query.filter_by(name="Custom").first()
        assert p is not None and json.loads(p.config)["inner_style"] == "thermal"


def test_import_invalid_json_rejected(client, app):
    register_and_login(client, "badimport")
    r = client.post(
        "/presets/import",
        data={"file": (io.BytesIO(b"non-json {{{"), "p.json")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert "non è un JSON valido".encode() in r.data
    with app.app_context():
        assert Preset.query.count() == 0


def test_import_rejects_non_json_extension(client):
    register_and_login(client, "extimport")
    r = client.post(
        "/presets/import",
        data={"file": (io.BytesIO(b"{}"), "p.txt")},
        content_type="multipart/form-data",
        follow_redirects=True,
    )
    assert b"Solo file JSON" in r.data


# --- Ricerca + paginazione galleria ----------------------------------------

def _save_creations(client, titles):
    for t in titles:
        _post_studio(client, _png(), "save", preset_name=t)


def _png():
    import cv2
    import numpy as np
    img = np.zeros((60, 80, 3), np.uint8)
    cv2.circle(img, (40, 30), 15, (255, 255, 255), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def test_gallery_search_filters_by_title(client):
    register_and_login(client, "search")
    _save_creations(client, ["Tramonto rosso", "Alba blu", "Tramonto viola"])
    r = client.get("/dashboard?q=tramonto")
    assert r.status_code == 200
    assert b"Tramonto rosso" in r.data and b"Tramonto viola" in r.data
    assert b"Alba blu" not in r.data


def test_gallery_search_no_results_empty_state(client):
    register_and_login(client, "noresult")
    _save_creations(client, ["Uno"])
    r = client.get("/dashboard?q=inesistente")
    assert b"Nessun risultato" in r.data


def test_gallery_pagination(client, app, monkeypatch):
    from app.blueprints import studio as studio_mod

    monkeypatch.setattr(studio_mod, "GALLERY_PER_PAGE", 3)
    register_and_login(client, "paged")
    _save_creations(client, [f"Opera {i}" for i in range(5)])

    p1 = client.get("/dashboard?page=1")
    assert b"Pagina 1 di 2" in p1.data
    assert p1.data.count(b"tile-title") == 3  # per_page rispettato
    p2 = client.get("/dashboard?page=2")
    assert p2.data.count(b"tile-title") == 2
