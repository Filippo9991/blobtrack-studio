"""Test della Live cam (Pass 3): pagina, endpoint frame, preset/snapshot."""
import base64

from conftest import register_and_login
from test_studio import BASE

from app.models import Creation, Preset


def _data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def test_live_requires_login(client):
    r = client.get("/live")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_live_page_renders(client):
    register_and_login(client, "userlive")
    r = client.get("/live")
    assert r.status_code == 200
    assert b"LIVE" in r.data
    assert b"live.js" in r.data


def test_live_frame_returns_jpeg(client, sample_png):
    register_and_login(client, "userframe")
    r = client.post(
        "/live/frame",
        json={"frame": _data_url(sample_png),
              "config": {"detection_engine": "color", "threshold_mode": "fixed"}},
    )
    assert r.status_code == 200
    assert r.mimetype == "image/jpeg"
    assert r.data[:2] == b"\xff\xd8"  # magic JPEG (SOI)


def test_live_frame_rejects_missing_frame(client):
    register_and_login(client, "usernoframe")
    r = client.post("/live/frame", json={"config": {}})
    assert r.status_code == 400


def test_live_frame_stream_keeps_state(client, sample_png):
    """Con uno stream id lo stato del motore persiste fra i frame (scie/tracking)."""
    register_and_login(client, "userstream")
    from app.services import frame_engine

    frame_engine._live_sessions.clear()  # registry globale: isola il test
    cfg = {"detection_engine": "color", "threshold_mode": "fixed", "trails_enabled": True}
    for _ in range(2):
        r = client.post(
            "/live/frame",
            json={"frame": _data_url(sample_png), "config": cfg, "stream": "s1"},
        )
        assert r.status_code == 200 and r.data[:2] == b"\xff\xd8"

    assert len(frame_engine._live_sessions) == 1
    ((key, entry),) = frame_engine._live_sessions.items()
    assert key.endswith(":s1")
    assert entry["engine"]._frame_count == 2  # nessun reset fra i frame


def test_live_frame_audio_level_modulates(client, sample_png):
    """Il livello del microfono viene accettato e modula il rendering senza errori."""
    register_and_login(client, "usermic")
    cfg = {
        "detection_engine": "color", "threshold_mode": "fixed",
        "audio_modulate_size": True, "audio_mod_intensity": 1.5,
    }
    r = client.post(
        "/live/frame",
        json={"frame": _data_url(sample_png), "config": cfg,
              "stream": "s2", "audio_level": 0.9},
    )
    assert r.status_code == 200 and r.mimetype == "image/jpeg"


def test_live_save_preset(client, app):
    register_and_login(client, "userlivepreset")
    data = dict(BASE)
    data["action"] = "save_preset"
    data["preset_name"] = "Live look"
    client.post("/live", data=data, follow_redirects=True)
    with app.app_context():
        p = Preset.query.filter_by(name="Live look").first()
        assert p is not None and p.source == "manual"


def test_live_save_snapshot(client, app, sample_png):
    register_and_login(client, "usersnap")
    data = dict(BASE)
    data["action"] = "save_snapshot"
    data["preset_name"] = "Snap 1"
    data["snapshot_raw"] = _data_url(sample_png)
    client.post("/live", data=data, follow_redirects=True)
    with app.app_context():
        c = Creation.query.filter_by(title="Snap 1").first()
        assert c is not None and c.image_data[:4] == b"\x89PNG"


def test_live_snapshot_without_frame_warns(client, app):
    register_and_login(client, "usernosnap")
    data = dict(BASE)
    data["action"] = "save_snapshot"
    data["preset_name"] = "Empty"
    r = client.post("/live", data=data, follow_redirects=True)
    assert r.status_code == 200
    with app.app_context():
        assert Creation.query.filter_by(title="Empty").first() is None


def test_live_frame_busy_returns_429(client, sample_png):
    """Un frame per utente alla volta: il secondo simultaneo riceve 429."""
    register_and_login(client, "busylive")
    from app.blueprints import studio as studio_mod

    studio_mod._live_busy_users.add(1)  # primo utente dell'app di test
    try:
        r = client.post(
            "/live/frame",
            json={"frame": _data_url(sample_png),
                  "config": {"detection_engine": "color", "threshold_mode": "fixed"}},
        )
        assert r.status_code == 429
    finally:
        studio_mod._live_busy_users.discard(1)
