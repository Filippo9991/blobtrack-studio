"""Test dello Studio: elaborazione, creazioni, preset, isolamento utenti."""
import io

from conftest import register_and_login

from models import Creation, Preset

BASE = dict(
    track_mode="luminance", threshold="120", min_size="200", max_blobs="40",
    blob_shape="rectangular", blob_style="solid", blob_color="#00ff9d",
    blob_thickness="2", corner_radius="0", wf_type="linear", wf_style="solid",
    wf_color="#00ff9d", wiring_density="3", inner_style="acid",
    bg_mode="original", label_type="none",
)


def _post_studio(client, png, action, **extra):
    data = dict(BASE)
    data["action"] = action
    data.update(extra)
    if png is not None:
        data["image"] = (io.BytesIO(png), "test.png")
    return client.post(
        "/studio", data=data, content_type="multipart/form-data", follow_redirects=True
    )


def test_preview_returns_inline_image(client, sample_png):
    register_and_login(client, "userprev")
    r = _post_studio(client, sample_png, "preview")
    assert b"data:image/png;base64" in r.data


def test_save_creation_persists(client, app, sample_png):
    register_and_login(client, "usersave")
    _post_studio(client, sample_png, "save", preset_name="My art")
    with app.app_context():
        c = Creation.query.filter_by(title="My art").first()
        assert c is not None and c.image_data[:4] == b"\x89PNG"


def test_save_preset_persists(client, app, sample_png):
    register_and_login(client, "userpreset")
    _post_studio(client, None, "save_preset", preset_name="Acid")
    with app.app_context():
        p = Preset.query.filter_by(name="Acid").first()
        assert p is not None and p.source == "manual"


def test_creation_isolation_between_users(client, app, sample_png):
    register_and_login(client, "owner")
    _post_studio(client, sample_png, "save", preset_name="Secret")
    with app.app_context():
        cid = Creation.query.filter_by(title="Secret").first().id

    client.get("/logout")
    register_and_login(client, "intruder")
    assert client.get(f"/creation/{cid}/image").status_code == 404


def test_dashboard_empty_state(client):
    register_and_login(client, "empty")
    r = client.get("/dashboard")
    assert "Nessuna creazione".encode() in r.data
