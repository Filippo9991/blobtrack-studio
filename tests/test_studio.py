"""Test dello Studio: elaborazione, creazioni, preset, isolamento utenti."""
import io

from conftest import register_and_login

from models import Creation, Preset

# Set completo di campi della StudioForm (valori validi, detection color = veloce)
BASE = dict(
    detection_engine="color", yolo_model_file="yolov8n.pt", track_mode="luminance",
    threshold="120", threshold_mode="adaptive", color_target_hex="#ff0000",
    color_target_tolerance="30", morph_kernel_size="3", edge_low="50", edge_high="150",
    min_blob_size="100", max_blob_size="50000", max_blobs="20",
    preprocess_method="CrowdBoost", preprocess_strength="1.0",
    blob_shape="circular", blob_color="#ffffff", blob_thickness="2", blob_style="solid",
    corner_radius="0", blob_dot_gap="10",
    wf_type="linear", wf_color="#ffffff", wf_thickness="1", wf_style="solid",
    wf_dot_gap="20", wiring_density="5", end_cap="none",
    center_color="#ffff00", center_shape="circle", center_style="filled", center_size_level="1",
    label_type="none", text_color="#ffffff", custom_text="REC", font_weight="regular",
    label_pos="bottom", text_size="0.6", text_outline_color="#000000",
    inner_style="acid", bg_mode="original", opacity="1.0",
    glow_intensity="1.0", glow_radius="21",
    mp_confidence="0.5", mp_num_poses="4", mp_pose_num_points="6", mp_hands_num_points="5",
    mp_num_faces="2", mp_face_num_points="7", mp_blob_size="1.0", mp_merge_distance="0",
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


def test_save_preset_persists_full_config(client, app, sample_png):
    register_and_login(client, "userpreset")
    _post_studio(client, None, "save_preset", preset_name="Acid")
    with app.app_context():
        p = Preset.query.filter_by(name="Acid").first()
        assert p is not None and p.source == "manual"
        import json
        cfg = json.loads(p.config)
        assert cfg["inner_style"] == "acid"
        assert "wf_type" in cfg and "glow_radius" in cfg


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
