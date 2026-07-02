"""Test della sezione MediaPipe (pagina dedicata, separata dal Live)."""
import base64

import pytest
from conftest import register_and_login

from engine import capabilities
from app.models import Preset

mediapipe_only = pytest.mark.skipif(
    not capabilities()["mediapipe"], reason="richiede mediapipe (profilo full)"
)


def _data_url(png_bytes):
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


# Config MediaPipe minimo: detection_engine='mediapipe' + i campi mp_* e stile.
MP_BASE = dict(
    detection_engine="mediapipe",
    mp_pose_enabled="y", mp_confidence="0.5", mp_num_poses="4", mp_pose_num_points="6",
    mp_hands_num_points="5", mp_num_faces="2", mp_face_num_points="7",
    mp_blob_size="1.0", mp_merge_distance="0",
    # stile
    blob_shape="circular", blob_style="solid", blob_color="#00ff9d", blob_thickness="2",
    corner_radius="0", blob_dot_gap="10",
    wf_type="linear", wf_style="solid", wf_color="#00ff9d", wf_thickness="1",
    wf_dot_gap="20", wiring_density="5", end_cap="none",
    center_color="#ffff00", center_shape="circle", center_style="filled", center_size_level="1",
    label_type="none", text_color="#ffffff", custom_text="REC", font_weight="regular",
    label_pos="bottom", text_size="0.6", text_outline_color="#000000",
    inner_style="normal", bg_mode="original", opacity="1.0",
    glow_intensity="1.0", glow_radius="21",
    # motion + i campi detection nascosti (default validi)
    smoothing="5", persistence="30", tracker_match_radius="150",
    trail_length="20", trail_opacity="0.6", trail_style="line",
    threshold="127", threshold_mode="fixed", track_mode="luminance",
    color_target_hex="#ff0000", color_target_tolerance="30", morph_kernel_size="3",
    min_blob_size="100", max_blob_size="50000", max_blobs="20",
    edge_low="50", edge_high="150", preprocess_method="CrowdBoost", preprocess_strength="1.0",
    yolo_model_file="yolov8n.pt",
)


@mediapipe_only
def test_mediapipe_page_renders(client):
    register_and_login(client, "mpuser")
    r = client.get("/mediapipe")
    assert r.status_code == 200
    assert b"MEDIAPIPE" in r.data
    assert b"live.js" in r.data            # riusa il motore webcam del Live
    assert b'value="mediapipe"' in r.data  # detection_engine nascosto


# La sezione MediaPipe nella sidebar ha questo titolo; il link in navbar no.
_MP_SECTION = b'section-title">MediaPipe'


def test_live_page_has_no_mediapipe_section(client):
    """MediaPipe è stato spostato fuori dal Live (niente conflitto di blob)."""
    register_and_login(client, "livenomp")
    r = client.get("/live")
    assert r.status_code == 200
    assert _MP_SECTION not in r.data
    assert b"mp_pose_enabled" not in r.data  # nessun controllo MediaPipe nel form


def test_studio_page_has_no_mediapipe_section(client):
    register_and_login(client, "studionomp")
    r = client.get("/studio")
    assert r.status_code == 200
    assert _MP_SECTION not in r.data
    assert b"mp_pose_enabled" not in r.data


@mediapipe_only
def test_mediapipe_frame_returns_jpeg(client, sample_png):
    """Un frame in modalità mediapipe viene elaborato e torna JPEG (no color blobs)."""
    register_and_login(client, "mpframe")
    r = client.post(
        "/live/frame",
        json={"frame": _data_url(sample_png), "config": dict(MP_BASE)},
    )
    assert r.status_code == 200 and r.mimetype == "image/jpeg"
    assert r.data[:2] == b"\xff\xd8"


@mediapipe_only
def test_mediapipe_save_preset(client, app):
    """Il salvataggio preset dalla pagina MediaPipe funziona (form classico CSRF)."""
    register_and_login(client, "mppreset")
    data = dict(MP_BASE)
    data["action"] = "save_preset"
    data["preset_name"] = "Corpo neon"
    client.post("/mediapipe", data=data, follow_redirects=True)
    with app.app_context():
        p = Preset.query.filter_by(name="Corpo neon").first()
        assert p is not None and p.source == "manual"
        import json
        assert json.loads(p.config)["detection_engine"] == "mediapipe"
