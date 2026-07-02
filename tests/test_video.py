"""Test dell'elaborazione VIDEO (upload → render → file servibile)."""
import io
import os
import tempfile

import cv2
import numpy as np
import pytest
from conftest import register_and_login

from engine import capabilities

# Config completo VideoForm (config base + campi temporali), detection color = veloce
VBASE = dict(
    detection_engine="color", yolo_model_file="yolov8n.pt", track_mode="luminance",
    threshold="120", threshold_mode="fixed", color_target_hex="#ff0000",
    color_target_tolerance="30", morph_kernel_size="3", edge_low="50", edge_high="150",
    min_blob_size="150", max_blob_size="50000", max_blobs="20",
    preprocess_method="CrowdBoost", preprocess_strength="1.0",
    blob_shape="circular", blob_color="#00ff9d", blob_thickness="2", blob_style="dotted",
    corner_radius="0", blob_dot_gap="10",
    wf_type="linear", wf_color="#00ff9d", wf_thickness="1", wf_style="solid",
    wf_dot_gap="20", wiring_density="5", end_cap="none",
    center_color="#ffff00", center_shape="circle", center_style="filled", center_size_level="1",
    label_type="none", text_color="#ffffff", custom_text="REC", font_weight="regular",
    label_pos="bottom", text_size="0.6", text_outline_color="#000000",
    inner_style="acid", bg_mode="black", opacity="1.0",
    glow_intensity="1.0", glow_radius="21",
    mp_confidence="0.5", mp_num_poses="4", mp_pose_num_points="6", mp_hands_num_points="5",
    mp_num_faces="2", mp_face_num_points="7", mp_blob_size="1.0", mp_merge_distance="0",
    # campi video
    frame_skip="1", smoothing="5", persistence="30", tracker_match_radius="150",
    trail_length="10", trail_opacity="0.6", trail_style="fade",
    # campi audio (default; la traccia è opzionale)
    audio_band="bass", audio_sensitivity="1.0", audio_offset="0.0", audio_mod_intensity="1.0",
)


def _tiny_mp4_bytes():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "src.mp4")
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 12, (320, 240))
    for i in range(18):
        f = np.full((240, 320, 3), 18, np.uint8)
        cv2.circle(f, (40 + i * 12, 120), 26, (255, 255, 255), -1)
        vw.write(f)
    vw.release()
    with open(path, "rb") as fh:
        return fh.read()


def test_video_get_page(client):
    register_and_login(client, "vid")
    r = client.get("/video")
    assert r.status_code == 200 and b"Elabora video" in r.data


def test_video_processing_produces_playable_output(client, app):
    register_and_login(client, "vid2")
    data = dict(VBASE)
    data["trails_enabled"] = "y"
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")

    r = client.post("/video", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Video elaborato" in r.data
    assert b"uploads/" in r.data  # link al video risultante

    # il file output esiste in static/uploads ed è leggibile come video
    uploads = app.config["UPLOAD_FOLDER"]
    mp4s = [f for f in os.listdir(uploads) if f.endswith(".mp4") and not f.startswith("_in_")]
    assert mp4s, "nessun video di output prodotto"
    cap = cv2.VideoCapture(os.path.join(uploads, sorted(mp4s)[-1]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


def test_video_requires_file(client):
    register_and_login(client, "vid3")
    r = client.post("/video", data=dict(VBASE), content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Carica un video" in r.data


def test_video_limits_cap_resolution_and_duration(client, app):
    """Con i limiti di produzione attivi l'output è ridotto in risoluzione e durata."""
    import re

    register_and_login(client, "vidlim")
    app.config["VIDEO_MAX_SECONDS"] = 1   # sorgente: 18 frame a 12 fps
    app.config["VIDEO_MAX_DIM"] = 160     # sorgente: 320x240
    try:
        data = dict(VBASE)
        data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
        r = client.post("/video", data=data, content_type="multipart/form-data")
        assert r.status_code == 200 and b"Video elaborato" in r.data

        match = re.search(rb"uploads/([0-9a-f]+\.mp4)", r.data)
        assert match, "nome del video di output non trovato nella pagina"
        out = os.path.join(app.config["UPLOAD_FOLDER"], match.group(1).decode())
        cap = cv2.VideoCapture(out)
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
        assert 0 < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 12  # 1s a 12 fps
    finally:
        app.config["VIDEO_MAX_SECONDS"] = 0
        app.config["VIDEO_MAX_DIM"] = 0


def _tiny_wav_bytes():
    """Mezzo secondo di sinusoide pulsata (genera onset rilevabili)."""
    import math
    import struct
    import wave

    buf = io.BytesIO()
    w = wave.open(buf, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(22050)
    for i in range(11025):
        env = 1.0 if (i // 2205) % 2 == 0 else 0.05  # pulsa ogni 0.1s
        w.writeframes(struct.pack("<h", int(8000 * env * math.sin(2 * math.pi * 110 * i / 22050))))
    w.close()
    return buf.getvalue()


@pytest.mark.skipif(not capabilities()["audio"], reason="richiede librosa (profilo full)")
def test_video_audio_reactivity_produces_output(client, app):
    """Con una traccia audio caricata l'elaborazione va a buon fine (beat + mux)."""
    register_and_login(client, "vidaudio")
    data = dict(VBASE)
    data["audio_modulate_size"] = "y"
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    data["audio"] = (io.BytesIO(_tiny_wav_bytes()), "track.wav")

    r = client.post("/video", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Video elaborato" in r.data

    uploads = app.config["UPLOAD_FOLDER"]
    # niente file temporanei audio/input lasciati indietro
    assert not [f for f in os.listdir(uploads) if f.startswith(("_in_", "_aud_"))]
    mp4s = [f for f in os.listdir(uploads) if f.endswith(".mp4") and not f.startswith("_")]
    cap = cv2.VideoCapture(os.path.join(uploads, sorted(mp4s)[-1]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


def test_video_busy_returns_friendly_warning(client, monkeypatch):
    """A slot pieni il server risponde 'occupato' invece di accodare il job."""
    import threading

    from app.services import video_processing as vp

    monkeypatch.setattr(vp, "_job_slots", threading.Semaphore(0))  # tutto occupato
    register_and_login(client, "vidbusy")
    data = dict(VBASE)
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    r = client.post("/video", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert "sta già elaborando".encode() in r.data
    assert b"Video elaborato" not in r.data


def test_uploads_route_requires_login(client):
    r = client.get("/uploads/qualcosa.mp4")
    assert r.status_code == 302 and "/login" in r.headers["Location"]


def test_uploads_cleanup_removes_stale_files(client, app):
    """I file più vecchi della ritenzione vengono eliminati all'avvio di un job."""
    register_and_login(client, "vidclean")
    uploads = app.config["UPLOAD_FOLDER"]
    os.makedirs(uploads, exist_ok=True)
    stale = os.path.join(uploads, "vecchio.mp4")
    with open(stale, "wb") as fh:
        fh.write(b"x")
    os.utime(stale, (0, 0))  # epoca 1970: ben oltre la ritenzione

    data = dict(VBASE)
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    r = client.post("/video", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Video elaborato" in r.data
    assert not os.path.exists(stale)


def test_video_async_job_with_progress_and_download(client, app):
    """Percorso asincrono: job id → polling → done → pagina risultato → download."""
    import re
    import time as _time

    register_and_login(client, "vidasync")
    data = dict(VBASE)
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    r = client.post("/video", data=data, content_type="multipart/form-data",
                    headers={"X-Requested-With": "fetch"})
    assert r.status_code == 200
    job = r.get_json()["job"]

    status = None
    for _ in range(150):  # il clip di test si elabora in ~1s
        status = client.get(f"/video/status/{job}").get_json()
        if status["state"] != "running":
            break
        _time.sleep(0.1)
    assert status["state"] == "done", status
    assert status["progress"] == 1.0 and "redirect" in status

    page = client.get(status["redirect"])
    assert b"Scarica video" in page.data

    name = re.search(rb"uploads/([0-9a-f]+\.mp4)", page.data).group(1).decode()
    dl = client.get(f"/uploads/{name}?download=1")
    assert dl.status_code == 200
    assert "attachment" in dl.headers.get("Content-Disposition", "")


def test_video_job_status_is_private(client):
    """Lo stato del job è consultabile solo dal proprietario."""
    import time as _time

    register_and_login(client, "vidowner")
    data = dict(VBASE)
    data["video"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    job = client.post("/video", data=data, content_type="multipart/form-data",
                      headers={"X-Requested-With": "fetch"}).get_json()["job"]
    for _ in range(150):  # lascialo finire: niente job appesi fra i test
        if client.get(f"/video/status/{job}").get_json()["state"] != "running":
            break
        _time.sleep(0.1)

    client.get("/logout")
    register_and_login(client, "vidintruder")
    assert client.get(f"/video/status/{job}").status_code == 404


def test_creation_image_download_disposition(client, app):
    """?download=1 forza l'attachment; senza, l'immagine resta inline."""
    from app.extensions import db as _db
    from app.models import Creation, User

    register_and_login(client, "dlimg")
    with app.app_context():
        uid = User.query.filter_by(username="dlimg").first().id
        c = Creation(user_id=uid, title="Opera Uno", image_data=b"\x89PNG\r\n", settings="{}")
        _db.session.add(c)
        _db.session.commit()
        cid = c.id

    r = client.get(f"/creation/{cid}/image?download=1")
    assert "attachment" in r.headers.get("Content-Disposition", "")
    assert 'filename="Opera_Uno.png"' in r.headers["Content-Disposition"]
    r2 = client.get(f"/creation/{cid}/image")
    assert "Content-Disposition" not in r2.headers
