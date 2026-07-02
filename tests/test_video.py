"""Test dell'elaborazione VIDEO nello Studio unificato (source video → render → download)."""
import io
import os
import tempfile
import time

import cv2
import numpy as np
import pytest
from conftest import register_and_login
from test_studio import BASE  # StudioForm unificata: config completo condiviso

from engine import capabilities


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


def _video_data(**extra):
    data = dict(BASE)
    data["action"] = "render_video"
    data["source"] = (io.BytesIO(_tiny_mp4_bytes()), "clip.mp4")
    data.update(extra)
    return data


def _wait_done(client, job, tries=150):
    for _ in range(tries):
        s = client.get(f"/video/status/{job}").get_json()
        if s["state"] != "running":
            return s
        time.sleep(0.1)
    return s


def test_studio_page_has_unified_source(client):
    register_and_login(client, "vid")
    r = client.get("/studio")
    assert r.status_code == 200
    assert b"Carica immagine o video" in r.data       # sorgente unica
    assert b"Elabora e scarica video" in r.data       # azione video presente


def test_video_route_redirects_to_studio(client):
    register_and_login(client, "vidredir")
    r = client.get("/video")
    assert r.status_code == 302 and "/studio" in r.headers["Location"]


def test_video_render_sync_produces_playable_output(client, app):
    """Fallback senza JS: POST classico → pagina risultato con video e download."""
    register_and_login(client, "vid2")
    data = _video_data(trails_enabled="y")
    r = client.post("/studio", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Scarica video" in r.data
    assert b"uploads/" in r.data

    uploads = app.config["UPLOAD_FOLDER"]
    mp4s = [f for f in os.listdir(uploads) if f.endswith(".mp4") and not f.startswith("_in_")]
    assert mp4s, "nessun video di output prodotto"
    cap = cv2.VideoCapture(os.path.join(uploads, sorted(mp4s)[-1]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


def test_video_render_requires_video_source(client):
    register_and_login(client, "vid3")
    data = dict(BASE)
    data["action"] = "render_video"  # niente file sorgente
    r = client.post("/studio", data=data, content_type="multipart/form-data")
    assert r.status_code == 200
    assert b"Carica un video" in r.data


def test_video_async_job_with_progress_and_download(client, app):
    """Percorso asincrono: job id → polling → done → pagina risultato → download."""
    import re

    register_and_login(client, "vidasync")
    r = client.post("/studio", data=_video_data(), content_type="multipart/form-data",
                    headers={"X-Requested-With": "fetch"})
    assert r.status_code == 200
    job = r.get_json()["job"]

    status = _wait_done(client, job)
    assert status["state"] == "done", status
    assert status["progress"] == 1.0 and "/studio" in status["redirect"]

    page = client.get(status["redirect"])
    assert b"Scarica video" in page.data
    name = re.search(rb"uploads/([0-9a-f]+\.mp4)", page.data).group(1).decode()
    dl = client.get(f"/uploads/{name}?download=1")
    assert dl.status_code == 200 and "attachment" in dl.headers.get("Content-Disposition", "")


def test_video_job_status_is_private(client):
    register_and_login(client, "vidowner")
    job = client.post("/studio", data=_video_data(), content_type="multipart/form-data",
                      headers={"X-Requested-With": "fetch"}).get_json()["job"]
    _wait_done(client, job)  # niente job appesi fra i test

    client.get("/logout")
    register_and_login(client, "vidintruder")
    assert client.get(f"/video/status/{job}").status_code == 404


def test_video_limits_cap_resolution_and_duration(client, app):
    """Con i limiti di produzione attivi l'output è ridotto in risoluzione e durata."""
    import re

    register_and_login(client, "vidlim")
    app.config["VIDEO_MAX_SECONDS"] = 1   # sorgente: 18 frame a 12 fps
    app.config["VIDEO_MAX_DIM"] = 160     # sorgente: 320x240
    try:
        r = client.post("/studio", data=_video_data(), content_type="multipart/form-data")
        assert r.status_code == 200 and b"Scarica video" in r.data
        name = re.search(rb"uploads/([0-9a-f]+\.mp4)", r.data).group(1).decode()
        cap = cv2.VideoCapture(os.path.join(app.config["UPLOAD_FOLDER"], name))
        assert int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) == 160
        assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 120
        assert 0 < int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) <= 12  # 1s a 12 fps
    finally:
        app.config["VIDEO_MAX_SECONDS"] = 0
        app.config["VIDEO_MAX_DIM"] = 0


def test_video_busy_returns_friendly_warning(client, monkeypatch):
    """A slot pieni il server risponde 'occupato' invece di accodare il job."""
    import threading

    from app.services import video_processing as vp

    monkeypatch.setattr(vp, "_job_slots", threading.Semaphore(0))  # tutto occupato
    register_and_login(client, "vidbusy")
    r = client.post("/studio", data=_video_data(), content_type="multipart/form-data")
    assert r.status_code == 200
    assert "sta già elaborando".encode() in r.data
    assert b"Scarica video" not in r.data


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
    data = _video_data(audio_modulate_size="y")
    data["audio"] = (io.BytesIO(_tiny_wav_bytes()), "track.wav")

    r = client.post("/studio", data=data, content_type="multipart/form-data")
    assert r.status_code == 200 and b"Scarica video" in r.data

    uploads = app.config["UPLOAD_FOLDER"]
    assert not [f for f in os.listdir(uploads) if f.startswith(("_in_", "_aud_"))]
    mp4s = [f for f in os.listdir(uploads) if f.endswith(".mp4") and not f.startswith("_")]
    cap = cv2.VideoCapture(os.path.join(uploads, sorted(mp4s)[-1]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0


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

    r = client.post("/studio", data=_video_data(), content_type="multipart/form-data")
    assert r.status_code == 200 and b"Scarica video" in r.data
    assert not os.path.exists(stale)
