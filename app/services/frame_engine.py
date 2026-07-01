"""Adapter web → motore CV: elabora i byte di un'immagine caricata.

Decodifica/ridimensiona/encoda; la detection + rendering (e la gestione dello
stato) sono delegate all'API del package `engine`. Due encoder: PNG (Studio,
qualità lossless per la galleria) e JPEG (Live, leggero per lo streaming).

Per il Live cam esiste anche uno stato PER-SESSIONE (`render_live_frame_jpeg`):
ogni stream ha il suo LiveEngine persistente, così scie e ID di tracking si
accumulano fra i frame come nel video. Le sessioni scadono per inattività.
"""
import threading
import time

import cv2
import numpy as np

from engine import ProcessingConfig, create_live_engine, process_image_frame

MAX_DIM = 1600  # ridimensiona le immagini molto grandi (performance)

LIVE_SESSION_TTL = 300  # secondi di inattività dopo cui uno stream live scade
_live_sessions = {}
_live_registry_lock = threading.Lock()


def _decode(image_bytes):
    """Decodifica i byte in un frame BGR, ridimensionando se troppo grande."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Immagine non valida o formato non supportato.")

    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return frame


def _process(frame, config_dict):
    """Applica il config del motore al frame e ritorna il frame renderizzato."""
    config = ProcessingConfig(**config_dict)
    rendered = process_image_frame(frame, config)

    if rendered is None or not isinstance(rendered, np.ndarray):
        raise RuntimeError("Elaborazione fallita: nessun frame restituito.")
    return rendered


def render_image(image_bytes, config_dict):
    """Elabora un'immagine col config dato e ritorna i byte PNG.

    Solleva ValueError se l'immagine non è decodificabile.
    """
    rendered = _process(_decode(image_bytes), config_dict)
    ok, buffer = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("Encoding PNG fallito.")
    return buffer.tobytes()


def render_frame_jpeg(image_bytes, config_dict, quality=80):
    """Elabora un frame (Live cam) e ritorna i byte JPEG.

    Il JPEG è molto più leggero del PNG: adatto allo streaming quasi-real-time.
    """
    rendered = _process(_decode(image_bytes), config_dict)
    ok, buffer = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Encoding JPEG fallito.")
    return buffer.tobytes()


def _live_session(key):
    """Ritorna (creandola se serve) la sessione live per `key`, con eviction TTL."""
    now = time.time()
    with _live_registry_lock:
        expired = [k for k, s in _live_sessions.items() if now - s["last"] > LIVE_SESSION_TTL]
        for k in expired:
            del _live_sessions[k]
        entry = _live_sessions.get(key)
        if entry is None:
            entry = {"engine": create_live_engine(), "lock": threading.Lock(), "last": now}
            _live_sessions[key] = entry
        entry["last"] = now
        return entry


def drop_live_session(key):
    """Chiude subito uno stream (es. alla disconnessione del WebSocket)."""
    with _live_registry_lock:
        _live_sessions.pop(key, None)


def render_live_frame_jpeg(image_bytes, config_dict, session_key, audio_level=0.0, quality=80):
    """Frame live con stato persistente per-sessione (scie, tracking, mic).

    `audio_level` (0..1) è il livello del microfono calcolato nel browser:
    modula il rendering secondo i flag audio_modulate_* del config. Il lock
    per-sessione serializza i frame dello stesso stream; stream diversi
    procedono in parallelo.
    """
    frame = _decode(image_bytes)
    config = ProcessingConfig(**config_dict)
    entry = _live_session(session_key)
    with entry["lock"]:
        rendered = entry["engine"].process_frame(frame, config, mod_energy=audio_level)

    if rendered is None or not isinstance(rendered, np.ndarray):
        raise RuntimeError("Elaborazione fallita: nessun frame restituito.")
    ok, buffer = cv2.imencode(".jpg", rendered, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("Encoding JPEG fallito.")
    return buffer.tobytes()
