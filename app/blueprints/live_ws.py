"""Live cam via WebSocket (flask-sock) — trasporto alternativo a /live/frame.

Il client manda messaggi JSON {frame, config, audio_level} e riceve i byte JPEG
del frame elaborato (messaggio binario). Rispetto al polling HTTP evita un
handshake per frame: in locale il framerate sale sensibilmente. Il client
(live.js) prova prima il WebSocket e ripiega sul polling se non disponibile.

Lo stato di tracking vive per-connessione: ogni socket ha il suo LiveEngine
(scie e ID stabili), scartato alla disconnessione.
"""
import json
import uuid

from flask import session

from app.extensions import sock
from app.services.frame_engine import drop_live_session, render_live_frame_jpeg


def _decode_data_url(data_url):
    """Estrae i byte da un data URL. None se invalido (copia locale, no import ciclici)."""
    import base64

    if not data_url or "," not in data_url:
        return None
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def _clamp_audio_level(value):
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


@sock.route("/live/ws")
def live_ws(ws):
    """Un LiveEngine per connessione; loop finché il client non chiude."""
    if not session.get("user_id"):
        ws.close(reason=4401, message="Login richiesto.")
        return

    session_key = f"{session['user_id']}:ws-{uuid.uuid4().hex}"
    try:
        while True:
            message = ws.receive()  # solleva ConnectionClosed alla chiusura

            # Il client tiene UNA richiesta in volo: a ogni messaggio deve sempre
            # arrivare una risposta. Frame saltato = messaggio di testo "{}".
            try:
                data = json.loads(message)
            except (TypeError, ValueError):
                ws.send("{}")
                continue

            raw = _decode_data_url(data.get("frame"))
            if not raw:
                ws.send("{}")
                continue

            try:
                jpeg = render_live_frame_jpeg(
                    raw,
                    data.get("config") or {},
                    session_key,
                    audio_level=_clamp_audio_level(data.get("audio_level")),
                )
            except Exception:
                # Config invalido o errore del motore: salta il frame, non chiudere
                ws.send("{}")
                continue
            ws.send(jpeg)
    finally:
        drop_live_session(session_key)
