"""Wrapper sul motore completo di BlobTrack per l'elaborazione di una singola immagine.

Riusa `LiveEngine.process_frame(frame, config)` (blob_engine.py), il punto d'ingresso
single-frame che esegue detection (color/YOLO/MediaPipe/silhouette/edge) + rendering
completo. L'engine è un singleton lazy: carica YOLO/MediaPipe alla prima richiesta e li
tiene in memoria (così le richieste successive sono veloci).
"""
import threading

import cv2
import numpy as np

MAX_DIM = 1600  # ridimensiona le immagini molto grandi (performance)

_engine = None
_engine_lock = threading.Lock()


def get_engine():
    """Ritorna il singleton LiveEngine (import pesante differito)."""
    global _engine
    if _engine is None:
        with _engine_lock:
            if _engine is None:
                from blob_engine import LiveEngine  # import lazy: carica torch/mediapipe

                _engine = LiveEngine()
    return _engine


def render_image(image_bytes, config_dict):
    """Elabora un'immagine col config dato. Ritorna (png_bytes, n_blobs).

    Solleva ValueError se l'immagine non è decodificabile.
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Immagine non valida o formato non supportato.")

    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    from schemas import ProcessingConfig

    config = ProcessingConfig(**config_dict)

    engine = get_engine()
    # L'engine ha stato interno (modelli, tracker): serializziamo le chiamate.
    with _engine_lock:
        rendered = engine.process_frame(frame, config)

    if rendered is None or not isinstance(rendered, np.ndarray):
        raise RuntimeError("Elaborazione fallita: nessun frame restituito.")

    ok, buffer = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("Encoding PNG fallito.")

    # n_blobs non è restituito da process_frame: indichiamo solo il successo
    return buffer.tobytes(), None
