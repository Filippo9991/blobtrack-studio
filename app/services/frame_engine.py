"""Adapter web → motore CV: elabora i byte di un'immagine caricata.

Decodifica/ridimensiona/encoda; la detection + rendering (e la gestione dello
stato) sono delegate all'API del package `engine`. Due encoder: PNG (Studio,
qualità lossless per la galleria) e JPEG (Live, leggero per lo streaming).
"""
import cv2
import numpy as np

from engine import ProcessingConfig, process_image_frame

MAX_DIM = 1600  # ridimensiona le immagini molto grandi (performance)


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
