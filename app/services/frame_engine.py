"""Adapter web → motore CV: elabora i byte di un'immagine caricata.

Decodifica/ridimensiona/encoda; la detection + rendering (e la gestione dello
stato) sono delegate all'API del package `engine`.
"""
import cv2
import numpy as np

from engine import ProcessingConfig, process_image_frame

MAX_DIM = 1600  # ridimensiona le immagini molto grandi (performance)


def render_image(image_bytes, config_dict):
    """Elabora un'immagine col config dato e ritorna i byte PNG.

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

    config = ProcessingConfig(**config_dict)
    rendered = process_image_frame(frame, config)

    if rendered is None or not isinstance(rendered, np.ndarray):
        raise RuntimeError("Elaborazione fallita: nessun frame restituito.")

    ok, buffer = cv2.imencode(".png", rendered)
    if not ok:
        raise RuntimeError("Encoding PNG fallito.")
    return buffer.tobytes()
