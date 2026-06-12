"""Elaborazione di una singola immagine: blob detection + styling.

Adatta la pipeline 'color/threshold' del motore originale (vedi legacy/app.py) a
un input/output in memoria, così da essere chiamata direttamente da una route Flask.

    png_bytes, n_blobs = process_image(image_bytes, settings)
"""
import math

import cv2
import numpy as np

import blob_engine_core as be

# Opzioni esposte all'interfaccia (single source of truth, importate da forms.py)
TRACK_MODES = ["luminance", "red", "green", "blue", "average"]
BLOB_SHAPES = ["rectangular", "circular"]
BLOB_STYLES = ["solid", "dotted", "dashed", "corners", "brackets", "none"]
WF_TYPES = ["none", "linear", "curved"]
WF_STYLES = ["solid", "dotted", "dashed"]
INNER_STYLES = [
    "normal", "negative", "acid", "bw", "red_only", "green_only", "blue_only",
    "posterize", "edge", "thermal", "sketch", "emboss", "pixelate", "halftone",
]
BG_MODES = ["original", "black", "green"]
LABEL_TYPES = ["none", "coordinates", "index"]

MAX_DIM = 1280  # ridimensiona le immagini molto grandi (performance + memoria)

DEFAULTS = {
    "track_mode": "luminance",
    "threshold": 127,
    "min_size": 150,
    "max_blobs": 40,
    "blob_shape": "rectangular",
    "blob_style": "solid",
    "blob_color": "#00ff9d",
    "blob_thickness": 2,
    "corner_radius": 0,
    "wf_type": "none",
    "wf_style": "solid",
    "wf_color": "#00ff9d",
    "wf_thickness": 1,
    "wiring_density": 3,
    "inner_style": "normal",
    "bg_mode": "original",
    "show_center": False,
    "center_color": "#ffcc4d",
    "label_type": "none",
    "opacity": 1.0,
}


def normalize_settings(settings):
    """Unisce i settings ricevuti con i default e applica coercizione + clamping.

    Robusto a valori mancanti, di tipo sbagliato o fuori range (utile sia per i form
    sia per il JSON generato dall'AI).
    """
    s = dict(DEFAULTS)
    for key, value in (settings or {}).items():
        if key in s and value is not None and value != "":
            s[key] = value

    def as_int(key, lo, hi):
        try:
            s[key] = int(np.clip(int(float(s[key])), lo, hi))
        except (TypeError, ValueError):
            s[key] = DEFAULTS[key]

    as_int("threshold", 0, 255)
    as_int("min_size", 1, 100000)
    as_int("max_blobs", 1, 200)
    as_int("blob_thickness", 1, 8)
    as_int("wf_thickness", 1, 8)
    as_int("wiring_density", 1, 20)
    as_int("corner_radius", 0, 60)

    try:
        s["opacity"] = float(np.clip(float(s["opacity"]), 0.1, 1.0))
    except (TypeError, ValueError):
        s["opacity"] = DEFAULTS["opacity"]

    s["show_center"] = str(s["show_center"]).lower() in ("true", "1", "yes", "on")

    # I campi enum vengono riportati ai default se contengono valori sconosciuti
    for key, allowed in (
        ("track_mode", TRACK_MODES), ("blob_shape", BLOB_SHAPES),
        ("blob_style", BLOB_STYLES), ("wf_type", WF_TYPES), ("wf_style", WF_STYLES),
        ("inner_style", INNER_STYLES), ("bg_mode", BG_MODES), ("label_type", LABEL_TYPES),
    ):
        if s[key] not in allowed:
            s[key] = DEFAULTS[key]

    return s


def _detect_blobs(frame, s):
    shape = s["blob_shape"]
    gray = be.get_channel(frame, s["track_mode"])
    _, thresh = cv2.threshold(gray, s["threshold"], 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    blobs = []
    for c in contours:
        if cv2.contourArea(c) < s["min_size"]:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        cx, cy = x + bw // 2, y + bh // 2
        if shape == "circular":
            r = max(bw, bh) // 2 + 5
            blobs.append((cx, cy, cx - r, cy - r, cx + r, cy + r, r))
        else:
            blobs.append((cx, cy, x - 5, y - 5, x + bw + 5, y + bh + 5, 0))

    # I blob più grandi per primi, poi tagliamo a max_blobs
    blobs.sort(key=lambda b: (b[4] - b[2]) * (b[5] - b[3]), reverse=True)
    return blobs[: s["max_blobs"]]


def _fill_inner(base_img, inner, blobs, s, w, h):
    shape = s["blob_shape"]
    corner_radius = s["corner_radius"]
    for cx, cy, x1, y1, x2, y2, r in blobs:
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue
        roi_mask = np.zeros((rh, rw), dtype=np.uint8)
        if shape == "circular":
            cv2.circle(roi_mask, (cx - x1, cy - y1), r, 255, -1)
        elif corner_radius > 0:
            er = min(corner_radius, rw // 2, rh // 2)
            cv2.rectangle(roi_mask, (er, 0), (rw - er, rh), 255, -1)
            cv2.rectangle(roi_mask, (0, er), (rw, rh - er), 255, -1)
            cv2.ellipse(roi_mask, (er, er), (er, er), 180, 0, 90, 255, -1)
            cv2.ellipse(roi_mask, (rw - er, er), (er, er), 270, 0, 90, 255, -1)
            cv2.ellipse(roi_mask, (rw - er, rh - er), (er, er), 0, 0, 90, 255, -1)
            cv2.ellipse(roi_mask, (er, rh - er), (er, er), 90, 0, 90, 255, -1)
        else:
            roi_mask[:] = 255
        base_img[y1:y2, x1:x2][roi_mask > 0] = inner[y1:y2, x1:x2][roi_mask > 0]


def process_image(image_bytes, settings=None):
    """Rileva i blob in un'immagine e applica lo stile. Ritorna (png_bytes, n_blobs)."""
    s = normalize_settings(settings)

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Immagine non valida o formato non supportato.")

    h, w = frame.shape[:2]
    if max(h, w) > MAX_DIM:
        scale = MAX_DIM / max(h, w)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        h, w = frame.shape[:2]

    frame_clean = frame.copy()
    shape = s["blob_shape"]
    blobs = _detect_blobs(frame, s)

    # Sfondo
    if s["bg_mode"] == "green":
        base_img = np.full_like(frame_clean, (0, 255, 0))
    elif s["bg_mode"] == "black":
        base_img = np.zeros_like(frame_clean)
    else:
        base_img = frame_clean.copy()

    graphics_img = np.zeros_like(frame_clean)
    graphics_mask = np.zeros(frame_clean.shape[:2], dtype=np.uint8)

    # Riempimento interno dei blob con il filtro scelto
    inner = be.apply_inner_style(frame_clean, s["inner_style"])
    _fill_inner(base_img, inner, blobs, s, w, h)

    blob_col = be.hex_to_bgr(s["blob_color"])
    wf_col = be.hex_to_bgr(s["wf_color"])
    center_col = be.hex_to_bgr(s["center_color"])

    # Wireframe: collega ogni blob ai vicini più prossimi
    if s["wf_type"] != "none" and len(blobs) > 1:
        density = s["wiring_density"]
        for i, b1 in enumerate(blobs):
            neighbours = sorted(
                ((math.hypot(b1[0] - b2[0], b1[1] - b2[1]), b2)
                 for j, b2 in enumerate(blobs) if j != i),
                key=lambda t: t[0],
            )
            for _, b2 in neighbours[:density]:
                p1 = be.get_intersection(b1[0], b1[1], b1[6], b1[2], b1[3], b1[4], b1[5],
                                         b2[0], b2[1], shape, b1[4] - b1[2], b1[5] - b1[3])
                p2 = be.get_intersection(b2[0], b2[1], b2[6], b2[2], b2[3], b2[4], b2[5],
                                         b1[0], b1[1], shape, b2[4] - b2[2], b2[5] - b2[3])
                be.draw_line_custom(graphics_img, graphics_mask, p1, p2, s["wf_type"],
                                    s["wf_style"], wf_col, s["wf_thickness"], 20, "none")

    # Contorni, centri ed etichette
    for idx, b in enumerate(blobs):
        be.draw_blob_shape(graphics_img, graphics_mask, b, shape, s["blob_style"],
                           blob_col, s["blob_thickness"], s["corner_radius"], 10)
        if s["show_center"]:
            be.draw_center_custom(graphics_img, graphics_mask, b[0], b[1], center_col,
                                 "circle", "filled", s["blob_thickness"], 1)
        be.draw_label(graphics_img, graphics_mask, b, s["label_type"], "", blob_col,
                     shape, "regular", "top", blob_index=idx)

    # Composizione finale
    mask3 = cv2.merge([graphics_mask, graphics_mask, graphics_mask])
    if s["opacity"] >= 0.99:
        final = np.where(mask3 > 0, graphics_img, base_img)
    else:
        blended = cv2.addWeighted(base_img, 1.0, graphics_img, s["opacity"], 0)
        final = np.where(mask3 > 0, blended, base_img)

    ok, buffer = cv2.imencode(".png", final)
    if not ok:
        raise RuntimeError("Encoding PNG fallito.")
    return buffer.tobytes(), len(blobs)
