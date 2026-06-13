"""Liste di scelta per i SelectField dello Studio (single source of truth).

Riflettono i valori supportati dal motore (blob_engine.py / schemas.ProcessingConfig).
"""

DETECTION_ENGINES = ["color", "yolo"]
YOLO_MODELS = ["yolov8n.pt", "yolov8m.pt", "yolov8x.pt"]
TRACK_MODES = [
    "luminance", "red", "green", "blue", "average",
    "hsv_hue", "hsv_saturation", "hsv_value",
    "lab_lightness", "lab_a", "lab_b", "color_target",
]
THRESHOLD_MODES = ["adaptive", "fixed", "otsu"]
PREPROCESS_METHODS = ["CrowdBoost", "CLAHE", "DetailEnhance", "Sharpen", "Gamma"]

BLOB_SHAPES = ["circular", "rectangular"]
BLOB_STYLES = ["solid", "dotted", "dashed", "corners", "brackets", "segments_4", "segments_2", "neon", "none"]
WF_TYPES = ["linear", "curved", "none"]
WF_STYLES = ["solid", "dotted", "dashed"]
END_CAPS = ["none", "circle", "arrow", "both_circles", "both_arrows"]

CENTER_SHAPES = ["circle", "square"]
CENTER_STYLES = ["filled", "outline"]

LABEL_TYPES = ["none", "coordinates", "index", "id", "area", "distance", "text"]
FONT_WEIGHTS = ["light", "regular", "bold"]
LABEL_POSITIONS = ["top", "center", "bottom"]

INNER_STYLES = [
    "normal", "negative", "acid", "bw", "red_only", "green_only", "blue_only",
    "ascii", "posterize", "edge", "thermal", "chromatic", "scanlines",
    "halftone", "pixelate", "emboss", "sketch", "vhs", "glitch", "infrared",
]
BG_MODES = ["original", "black", "green"]
TRAIL_STYLES = ["line", "dots", "fade"]
AUDIO_BANDS = ["bass", "low_mids", "high_mids", "highs", "full"]


def choices(values):
    """(value, label) per i SelectField — label leggibile in maiuscolo."""
    return [(v, v.replace("_", " ").replace(".pt", "").upper()) for v in values]
