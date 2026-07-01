"""Genera le immagini demo del README: sorgente sintetica + 3 stili del motore.

Uso (dalla root del repo):  venv/bin/python docs/demo/generate.py
È anche un esempio minimale di come usare l'API pubblica del package `engine`.
"""
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.getcwd())
from engine import ProcessingConfig, process_image_frame  # noqa: E402

OUT = os.path.join(os.getcwd(), "docs", "demo")
os.makedirs(OUT, exist_ok=True)

W, H = 960, 640
rng = np.random.default_rng(42)


def make_source():
    """Immagine 'bokeh notturno': cerchi luminosi definiti su fondo scuro."""
    img = np.zeros((H, W, 3), dtype=np.float32)
    palette = [
        (60, 200, 255),   # ambra (BGR)
        (200, 120, 255),  # rosa
        (255, 200, 80),   # ciano
        (120, 255, 160),  # verde acido
        (255, 120, 200),  # viola
        (80, 160, 255),   # arancio
    ]
    # alone soffuso di fondo
    glowbg = np.zeros_like(img)
    for _ in range(6):
        center = (int(rng.uniform(0, W)), int(rng.uniform(0, H)))
        color = palette[int(rng.integers(len(palette)))]
        cv2.circle(glowbg, center, int(rng.uniform(140, 240)), color, -1)
    img += cv2.GaussianBlur(glowbg, (0, 0), sigmaX=60) * 0.25

    # bokeh: cerchi definiti, blur leggero, dimensioni variabili
    for radius_range, sigma, alpha, count in [
        ((45, 85), 9, 0.75, 9),
        ((22, 45), 5, 0.9, 12),
        ((9, 20), 2.5, 1.0, 14),
    ]:
        layer = np.zeros_like(img)
        for _ in range(count):
            center = (int(rng.uniform(30, W - 30)), int(rng.uniform(30, H - 30)))
            color = np.array(palette[int(rng.integers(len(palette)))], dtype=np.float32)
            brightness = rng.uniform(0.65, 1.0)
            cv2.circle(layer, center, int(rng.uniform(*radius_range)), (color * brightness).tolist(), -1)
        layer = cv2.GaussianBlur(layer, (0, 0), sigmaX=sigma)
        img = np.maximum(img, layer * alpha)
    img = np.clip(img, 0, 245).astype(np.uint8)
    # vignettatura leggera
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vign = np.clip(1.1 - 0.35 * d, 0.6, 1.0)[..., None]
    return (img * vign).astype(np.uint8)


COMMON = dict(
    detection_engine="color", threshold_mode="fixed",
    min_blob_size=200, max_blob_size=400000,
)

STYLES = {
    "wireframe": dict(
        COMMON, track_mode="luminance", threshold=70, max_blobs=30,
        blob_shape="circular", blob_style="neon", blob_color="#00ff9d",
        blob_thickness=2, wf_type="curved", wf_style="solid",
        wf_color="#00ff9d", wiring_density=6,
        show_center=True, center_color="#ffffff",
        inner_style="bw", bg_mode="black", glow_enabled=True,
    ),
    "acid": dict(
        COMMON, track_mode="luminance", threshold=60, max_blobs=24,
        blob_shape="rectangular", blob_style="brackets", blob_color="#ffffff",
        blob_thickness=2, wf_type="linear", wf_style="dotted",
        wf_color="#ffffff", wiring_density=4,
        inner_style="acid", bg_mode="original", glow_enabled=False,
        label_type="area", label_color="#ffffff",
    ),
    "thermal": dict(
        COMMON, track_mode="luminance", threshold=80, max_blobs=16,
        blob_shape="circular", blob_style="segments_4", blob_color="#ff9500",
        blob_thickness=3, wf_type="linear", wf_style="dashed",
        wf_color="#ff9500", wiring_density=3, corner_radius=0,
        inner_style="thermal", bg_mode="original", glow_enabled=True,
    ),
}


def main():
    src = make_source()
    cv2.imwrite(os.path.join(OUT, "original.png"), src)
    print("original.png OK")
    for name, cfg in STYLES.items():
        out = process_image_frame(src.copy(), ProcessingConfig(**cfg))
        assert out is not None and isinstance(out, np.ndarray), name
        cv2.imwrite(os.path.join(OUT, f"{name}.png"), out)
        print(f"{name}.png OK  {out.shape}")


if __name__ == "__main__":
    main()
