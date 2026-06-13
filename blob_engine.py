import cv2
import numpy as np
import math
import os
import subprocess
from collections import deque
from dataclasses import dataclass, field
from types import SimpleNamespace
import frame_processor
import audio_processor
from signal_math import OneEuroFilter

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("Warning: ultralytics not installed. YOLO mode disabled.")

try:
    import mediapipe as mp
    _mp_vision = mp.tasks.vision
    _mp_BaseOptions = mp.tasks.BaseOptions
    MP_AVAILABLE = True
    _MP_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
except ImportError:
    MP_AVAILABLE = False
    print("Warning: mediapipe not installed. MediaPipe overlay disabled.")

# --- DATA STRUCTURES ---
@dataclass
class RenderBlob:
    """Normalized blob format used by the unified render pipeline."""
    tid: int
    cx: int
    cy: int
    x1: int
    y1: int
    x2: int
    y2: int
    r: int
    last_seen: int = 0
    fade_alpha: float = 1.0

    @staticmethod
    def from_preview(blob_dict):
        """Convert preview format {'id': tid, 'data': (cx,cy,x1,y1,x2,y2,r)}."""
        d = blob_dict['data']
        return RenderBlob(tid=blob_dict['id'], cx=d[0], cy=d[1],
                          x1=d[2], y1=d[3], x2=d[4], y2=d[5], r=d[6])

    @staticmethod
    def from_tracker(tid, tdata, frame_count=0, max_persist=30, persistence_fade=False):
        """Convert export tracker format (cx,cy,x1,y1,x2,y2,r,last_seen)."""
        cx, cy, x1, y1, x2, y2, r_val, last_seen = tdata
        fade = 1.0
        if persistence_fade and max_persist > 0:
            frames_since = frame_count - last_seen
            if frames_since > 0:
                fade = max(0.0, 1.0 - (frames_since / max_persist))
        return RenderBlob(tid=tid, cx=cx, cy=cy, x1=x1, y1=y1, x2=x2, y2=y2,
                          r=r_val, last_seen=last_seen, fade_alpha=fade)

    @property
    def data_tuple(self):
        """Return 7-tuple for draw functions."""
        return (self.cx, self.cy, self.x1, self.y1, self.x2, self.y2, self.r)


@dataclass
class RenderConfig:
    """Pre-converted rendering parameters. Colors already in BGR."""
    # Colors (BGR)
    blob_color: tuple = (255, 255, 255)
    wf_color: tuple = (255, 255, 255)
    text_color: tuple = (255, 255, 255)
    center_color: tuple = (0, 255, 255)
    text_outline_color: tuple = (0, 0, 0)

    # Shape
    blob_shape: str = "circular"
    blob_style: str = "solid"
    blob_thickness: int = 2
    corner_radius: int = 0
    blob_dot_gap: int = 10

    # Wireframe
    wf_type: str = "linear"
    wf_style: str = "solid"
    wf_thickness: int = 1
    wf_dot_gap: int = 20
    wiring_density: int = 5
    end_cap: str = "none"

    # Center
    show_center: bool = False
    center_shape: str = "circle"
    center_style: str = "filled"
    center_size_level: int = 1

    # Label
    label_type: str = "none"
    label_pos: str = "bottom"
    custom_text: str = "REC"
    font_weight: str = "regular"
    text_size: float = 0.6
    text_outline: bool = False

    # Scene
    inner_style: str = "normal"
    bg_mode: str = "original"
    opacity: float = 1.0

    # Glow
    glow_enabled: bool = False
    glow_intensity: float = 1.0
    glow_radius: int = 21

    # Trails
    trails_enabled: bool = False
    trail_length: int = 20
    trail_opacity: float = 0.6
    trail_style: str = "line"

    # Persistence
    persistence_fade: bool = False

    # Frame info
    frame_center: tuple = (640, 360)

    @staticmethod
    def from_pydantic(c):
        """Create from ProcessingConfig (preview/live)."""
        return RenderConfig(
            blob_color=hex_to_bgr(c.blob_color),
            wf_color=hex_to_bgr(c.wf_color),
            text_color=hex_to_bgr(c.text_color),
            center_color=hex_to_bgr(c.center_color),
            text_outline_color=hex_to_bgr(c.text_outline_color) if c.text_outline else (0, 0, 0),
            blob_shape=c.blob_shape, blob_style=c.blob_style,
            blob_thickness=c.blob_thickness, corner_radius=c.corner_radius,
            blob_dot_gap=c.blob_dot_gap,
            wf_type=c.wf_type, wf_style=c.wf_style, wf_thickness=c.wf_thickness,
            wf_dot_gap=c.wf_dot_gap, wiring_density=c.wiring_density, end_cap=c.end_cap,
            show_center=c.show_center, center_shape=c.center_shape,
            center_style=c.center_style, center_size_level=c.center_size_level,
            label_type=c.label_type, label_pos=c.label_pos,
            custom_text=c.custom_text, font_weight=c.font_weight,
            text_size=c.text_size, text_outline=c.text_outline,
            inner_style=c.inner_style, bg_mode=c.bg_mode, opacity=c.opacity,
            glow_enabled=c.glow_enabled, glow_intensity=c.glow_intensity,
            glow_radius=c.glow_radius,
            trails_enabled=c.trails_enabled, trail_length=c.trail_length,
            trail_opacity=c.trail_opacity, trail_style=c.trail_style,
            persistence_fade=c.persistence_fade,
        )

    @staticmethod
    def from_dict(config):
        """Create from dict (export)."""
        return RenderConfig(
            blob_color=hex_to_bgr(config['blob_color']),
            wf_color=hex_to_bgr(config.get('wf_color', '#FFFFFF')),
            text_color=hex_to_bgr(config['text_color']),
            center_color=hex_to_bgr(config.get('center_color', config['blob_color'])),
            text_outline_color=hex_to_bgr(config.get('text_outline_color', '#000000')) if config.get('text_outline') else (0, 0, 0),
            blob_shape=config['blob_shape'], blob_style=config['blob_style'],
            blob_thickness=config['blob_thickness'],
            corner_radius=config.get('corner_radius', 0),
            blob_dot_gap=config.get('blob_dot_gap', 10),
            wf_type=config['wf_type'], wf_style=config.get('wf_style', 'solid'),
            wf_thickness=config['wf_thickness'],
            wf_dot_gap=config.get('wf_dot_gap', 20),
            wiring_density=config.get('wiring_density', 99),
            end_cap=config.get('end_cap', 'none'),
            show_center=config['show_center'],
            center_shape=config.get('center_shape', 'circle'),
            center_style=config.get('center_style', 'filled'),
            center_size_level=config.get('center_size_level', 1),
            label_type=config['label_type'], label_pos=config.get('label_pos', 'bottom'),
            custom_text=config['custom_text'],
            font_weight=config.get('font_weight', 'regular'),
            text_size=config.get('text_size', 0.6),
            text_outline=config.get('text_outline', False),
            inner_style=config.get('inner_style', 'normal'),
            bg_mode=config.get('bg_mode', 'black'),
            opacity=config.get('opacity', 1.0),
            glow_enabled=config.get('glow_enabled', False),
            glow_intensity=config.get('glow_intensity', 1.0),
            glow_radius=config.get('glow_radius', 21),
            trails_enabled=config.get('trails_enabled', False),
            trail_length=config.get('trail_length', 20),
            trail_opacity=config.get('trail_opacity', 0.6),
            trail_style=config.get('trail_style', 'line'),
            persistence_fade=config.get('persistence_fade', False),
        )


class BufferPool:
    """Pre-allocated buffers for zero-allocation rendering."""
    def __init__(self, h, w):
        self.h = h
        self.w = w
        self.graphics_layer = np.zeros((h, w, 3), dtype=np.uint8)
        self.graphics_mask = np.zeros((h, w), dtype=np.uint8)
        self.mask_3c = np.zeros((h, w, 3), dtype=np.uint8)
        self._roi_cache = {}

    def reset(self):
        self.graphics_layer.fill(0)
        self.graphics_mask.fill(0)

    def fits(self, h, w):
        return self.h == h and self.w == w

    def get_mask_roi(self, roi_h, roi_w):
        key = (roi_h, roi_w)
        if key not in self._roi_cache:
            self._roi_cache[key] = np.zeros((roi_h, roi_w), dtype=np.uint8)
        else:
            self._roi_cache[key].fill(0)
        return self._roi_cache[key]


# --- HELPER COLORS ---
def hex_to_bgr(hex_code):
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))

# --- HELPER FFMPEG (AUDIO MERGE) ---
def merge_audio_with_ffmpeg(video_path, audio_path, offset, output_path):
    if not os.path.exists(audio_path):
        print("Audio file not found for merge.")
        return video_path

    temp_output = output_path.replace(".mp4", "_audio_muxed.mp4")
    
    cmd = [
        'ffmpeg', '-y',
        '-i', video_path,
        '-ss', str(max(0, offset)),
        '-i', audio_path,
        '-map', '0:v',
        '-map', '1:a',
        '-c:v', 'copy',
        '-c:a', 'aac',
        '-shortest',
        temp_output
    ]
    
    try:
        print(f"Running FFmpeg merge: {' '.join(cmd)}")
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        if os.path.exists(temp_output):
            os.remove(video_path) 
            os.rename(temp_output, video_path)
            return video_path
            
    except Exception as e:
        print(f"FFmpeg Merge Error: {e}")
        return video_path
    
    return video_path

# --- 1. FILTRI E STILI INTERNI ---
def apply_inner_style(img, style):
    if style == 'normal': return img
    if style == 'negative': return cv2.bitwise_not(img)
    if style == 'acid': return cv2.applyColorMap(img, cv2.COLORMAP_HSV)
    if style == 'red_only':
        b, g, r = cv2.split(img); z = np.zeros_like(b); return cv2.merge([z, z, r])
    if style == 'green_only':
        b, g, r = cv2.split(img); z = np.zeros_like(b); return cv2.merge([z, g, z])
    if style == 'blue_only':
        b, g, r = cv2.split(img); z = np.zeros_like(b); return cv2.merge([b, z, z])
    if style == 'bw':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    if style == 'ascii':
        h, w = img.shape[:2]
        scale = 0.21
        small_w, small_h = int(w * scale), int(h * scale)
        if small_w < 1 or small_h < 1: return img

        small = cv2.resize(img, (small_w, small_h), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        canvas = np.zeros_like(img)
        font = cv2.FONT_HERSHEY_SIMPLEX
        chars = " .'`^\",:;Il!i><~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"
        n_chars = len(chars) - 1

        step_x = w / small_w
        step_y = h / small_h

        for y in range(small_h):
            for x in range(small_w):
                val = gray[y, x]
                idx = int((val / 255) * n_chars)
                if idx > 0:
                    color = small[y, x]
                    c_bgr = (int(color[0]), int(color[1]), int(color[2]))
                    pos_x = int(x * step_x)
                    pos_y = int(y * step_y + step_y * 0.8)
                    cv2.putText(canvas, chars[idx], (pos_x, pos_y), font, 0.35, c_bgr, 1, cv2.LINE_AA)
        return canvas
    if style == 'posterize':
        levels = 6
        q = 256 // levels
        return (img // q * q + q // 2).astype(np.uint8)
    if style == 'edge':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    if style == 'thermal':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    if style == 'chromatic':
        out = np.zeros_like(img)
        h, w = img.shape[:2]
        # Red channel shifted +5px right
        out[:, :w-5, 2] = img[:, 5:, 2]
        # Green channel unchanged
        out[:, :, 1] = img[:, :, 1]
        # Blue channel shifted -5px left
        out[:, 5:, 0] = img[:, :w-5, 0]
        return out
    if style == 'scanlines':
        out = img.copy()
        out[::4, :] = (out[::4, :].astype(np.int16) * 0.4).clip(0, 255).astype(np.uint8)
        return out
    if style == 'halftone':
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        canvas = np.zeros_like(img)
        step = 6
        for yy in range(0, h, step):
            for xx in range(0, w, step):
                val = int(gray[yy, xx])
                radius = int((val / 255.0) * (step // 2))
                if radius > 0:
                    color = img[yy, xx]
                    cv2.circle(canvas, (xx + step // 2, yy + step // 2), radius,
                               (int(color[0]), int(color[1]), int(color[2])), -1)
        return canvas
    if style == 'pixelate':
        h, w = img.shape[:2]
        block = 8
        small = cv2.resize(img, (max(1, w // block), max(1, h // block)), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    if style == 'emboss':
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        emb = cv2.filter2D(gray, -1, kernel) + 128
        return cv2.cvtColor(np.clip(emb, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    if style == 'sketch':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        inv = cv2.bitwise_not(gray)
        blur = cv2.GaussianBlur(inv, (21, 21), 0)
        sketch_gray = cv2.divide(gray, cv2.bitwise_not(blur), scale=256)
        return cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    if style == 'vhs':
        h, w = img.shape[:2]
        out = img.copy()
        # Color bleed: shift red channel right
        out[:, 3:, 2] = img[:, :w-3, 2]
        # Scanline dimming every 3rd line
        out[::3, :] = (out[::3, :].astype(np.int16) * 0.6).clip(0, 255).astype(np.uint8)
        # Noise
        noise = np.random.randint(0, 25, (h, w), dtype=np.uint8)
        noise_bgr = cv2.cvtColor(noise, cv2.COLOR_GRAY2BGR)
        out = cv2.add(out, noise_bgr)
        return out
    if style == 'glitch':
        h, w = img.shape[:2]
        out = img.copy()
        n_slices = max(3, h // 60)
        rng = np.random.RandomState(42)
        for _ in range(n_slices):
            y0 = rng.randint(0, max(1, h - 20))
            sh = rng.randint(5, min(20, h - y0))
            dx = rng.randint(-30, 30)
            stripe = img[y0:y0+sh, :].copy()
            if dx > 0:
                out[y0:y0+sh, dx:] = stripe[:, :w-dx]
            elif dx < 0:
                out[y0:y0+sh, :w+dx] = stripe[:, -dx:]
        return out
    if style == 'infrared':
        b, g, r = cv2.split(img)
        # Swap: vegetation (green) becomes bright white/pink
        out_r = cv2.addWeighted(g, 0.8, r, 0.2, 0)
        out_g = cv2.addWeighted(r, 0.5, b, 0.5, 0)
        out_b = b
        return cv2.merge([out_b, out_g, out_r])
    return img

# --- 1b. GLOW / BLOOM ---
def apply_glow(graphics_layer, graphics_mask, glow_intensity=1.0, glow_radius=21):
    """Apply neon glow effect to the graphics layer using screen blend."""
    # Ensure radius is odd
    r = max(5, glow_radius)
    if r % 2 == 0: r += 1

    # Create glow from graphics layer
    glow_layer = cv2.GaussianBlur(graphics_layer, (r, r), 0)

    # Scale glow by intensity
    glow_layer = np.clip(glow_layer.astype(np.float32) * glow_intensity, 0, 255).astype(np.uint8)

    # Screen blend: result = a + b - (a*b)/255
    a = graphics_layer.astype(np.float32)
    b = glow_layer.astype(np.float32)
    result = np.clip(a + b - (a * b) / 255.0, 0, 255).astype(np.uint8)

    # Expand mask to include glow halo (smooth alpha for seamless blend)
    expanded_mask = cv2.GaussianBlur(graphics_mask, (r, r), 0)
    expanded_mask = np.maximum(expanded_mask, graphics_mask).astype(np.uint8)

    return result, expanded_mask

# --- 1c. MOTION TRAILS ---
def draw_trails(img, mask, trail_history, trail_style, trail_opacity, color, thickness):
    """Draw motion trails from trail_history onto img/mask."""
    for tid, positions in trail_history.items():
        n = len(positions)
        if n < 2: continue
        pts = list(positions)
        t_int = max(1, int(round(thickness)))

        if trail_style == "line":
            for i in range(1, n):
                alpha = (i / n) * trail_opacity
                c = tuple(int(v * alpha) for v in color)
                cv2.line(img, pts[i-1], pts[i], c, t_int, cv2.LINE_AA)
                if mask is not None:
                    cv2.line(mask, pts[i-1], pts[i], int(255 * alpha), t_int, cv2.LINE_AA)

        elif trail_style == "dots":
            for i in range(n):
                alpha = ((i + 1) / n) * trail_opacity
                c = tuple(int(v * alpha) for v in color)
                r = max(1, t_int)
                cv2.circle(img, pts[i], r, c, -1, cv2.LINE_AA)
                if mask is not None:
                    cv2.circle(mask, pts[i], r, int(255 * alpha), -1, cv2.LINE_AA)

        elif trail_style == "fade":
            for i in range(n):
                alpha = ((i + 1) / n) * trail_opacity
                c = tuple(int(v * alpha) for v in color)
                r = max(1, int(t_int * (1.0 + (n - i) * 0.3)))
                cv2.circle(img, pts[i], r, c, -1, cv2.LINE_AA)
                if mask is not None:
                    cv2.circle(mask, pts[i], r, int(255 * alpha), -1, cv2.LINE_AA)

# --- 2. HELPER GEOMETRICI ---
_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

def _hex_to_hsv(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    pixel = np.uint8([[[b, g, r]]])
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0][0]

def get_channel(frame, mode, config=None):
    # HSV modes
    if mode in ('hsv_hue', 'hsv_saturation', 'hsv_value'):
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        ch_idx = {'hsv_hue': 0, 'hsv_saturation': 1, 'hsv_value': 2}[mode]
        ch = hsv[:, :, ch_idx]
        if mode == 'hsv_hue':
            ch = (ch.astype(np.float32) * (255.0 / 180.0)).astype(np.uint8)
        return _clahe.apply(ch)

    # LAB modes
    if mode in ('lab_lightness', 'lab_a', 'lab_b'):
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        ch_idx = {'lab_lightness': 0, 'lab_a': 1, 'lab_b': 2}[mode]
        return _clahe.apply(lab[:, :, ch_idx])

    # Color target: track a specific hex color within tolerance
    if mode == 'color_target' and config is not None:
        target_hsv = _hex_to_hsv(config.color_target_hex)
        tol = max(1, min(90, config.color_target_tolerance))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h_target = int(target_hsv[0])
        # Handle hue wrap-around (red straddles 0/180)
        lower_h = h_target - tol
        upper_h = h_target + tol
        if lower_h < 0:
            mask1 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([upper_h, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([180 + lower_h, 40, 40]), np.array([180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        elif upper_h > 180:
            mask1 = cv2.inRange(hsv, np.array([lower_h, 40, 40]), np.array([180, 255, 255]))
            mask2 = cv2.inRange(hsv, np.array([0, 40, 40]), np.array([upper_h - 180, 255, 255]))
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, np.array([lower_h, 40, 40]), np.array([upper_h, 255, 255]))
        return mask

    # Single BGR channel
    if mode == 'red': return _clahe.apply(frame[:, :, 2])
    if mode == 'green': return _clahe.apply(frame[:, :, 1])
    if mode == 'blue': return _clahe.apply(frame[:, :, 0])

    # Average
    if mode == 'average':
        avg = np.mean(frame, axis=2).astype(np.uint8)
        return _clahe.apply(avg)

    # Default: luminance (grayscale)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return _clahe.apply(gray)

def get_intersection(cx1, cy1, r1, m1x, m1y, m2x, m2y, cx2, cy2, shape, w1, h1):
    dx, dy = cx2 - cx1, cy2 - cy1
    dist = math.hypot(dx, dy)
    if dist == 0: return (int(cx1), int(cy1))
    if shape == 'circular' and r1 > 0:
        nx, ny = dx / dist, dy / dist
        return (int(cx1 + nx * r1), int(cy1 + ny * r1))
    else:
        # Bounding-box intersection (also fallback when r1==0)
        if w1 <= 0 or h1 <= 0:
            return (int(cx1), int(cy1))
        if dx == 0: return (int(cx1), int(m1y if dy < 0 else m2y))
        if dy == 0: return (int(m1x if dx < 0 else m2x), int(cy1))
        scale = min((w1/2) / abs(dx), (h1/2) / abs(dy))
        return (int(cx1 + dx * scale), int(cy1 + dy * scale))

# --- 3. DISEGNO ---
def draw_arrow_head(img, mask, tip, tail, color, size=10, thickness=1):
    t_int = max(1, int(round(thickness)))
    dx, dy = tip[0] - tail[0], tip[1] - tail[1]
    angle = math.atan2(dy, dx)
    p1 = (int(tip[0] - size * math.cos(angle + math.pi/6)), int(tip[1] - size * math.sin(angle + math.pi/6)))
    p2 = (int(tip[0] - size * math.cos(angle - math.pi/6)), int(tip[1] - size * math.sin(angle - math.pi/6)))
    cv2.line(img, tip, p1, color, t_int, cv2.LINE_AA)
    cv2.line(img, tip, p2, color, t_int, cv2.LINE_AA)
    if mask is not None:
        cv2.line(mask, tip, p1, 255, t_int, cv2.LINE_AA)
        cv2.line(mask, tip, p2, 255, t_int, cv2.LINE_AA)

def draw_line_custom(img, mask, p1, p2, type, style, color, thickness, gap, end_cap):
    t_int = max(1, int(round(thickness)))
    cap_start, cap_end = p2, p1

    if type == 'curved':
        mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2 - int(math.hypot(p2[0]-p1[0], p2[1]-p1[1])*0.2))
        steps = 30
        if style in ('dotted', 'dashed'):
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1]) * 1.2
            steps = max(30, int(dist / max(5, gap)) * 2)

        pts = []
        for t in np.linspace(0, 1, steps):
            px = int((1-t)**2 * p1[0] + 2*(1-t)*t*mid[0] + t**2 * p2[0])
            py = int((1-t)**2 * p1[1] + 2*(1-t)*t*mid[1] + t**2 * p2[1])
            pts.append((px, py))

        if style == 'dashed':
            # Dashed: draw 3 segments, skip 1
            for i in range(len(pts)-1):
                if i % 4 == 3: continue
                cv2.line(img, pts[i], pts[i+1], color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.line(mask, pts[i], pts[i+1], 255, t_int, cv2.LINE_AA)
        else:
            for i in range(len(pts)-1):
                if style == 'dotted' and i % 2 == 0: continue
                cv2.line(img, pts[i], pts[i+1], color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.line(mask, pts[i], pts[i+1], 255, t_int, cv2.LINE_AA)
        cap_start, cap_end = (pts[1] if len(pts)>1 else p1), (pts[-2] if len(pts)>1 else p1)
    else:
        if style in ('dotted', 'dashed'):
            dist = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            if dist > 0:
                dx, dy = (p2[0]-p1[0])/dist, (p2[1]-p1[1])/dist
                if style == 'dashed':
                    dash_len = max(t_int*4, 10)
                    space_len = dash_len // 2
                else:
                    dash_len = max(t_int*2, 4)
                    space_len = max(t_int, gap)
                # Uniform redistribution
                unit = dash_len + space_len
                n_dashes = max(1, round(dist / unit))
                actual_unit = dist / n_dashes
                actual_dash = actual_unit * (dash_len / unit)
                for i in range(n_dashes):
                    start_d = i * actual_unit
                    end_d = start_d + actual_dash
                    s = (int(p1[0]+dx*start_d), int(p1[1]+dy*start_d))
                    e = (int(p1[0]+dx*end_d), int(p1[1]+dy*end_d))
                    cv2.line(img, s, e, color, t_int, cv2.LINE_AA)
                    if mask is not None: cv2.line(mask, s, e, 255, t_int, cv2.LINE_AA)
        else:
            cv2.line(img, p1, p2, color, t_int, cv2.LINE_AA)
            if mask is not None: cv2.line(mask, p1, p2, 255, t_int, cv2.LINE_AA)

    cs = 8 + t_int
    if end_cap in ['circle', 'both_circles']: 
        cv2.circle(img, p1, t_int+2, color, -1, cv2.LINE_AA)
        if mask is not None: cv2.circle(mask, p1, t_int+2, 255, -1, cv2.LINE_AA)
    if end_cap in ['arrow', 'both_arrows']: draw_arrow_head(img, mask, p1, cap_start, color, cs, t_int)
    if end_cap in ['circle', 'both_circles']: 
        cv2.circle(img, p2, t_int+2, color, -1, cv2.LINE_AA)
        if mask is not None: cv2.circle(mask, p2, t_int+2, 255, -1, cv2.LINE_AA)
    if end_cap in ['arrow', 'both_arrows']: draw_arrow_head(img, mask, p2, cap_end, color, cs, t_int)

def draw_rounded_rect(img, mask, rect, color, thickness, r, style, gap):
    x, y, w, h = rect
    r = min(r, w//2, h//2)
    t_int = max(1, int(round(thickness)))
    
    def dl(p1, p2):
        if style in ('dotted', 'dashed'):
            draw_line_custom(img, mask, p1, p2, 'linear', style, color, thickness, gap, 'none')
        else:
            cv2.line(img, p1, p2, color, t_int, cv2.LINE_AA)
            if mask is not None: cv2.line(mask, p1, p2, 255, t_int, cv2.LINE_AA)

    def da(c, a):
        cv2.ellipse(img, c, (r, r), a, 0, 90, color, t_int, cv2.LINE_AA)
        if mask is not None: cv2.ellipse(mask, c, (r, r), a, 0, 90, 255, t_int, cv2.LINE_AA)

    if r <= 0:
        if style in ('dotted', 'dashed'):
            pts = [(x,y), (x+w,y), (x+w,y+h), (x,y+h)]
            for i in range(4): draw_line_custom(img, mask, pts[i], pts[(i+1)%4], 'linear', style, color, thickness, gap, 'none')
        else:
            cv2.rectangle(img, (x, y), (x+w, y+h), color, t_int, cv2.LINE_AA)
            if mask is not None: cv2.rectangle(mask, (x, y), (x+w, y+h), 255, t_int, cv2.LINE_AA)
        return

    dl((x+r, y), (x+w-r, y))
    dl((x+w, y+r), (x+w, y+h-r))
    dl((x+w-r, y+h), (x+r, y+h))
    dl((x, y+h-r), (x, y+r))
    
    da((x+r, y+r), 180)
    da((x+w-r, y+r), 270)
    da((x+w-r, y+h-r), 0)
    da((x+r, y+h-r), 90)

def draw_center_custom(img, mask, cx, cy, color, shape, style, base_thickness, size_level):
    base_size = max(3, int(base_thickness + 2))
    size = base_size * size_level
    thick = -1 if style == 'filled' else max(1, int(round(base_thickness / 2)))
    if shape == 'square':
        cv2.rectangle(img, (cx-size, cy-size), (cx+size, cy+size), color, thick, cv2.LINE_AA)
        if mask is not None: cv2.rectangle(mask, (cx-size, cy-size), (cx+size, cy+size), 255, thick, cv2.LINE_AA)
    else:
        cv2.circle(img, (cx, cy), size, color, thick, cv2.LINE_AA)
        if mask is not None: cv2.circle(mask, (cx, cy), size, 255, thick, cv2.LINE_AA)

def draw_blob_shape(img, mask, b_data, shape, style, color, thickness, corner_radius, gap):
    cx, cy, x, y, x2, y2, r = b_data
    t_int = max(1, int(round(thickness)))
    w, h = x2-x, y2-y
    
    if style == 'none': return

    if shape == 'circular':
        if style == 'dotted':
            p = 2*math.pi*r
            n = max(4, int(p/max(5, gap)))
            # Round to nearest even number for uniform segments
            n = n + (n % 2)
            for i in range(0, n, 2):
                cv2.ellipse(img, (cx, cy), (r, r), 0, (i/n)*360, ((i+1)/n)*360, color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.ellipse(mask, (cx, cy), (r, r), 0, (i/n)*360, ((i+1)/n)*360, 255, t_int, cv2.LINE_AA)
        elif style == 'dashed':
            # Arcs of 30deg with 10deg gaps
            arc_deg = 30
            gap_deg = 10
            step = arc_deg + gap_deg
            angle = 0
            while angle < 360:
                end_angle = min(angle + arc_deg, 360)
                cv2.ellipse(img, (cx, cy), (r, r), 0, angle, end_angle, color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.ellipse(mask, (cx, cy), (r, r), 0, angle, end_angle, 255, t_int, cv2.LINE_AA)
                angle += step
        elif style == 'neon':
            # Draw solid shape
            cv2.circle(img, (cx, cy), r, color, t_int, cv2.LINE_AA)
            if mask is not None: cv2.circle(mask, (cx, cy), r, 255, t_int, cv2.LINE_AA)
            # Glow overlay: draw on temp layer, blur, blend
            h_img, w_img = img.shape[:2]
            glow_layer = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            cv2.circle(glow_layer, (cx, cy), r, color, t_int + 2, cv2.LINE_AA)
            glow_r = max(15, r // 3)
            if glow_r % 2 == 0: glow_r += 1
            glow_layer = cv2.GaussianBlur(glow_layer, (glow_r, glow_r), 0)
            cv2.add(img, glow_layer, img)
        elif style == 'segments_4':
            for sa in [45, 135, 225, 315]:
                cv2.ellipse(img, (cx, cy), (r, r), 0, sa-35, sa+35, color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.ellipse(mask, (cx, cy), (r, r), 0, sa-35, sa+35, 255, t_int, cv2.LINE_AA)
        elif style == 'segments_2':
            cv2.ellipse(img, (cx, cy), (r, r), 0, -50, 50, color, t_int, cv2.LINE_AA)
            cv2.ellipse(img, (cx, cy), (r, r), 0, 130, 230, color, t_int, cv2.LINE_AA)
            if mask is not None:
                cv2.ellipse(mask, (cx, cy), (r, r), 0, -50, 50, 255, t_int, cv2.LINE_AA)
                cv2.ellipse(mask, (cx, cy), (r, r), 0, 130, 230, 255, t_int, cv2.LINE_AA)
        else:
            cv2.circle(img, (cx, cy), r, color, t_int, cv2.LINE_AA)
            if mask is not None: cv2.circle(mask, (cx, cy), r, 255, t_int, cv2.LINE_AA)
    else:
        if style in ['corners', 'brackets']:
            cl = max(10, min(w, h) // 4)
            eff_r = min(corner_radius, cl)
            
            if eff_r <= 0:
                pts_h = {
                    'tl': ((x, y), (x+cl, y)), 'tr': ((x2, y), (x2-cl, y)),
                    'br': ((x2, y2), (x2-cl, y2)), 'bl': ((x, y2), (x+cl, y2))
                }
                pts_v = {
                    'tl': ((x, y), (x, y+cl)), 'tr': ((x2, y), (x2, y+cl)),
                    'br': ((x2, y2), (x2, y2-cl)), 'bl': ((x, y2), (x, y2-cl))
                }
                
                def draw_L(corner_key):
                    cv2.line(img, pts_h[corner_key][0], pts_h[corner_key][1], color, t_int, cv2.LINE_AA)
                    cv2.line(img, pts_v[corner_key][0], pts_v[corner_key][1], color, t_int, cv2.LINE_AA)
                    if mask is not None:
                        cv2.line(mask, pts_h[corner_key][0], pts_h[corner_key][1], 255, t_int, cv2.LINE_AA)
                        cv2.line(mask, pts_v[corner_key][0], pts_v[corner_key][1], 255, t_int, cv2.LINE_AA)

                if style == 'corners':
                    for k in ['tl', 'tr', 'br', 'bl']: draw_L(k)
                elif style == 'brackets':
                    draw_L('tl'); draw_L('bl')
                    draw_L('tr'); draw_L('br')
                    cv2.line(img, (x, y+cl), (x, y2-cl), color, t_int, cv2.LINE_AA)
                    cv2.line(img, (x2, y+cl), (x2, y2-cl), color, t_int, cv2.LINE_AA)
                    if mask is not None:
                        cv2.line(mask, (x, y+cl), (x, y2-cl), 255, t_int, cv2.LINE_AA)
                        cv2.line(mask, (x2, y+cl), (x2, y2-cl), 255, t_int, cv2.LINE_AA)
                return

            c_tl = (x + eff_r, y + eff_r)
            c_tr = (x2 - eff_r, y + eff_r)
            c_br = (x2 - eff_r, y2 - eff_r)
            c_bl = (x + eff_r, y2 - eff_r)

            def draw_corner_smooth(center, start_angle, end_angle, line_h_end, line_v_end):
                cv2.ellipse(img, center, (eff_r, eff_r), 0, start_angle, end_angle, color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.ellipse(mask, center, (eff_r, eff_r), 0, start_angle, end_angle, 255, t_int, cv2.LINE_AA)
                
                cx, cy = center
                p_north = (cx, cy - eff_r)
                p_south = (cx, cy + eff_r)
                p_west  = (cx - eff_r, cy)
                p_east  = (cx + eff_r, cy)
                
                start_h, start_v = None, None
                
                if start_angle == 180: start_h, start_v = p_north, p_west
                elif start_angle == 270: start_h, start_v = p_north, p_east
                elif start_angle == 0: start_h, start_v = p_south, p_east
                elif start_angle == 90: start_h, start_v = p_south, p_west
                
                if line_h_end:
                    cv2.line(img, start_h, line_h_end, color, t_int, cv2.LINE_AA)
                    if mask is not None: cv2.line(mask, start_h, line_h_end, 255, t_int, cv2.LINE_AA)
                
                if line_v_end:
                    cv2.line(img, start_v, line_v_end, color, t_int, cv2.LINE_AA)
                    if mask is not None: cv2.line(mask, start_v, line_v_end, 255, t_int, cv2.LINE_AA)

            if style == 'corners':
                draw_corner_smooth(c_tl, 180, 270, (x+cl, y), (x, y+cl))
                draw_corner_smooth(c_tr, 270, 360, (x2-cl, y), (x2, y+cl))
                draw_corner_smooth(c_br, 0, 90, (x2-cl, y2), (x2, y2-cl))
                draw_corner_smooth(c_bl, 90, 180, (x+cl, y2), (x, y2-cl))

            elif style == 'brackets':
                draw_corner_smooth(c_tl, 180, 270, (x+cl, y), None)
                draw_corner_smooth(c_bl, 90, 180, (x+cl, y2), None)
                cv2.line(img, (x, y+eff_r), (x, y2-eff_r), color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.line(mask, (x, y+eff_r), (x, y2-eff_r), 255, t_int, cv2.LINE_AA)

                draw_corner_smooth(c_tr, 270, 360, (x2-cl, y), None)
                draw_corner_smooth(c_br, 0, 90, (x2-cl, y2), None)
                cv2.line(img, (x2, y+eff_r), (x2, y2-eff_r), color, t_int, cv2.LINE_AA)
                if mask is not None: cv2.line(mask, (x2, y+eff_r), (x2, y2-eff_r), 255, t_int, cv2.LINE_AA)
        elif style == 'neon':
            # Solid rect + glow
            draw_rounded_rect(img, mask, (x, y, x2-x, y2-y), color, thickness, corner_radius, 'solid', gap)
            h_img, w_img = img.shape[:2]
            glow_layer = np.zeros((h_img, w_img, 3), dtype=np.uint8)
            draw_rounded_rect(glow_layer, None, (x, y, x2-x, y2-y), color, thickness + 2, corner_radius, 'solid', gap)
            glow_r = max(15, min(w, h) // 6)
            if glow_r % 2 == 0: glow_r += 1
            glow_layer = cv2.GaussianBlur(glow_layer, (glow_r, glow_r), 0)
            cv2.add(img, glow_layer, img)
        else:
            draw_rounded_rect(img, mask, (x, y, x2-x, y2-y), color, thickness, corner_radius, style, gap)

def draw_label(img, mask, b_data, label_type, custom_text, text_color, shape, font_weight, label_pos,
               text_size=0.6, text_outline=False, text_outline_color='#000000',
               tracker_id=None, blob_index=None, frame_center=None):
    if label_type == 'none': return
    cx, cy, x, y, x2, y2, r = b_data

    # Determine text based on label_type
    if label_type == 'coordinates':
        text = f"({cx},{cy})"
    elif label_type == 'id':
        text = f"#{tracker_id}" if tracker_id is not None else "#?"
    elif label_type == 'index':
        text = str((blob_index + 1) if blob_index is not None else "?")
    elif label_type == 'area':
        area = (x2 - x) * (y2 - y)
        text = str(area)
    elif label_type == 'distance':
        if frame_center is not None:
            dist = int(math.hypot(cx - frame_center[0], cy - frame_center[1]))
            text = f"{dist}px"
        else:
            text = "?px"
    else:
        text = custom_text

    if text:
        if font_weight == 'bold': font, thick = cv2.FONT_HERSHEY_TRIPLEX, 2
        elif font_weight == 'regular': font, thick = cv2.FONT_HERSHEY_DUPLEX, 1
        else: font, thick = cv2.FONT_HERSHEY_SIMPLEX, 1
        scale = text_size
        (tw, th_box), bl = cv2.getTextSize(text, font, scale, thick)
        if shape == 'circular': tx = cx - tw // 2
        else: tx = x + (x2 - x) // 2 - tw // 2

        if label_pos == 'top': ty = (cy - r - 15) if shape == 'circular' else (y - 15)
        elif label_pos == 'center': ty = cy + th_box // 2
        else: ty = (cy + r + 25) if shape == 'circular' else (y2 + 25)

        # Text outline (stroke behind text)
        if text_outline:
            outline_color = hex_to_bgr(text_outline_color) if isinstance(text_outline_color, str) else text_outline_color
            outline_thick = thick + 3
            cv2.putText(img, text, (tx, ty), font, scale, outline_color, outline_thick, cv2.LINE_AA)
            if mask is not None: cv2.putText(mask, text, (tx, ty), font, scale, 255, outline_thick, cv2.LINE_AA)

        cv2.putText(img, text, (tx, ty), font, scale, text_color, thick, cv2.LINE_AA)
        if mask is not None: cv2.putText(mask, text, (tx, ty), font, scale, 255, thick, cv2.LINE_AA)

# --- SILHOUETTE DETECTION ---
def detect_silhouette_blobs(frame_bgr, mp_image, segmenter, threshold, edge_low, edge_high, min_size, max_size, blob_shape):
    """Person silhouette → Canny edges → blob contours."""
    result = segmenter.segment(mp_image)
    mask = result.category_mask.numpy_view()  # uint8, 0=bg, 255=person

    _, binary_mask = cv2.threshold(mask, int(threshold * 255), 255, cv2.THRESH_BINARY)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.bitwise_and(gray, gray, mask=binary_mask)

    edges = cv2.Canny(masked_gray, edge_low, edge_high)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blobs = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_size <= area <= max_size:
            if blob_shape == 'circular':
                (cx, cy), radius = cv2.minEnclosingCircle(cnt)
                cx_i, cy_i, r_val = int(cx), int(cy), max(int(radius), 5)
                b_data = (cx_i, cy_i, cx_i - r_val, cy_i - r_val,
                          cx_i + r_val, cy_i + r_val, r_val)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                pad = 5
                b_data = (x + w // 2, y + h // 2, x - pad, y - pad,
                          x + w + pad, y + h + pad, 0)
            blobs.append({'id': 0, 'data': b_data})
    return blobs


# --- UNIFIED DETECTION ---
def detect_blobs(frame_ai, config, blob_shape, yolo_model=None, persist_tracking=False, segmenter=None, mp_image=None):
    """
    Unified blob detection for preview, export, and live.
    Returns list of {'id': tid, 'data': (cx,cy,x1,y1,x2,y2,r)}.
    `config` can be a pydantic model or SimpleNamespace with required fields.
    """
    blobs = []
    min_blob_size = config.min_blob_size if hasattr(config, 'min_blob_size') else getattr(config, 'min_blob_size', 100)
    max_blob_size = config.max_blob_size if hasattr(config, 'max_blob_size') else getattr(config, 'max_blob_size', 50000)
    detection_engine = config.detection_engine if hasattr(config, 'detection_engine') else getattr(config, 'detection_engine', 'color')
    use_high_res = config.use_high_res if hasattr(config, 'use_high_res') else getattr(config, 'use_high_res', False)
    track_mode = config.track_mode if hasattr(config, 'track_mode') else getattr(config, 'track_mode', 'luminance')
    threshold = config.threshold if hasattr(config, 'threshold') else getattr(config, 'threshold', 127)
    threshold_mode = config.threshold_mode if hasattr(config, 'threshold_mode') else getattr(config, 'threshold_mode', 'adaptive')
    morph_kernel_size = config.morph_kernel_size if hasattr(config, 'morph_kernel_size') else getattr(config, 'morph_kernel_size', 3)

    def is_valid_size(w, h):
        return min_blob_size <= (w * h) <= max_blob_size

    if detection_engine == 'yolo' and yolo_model is not None:
        inf_sz = 1280 if use_high_res else 640
        results = yolo_model.track(frame_ai, persist=persist_tracking,
                                   tracker="bytetrack.yaml", classes=[0],
                                   verbose=False, conf=0.15, iou=0.5, imgsz=inf_sz)
        for r in results:
            if r.boxes.id is not None:
                boxes = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy()
                for box, track_id in zip(boxes, ids):
                    x1, y1, x2, y2 = map(int, box)
                    tid = int(track_id)
                    w_b, h_b = x2 - x1, y2 - y1
                    if not is_valid_size(w_b, h_b):
                        continue
                    cx, cy = x1 + w_b // 2, y1 + h_b // 2
                    if blob_shape == 'circular':
                        r_blob = max(w_b, h_b) // 2 + 5
                        b_data = (cx, cy, cx - r_blob, cy - r_blob, cx + r_blob, cy + r_blob, r_blob)
                    else:
                        b_data = (cx, cy, x1 - 5, y1 - 5, x2 + 5, y2 + 5, 0)
                    blobs.append({'id': tid, 'data': b_data})
    elif detection_engine == 'edges':
        edge_low = getattr(config, 'edge_low', 50)
        edge_high = getattr(config, 'edge_high', 150)
        gray = cv2.cvtColor(frame_ai, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, edge_low, edge_high)
        # Aggressive morphology to close contours into solid object regions
        ks = max(5, min(15, morph_kernel_size * 2 + 1))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        edges = cv2.dilate(edges, kernel, iterations=3)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=4)
        edges = cv2.erode(edges, kernel, iterations=1)
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (min_blob_size <= area <= max_blob_size):
                continue
            if blob_shape == 'circular':
                (cx_c, cy_c), radius = cv2.minEnclosingCircle(cnt)
                cx_i, cy_i, r_val = int(cx_c), int(cy_c), int(radius) + 3
                b_data = (cx_i, cy_i, cx_i - r_val, cy_i - r_val,
                          cx_i + r_val, cy_i + r_val, r_val)
            else:
                x, y, w, h = cv2.boundingRect(cnt)
                pad = 5
                b_data = (x + w // 2, y + h // 2, x - pad, y - pad,
                          x + w + pad, y + h + pad, 0)
            blobs.append({'id': 0, 'data': b_data})
    elif detection_engine == 'silhouette' and segmenter is not None and mp_image is not None:
        edge_low = getattr(config, 'edge_low', 50)
        edge_high = getattr(config, 'edge_high', 150)
        sil_threshold = getattr(config, 'silhouette_threshold', 0.5)
        blobs = detect_silhouette_blobs(frame_ai, mp_image, segmenter, sil_threshold,
                                        edge_low, edge_high, min_blob_size, max_blob_size, blob_shape)
    else:
        gray = get_channel(frame_ai, track_mode, config=config)

        if track_mode == 'color_target':
            thresh_img = gray
        else:
            gray = cv2.GaussianBlur(gray, (5, 5), 0)
            if threshold_mode == 'otsu':
                _, thresh_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            elif threshold_mode == 'fixed':
                _, thresh_img = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            else:  # adaptive
                thresh_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY, 11, 2)

        ks = max(3, min(9, morph_kernel_size))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_OPEN, kernel, iterations=1)
        thresh_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel, iterations=1)

        cnts, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in cnts:
            area = cv2.contourArea(cnt)
            if not (min_blob_size <= area <= max_blob_size):
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            if blob_shape == 'circular':
                r_val = max(w, h) // 2 + 5
                b_data = (x + w // 2, y + h // 2, x + w // 2 - r_val, y + h // 2 - r_val,
                          x + w // 2 + r_val, y + h // 2 + r_val, r_val)
            else:
                b_data = (x + w // 2, y + h // 2, x - 5, y - 5, x + w + 5, y + h + 5, 0)
            blobs.append({'id': 0, 'data': b_data})

    # Sort by size (largest first)
    blobs.sort(key=lambda b: (b['data'][4] - b['data'][2]) * (b['data'][5] - b['data'][3]), reverse=True)
    return blobs


# --- UNIFIED WIREFRAME ---
def _draw_wireframes(blobs, rc, graphics_layer, graphics_mask):
    """Draw wireframe connections between blobs.
    blobs: list of RenderBlob. Uses vectorized distance if numpy available."""
    if rc.wf_type == 'none' or len(blobs) < 2:
        return
    n = len(blobs)
    # Build center arrays for vectorized distance
    centers = np.array([(b.cx, b.cy) for b in blobs], dtype=np.float64)
    # Pairwise squared distance matrix
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dist_sq = (diff ** 2).sum(axis=2)
    np.fill_diagonal(dist_sq, np.inf)

    density = min(rc.wiring_density, n - 1)
    drawn_pairs = set()

    for i in range(n):
        # Get top-K nearest using argpartition
        if density < n - 1:
            indices = np.argpartition(dist_sq[i], density)[:density]
        else:
            indices = np.arange(n)
            indices = indices[indices != i]

        for j in indices:
            pair = (min(i, j), max(i, j))
            if pair in drawn_pairs:
                continue
            drawn_pairs.add(pair)
            b1, b2 = blobs[i], blobs[j]
            t1, t2 = b1.data_tuple, b2.data_tuple
            if rc.blob_style == 'none':
                # Lines terminate at center point edge
                center_r = max(3, int(rc.blob_thickness + 2)) * rc.center_size_level
                p1 = get_intersection(t1[0], t1[1], center_r, t1[0]-center_r, t1[1]-center_r,
                                      t1[0]+center_r, t1[1]+center_r,
                                      t2[0], t2[1], 'circular', center_r*2, center_r*2)
                p2 = get_intersection(t2[0], t2[1], center_r, t2[0]-center_r, t2[1]-center_r,
                                      t2[0]+center_r, t2[1]+center_r,
                                      t1[0], t1[1], 'circular', center_r*2, center_r*2)
            else:
                p1 = get_intersection(t1[0], t1[1], t1[6], t1[2], t1[3], t1[4], t1[5],
                                      t2[0], t2[1], rc.blob_shape, t1[4] - t1[2], t1[5] - t1[3])
                p2 = get_intersection(t2[0], t2[1], t2[6], t2[2], t2[3], t2[4], t2[5],
                                      t1[0], t1[1], rc.blob_shape, t2[4] - t2[2], t2[5] - t2[3])
            draw_line_custom(graphics_layer, graphics_mask, p1, p2,
                             rc.wf_type, rc.wf_style, rc.wf_color, rc.wf_thickness,
                             rc.wf_dot_gap, rc.end_cap)


# --- UNIFIED RENDER PIPELINE ---
def render_frame(frame_clean, frame_ai, blobs_to_draw, rc, trail_history,
                 mod_energy=0.0, buffers=None):
    """
    Unified rendering pipeline for preview, export, and live.

    Args:
        frame_clean: original clean frame (numpy array)
        frame_ai: preprocessed frame for AI (can be same as frame_clean)
        blobs_to_draw: list of RenderBlob
        rc: RenderConfig with pre-converted colors
        trail_history: dict {tid: deque of (cx,cy)} — mutated in place
        mod_energy: audio modulation energy (0.0 if none)
        buffers: optional BufferPool for zero-allocation
    Returns:
        rendered frame (numpy array)
    """
    h_img, w_img = frame_clean.shape[:2]

    # 1. Base layer
    if rc.bg_mode == 'original':
        base_layer = frame_clean.copy()
    elif rc.bg_mode == 'green':
        base_layer = np.full_like(frame_clean, (0, 255, 0))
    elif rc.bg_mode == 'processed':
        base_layer = frame_ai.copy() if frame_ai is not None else frame_clean.copy()
    else:
        base_layer = np.zeros_like(frame_clean)

    # 2. Graphics layer + mask
    if buffers and buffers.fits(h_img, w_img):
        buffers.reset()
        graphics_layer = buffers.graphics_layer
        graphics_mask = buffers.graphics_mask
    else:
        graphics_layer = np.zeros_like(frame_clean)
        graphics_mask = np.zeros((h_img, w_img), dtype=np.uint8)

    # 3. Inner content
    inner_content = apply_inner_style(frame_clean, rc.inner_style)

    # 4. Apply inner content to blob regions
    for b in blobs_to_draw:
        x1c = max(0, b.x1)
        y1c = max(0, b.y1)
        x2c = min(w_img, b.x2)
        y2c = min(h_img, b.y2)
        roi_w, roi_h = x2c - x1c, y2c - y1c
        if roi_w <= 0 or roi_h <= 0:
            continue

        if buffers and buffers.fits(h_img, w_img):
            mask_roi = buffers.get_mask_roi(roi_h, roi_w)
        else:
            mask_roi = np.zeros((roi_h, roi_w), dtype=np.uint8)

        if rc.blob_shape == 'circular':
            cv2.circle(mask_roi, (b.cx - x1c, b.cy - y1c), b.r, 255, -1)
        else:
            if rc.corner_radius > 0:
                eff_r = min(rc.corner_radius, roi_w // 2, roi_h // 2)
                cv2.rectangle(mask_roi, (eff_r, 0), (roi_w - eff_r, roi_h), 255, -1)
                cv2.rectangle(mask_roi, (0, eff_r), (roi_w, roi_h - eff_r), 255, -1)
                cv2.ellipse(mask_roi, (eff_r, eff_r), (eff_r, eff_r), 180, 0, 90, 255, -1)
                cv2.ellipse(mask_roi, (roi_w - eff_r, eff_r), (eff_r, eff_r), 270, 0, 90, 255, -1)
                cv2.ellipse(mask_roi, (roi_w - eff_r, roi_h - eff_r), (eff_r, eff_r), 0, 0, 90, 255, -1)
                cv2.ellipse(mask_roi, (eff_r, roi_h - eff_r), (eff_r, eff_r), 90, 0, 90, 255, -1)
            else:
                mask_roi[:] = 255

        base_layer[y1c:y2c, x1c:x2c][mask_roi > 0] = inner_content[y1c:y2c, x1c:x2c][mask_roi > 0]

    # 5. Motion trails
    if rc.trails_enabled:
        active_ids = set()
        for b in blobs_to_draw:
            active_ids.add(b.tid)
            if b.tid not in trail_history:
                trail_history[b.tid] = deque(maxlen=rc.trail_length)
            trail_history[b.tid].append((b.cx, b.cy))
        dead = [k for k in trail_history if k not in active_ids]
        for k in dead:
            del trail_history[k]
        draw_trails(graphics_layer, graphics_mask, trail_history,
                    rc.trail_style, rc.trail_opacity, rc.blob_color, rc.blob_thickness)

    # 6. Wireframes
    _draw_wireframes(blobs_to_draw, rc, graphics_layer, graphics_mask)

    # 7. Audio modulation for thickness/glow
    eff_thickness = rc.blob_thickness
    eff_glow_intensity = rc.glow_intensity
    if mod_energy > 0:
        # Need audio_mod flags — check via config attrs that were preserved
        # We encode mod flags into mod_energy sign: positive means active
        eff_thickness = max(1, int(rc.blob_thickness * (1.0 + mod_energy)))
        eff_glow_intensity = rc.glow_intensity * (1.0 + mod_energy)

    # 8. Draw overlays (shape, center, label) with persistence fade
    for idx, b in enumerate(blobs_to_draw):
        draw_data = b.data_tuple
        # Audio size modulation
        if mod_energy > 0:
            scale = 1.0 + mod_energy
            new_r = int(b.r * scale)
            half_w = int((b.x2 - b.x1) / 2 * scale)
            half_h = int((b.y2 - b.y1) / 2 * scale)
            draw_data = (b.cx, b.cy, b.cx - half_w, b.cy - half_h,
                         b.cx + half_w, b.cy + half_h, new_r)

        # Apply fade_alpha to color
        draw_color = rc.blob_color
        if b.fade_alpha < 1.0:
            draw_color = tuple(int(v * b.fade_alpha) for v in rc.blob_color)

        draw_blob_shape(graphics_layer, graphics_mask, draw_data,
                        rc.blob_shape, rc.blob_style, draw_color,
                        eff_thickness, rc.corner_radius, rc.blob_dot_gap)
        if rc.show_center:
            draw_center_custom(graphics_layer, graphics_mask, b.cx, b.cy,
                               rc.center_color, rc.center_shape, rc.center_style,
                               eff_thickness, rc.center_size_level)
        draw_label(graphics_layer, graphics_mask, draw_data,
                   rc.label_type, rc.custom_text, rc.text_color,
                   rc.blob_shape, rc.font_weight, rc.label_pos,
                   text_size=rc.text_size, text_outline=rc.text_outline,
                   text_outline_color=rc.text_outline_color,
                   tracker_id=b.tid, blob_index=idx,
                   frame_center=rc.frame_center)

    # 9. Glow/bloom
    if rc.glow_enabled:
        graphics_layer, graphics_mask = apply_glow(graphics_layer, graphics_mask,
                                                   eff_glow_intensity, rc.glow_radius)

    # 10. Final compositing
    has_smooth_mask = rc.glow_enabled
    if has_smooth_mask:
        if buffers and buffers.fits(h_img, w_img):
            mask_3c = buffers.mask_3c
            mask_3c[:, :, 0] = graphics_mask
            mask_3c[:, :, 1] = graphics_mask
            mask_3c[:, :, 2] = graphics_mask
        else:
            mask_3c = cv2.merge([graphics_mask, graphics_mask, graphics_mask])
        alpha = mask_3c.astype(np.float32) / 255.0
        if rc.opacity < 0.99:
            alpha = alpha * rc.opacity
        final = (graphics_layer.astype(np.float32) * alpha +
                 base_layer.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    elif rc.opacity >= 0.99:
        # Fast path: boolean mask per-channel
        bool_mask = graphics_mask > 0
        final = base_layer.copy()
        for ch in range(3):
            final[:, :, ch][bool_mask] = graphics_layer[:, :, ch][bool_mask]
    else:
        if buffers and buffers.fits(h_img, w_img):
            mask_3c = buffers.mask_3c
            mask_3c[:, :, 0] = graphics_mask
            mask_3c[:, :, 1] = graphics_mask
            mask_3c[:, :, 2] = graphics_mask
        else:
            mask_3c = cv2.merge([graphics_mask, graphics_mask, graphics_mask])
        blended = cv2.addWeighted(base_layer, 1.0, graphics_layer, rc.opacity, 0)
        final = np.where(mask_3c > 0, blended, base_layer)

    return final


# --- MEDIAPIPE BLOB DETECTION ---
# Landmark priority orders
_HAND_LANDMARK_PRIORITY = [8, 12, 4, 16, 20, 0, 5, 9, 13, 17, 6, 7, 10, 11, 14, 15, 18, 19, 1, 2, 3]
_POSE_LANDMARK_PRIORITY = [0, 15, 16, 11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 7, 8, 19, 20, 31, 32,
                           1, 2, 3, 4, 5, 6, 9, 10, 17, 18, 21, 22, 29, 30]
_FACE_KEY_POINTS = [1, 152, 33, 263, 61, 291, 10, 46, 276, 0, 17, 133, 362, 234, 454]


def detect_hand_blobs_with_gesture(mp_image, mp_state, num_points, h, w, size_mul,
                                    gesture_enabled, min_scale, max_scale):
    """Detect hand landmarks as blobs + optional pinch gesture scale.

    Returns (blobs, gesture_scale) where gesture_scale is None if no gesture detected.
    """
    if 'hands' not in mp_state:
        return [], None

    result = mp_state['hands'].detect(mp_image)
    if not result.hand_landmarks:
        return [], None

    blobs = []
    gesture_scale = None
    num_points = max(1, min(num_points, 21))
    priority = _HAND_LANDMARK_PRIORITY[:num_points]

    for hand_lms in result.hand_landmarks:
        wrist = hand_lms[0]
        mid_tip = hand_lms[12]
        dist_px = math.hypot((mid_tip.x - wrist.x) * w, (mid_tip.y - wrist.y) * h)
        radius = max(8, int(dist_px / 8 * size_mul))

        for lm_idx in priority:
            if lm_idx >= len(hand_lms):
                continue
            lm = hand_lms[lm_idx]
            px, py = int(lm.x * w), int(lm.y * h)
            blobs.append({
                'id': 0,
                'data': (px, py, px - radius, py - radius, px + radius, py + radius, radius)
            })

        # Pinch gesture: thumb(4) to index(8) distance relative to hand size
        if gesture_enabled and dist_px > 0:
            thumb = hand_lms[4]
            index = hand_lms[8]
            pinch_dist = math.hypot((thumb.x - index.x) * w, (thumb.y - index.y) * h)
            ratio = pinch_dist / dist_px  # 0 = pinched, ~1 = open
            # Map ratio to scale range
            scale = min_scale + ratio * (max_scale - min_scale)
            scale = max(min_scale, min(max_scale, scale))
            gesture_scale = scale

    return blobs, gesture_scale


def detect_pose_blobs(mp_image, mp_state, num_points, confidence, h, w, size_mul):
    """Detect pose landmarks as blobs.

    Uses priority ordering and shoulder distance for radius.
    """
    if 'pose' not in mp_state:
        return []

    result = mp_state['pose'].detect(mp_image)
    if not result.pose_landmarks:
        return []

    blobs = []
    num_points = max(1, min(num_points, 33))
    priority = _POSE_LANDMARK_PRIORITY[:num_points]

    for pose_lms in result.pose_landmarks:
        # Radius from shoulder distance (11 <-> 12)
        sh_l = pose_lms[11]
        sh_r = pose_lms[12]
        sh_dist = math.hypot((sh_l.x - sh_r.x) * w, (sh_l.y - sh_r.y) * h)
        radius = max(8, int(sh_dist / 6 * size_mul))

        for lm_idx in priority:
            if lm_idx >= len(pose_lms):
                continue
            lm = pose_lms[lm_idx]
            vis = lm.visibility if hasattr(lm, 'visibility') and lm.visibility is not None else 1.0
            if vis < confidence:
                continue
            px, py = int(lm.x * w), int(lm.y * h)
            blobs.append({
                'id': 0,
                'data': (px, py, px - radius, py - radius, px + radius, py + radius, radius)
            })

    return blobs


def detect_face_blobs(mp_image, mp_state, num_points, h, w, size_mul):
    """Detect face key-point landmarks as blobs.

    Uses cheek distance (234 <-> 454) for radius.
    """
    if 'face_mesh' not in mp_state:
        return []

    result = mp_state['face_mesh'].detect(mp_image)
    if not result.face_landmarks:
        return []

    blobs = []
    num_points = max(1, min(num_points, 15))
    key_points = _FACE_KEY_POINTS[:num_points]

    for face_lms in result.face_landmarks:
        # Radius from cheek distance (234 <-> 454)
        cheek_l = face_lms[234]
        cheek_r = face_lms[454]
        cheek_dist = math.hypot((cheek_l.x - cheek_r.x) * w, (cheek_l.y - cheek_r.y) * h)
        radius = max(5, int(cheek_dist / 15 * size_mul))

        for lm_idx in key_points:
            if lm_idx >= len(face_lms):
                continue
            lm = face_lms[lm_idx]
            px, py = int(lm.x * w), int(lm.y * h)
            blobs.append({
                'id': 0,
                'data': (px, py, px - radius, py - radius, px + radius, py + radius, radius)
            })

    return blobs


def merge_nearby_blobs(blobs, merge_distance):
    """Merge blobs closer than merge_distance pixels.

    Center = midpoint, radius = sqrt(r1^2 + r2^2).
    """
    if merge_distance <= 0 or len(blobs) < 2:
        return blobs

    merged = list(blobs)
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(merged):
            j = i + 1
            while j < len(merged):
                b1 = merged[i]
                b2 = merged[j]
                d1, d2 = b1['data'], b2['data']
                dist = math.hypot(d1[0] - d2[0], d1[1] - d2[1])
                if dist < merge_distance:
                    # Merge: midpoint center, combined radius
                    cx = (d1[0] + d2[0]) // 2
                    cy = (d1[1] + d2[1]) // 2
                    r = int(math.sqrt(d1[6] ** 2 + d2[6] ** 2))
                    merged[i] = {
                        'id': 0,
                        'data': (cx, cy, cx - r, cy - r, cx + r, cy + r, r)
                    }
                    merged.pop(j)
                    changed = True
                else:
                    j += 1
            i += 1

    return merged


# --- CLASS ENGINE ---
class BlobEngine:
    def __init__(self):
        self.yolo_model = None
        self.current_model_name = None
        self.cap = None
        self.current_video_path = None
        
        self.cache = {
            'frame_index': -1,
            'detection_hash': None, 
            'clean_frame': None,
            'blobs': [],
        }
        self.audio_proc = audio_processor.AudioProcessor()
        
        # --- PREVIEW STATE FOR AUDIO QUEUE ---
        self.preview_queue = [] # List of indices (e.g., [0, 1, 2]) of blobs to show
        self.preview_cooldown = 0
        self.MAX_AUDIO_BLOBS = 5 # Used just for fetching raw audio data

        # --- MOTION TRAILS (Preview) ---
        self.trail_history = {}  # {tracker_id: deque of (cx,cy)}
        self._last_preview_frame = -1

        # --- AUDIO MODULATION (Preview) ---
        self.audio_features_cache = None
        self.audio_features_cache_key = None
        self.energy_filter = None

    def load_model(self, model_name):
        if not YOLO_AVAILABLE:
            print("Warning: YOLO not available")
            return None
        if self.yolo_model is None or self.current_model_name != model_name:
            print(f"Loading YOLO Model: {model_name}")
            self.yolo_model = YOLO(model_name)
            self.current_model_name = model_name
        return self.yolo_model
    
    def get_frame_from_video(self, video_path, frame_index):
        if self.cap is None or self.current_video_path != video_path:
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.current_video_path = video_path
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret or frame is None: raise ValueError("Frame read failed")
            return frame
        except Exception as e:
            print(f"[RECOVERY] Reset video cap: {e}")
            if self.cap: self.cap.release()
            self.cap = cv2.VideoCapture(video_path)
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = self.cap.read()
            if not ret: return np.zeros((1080, 1920, 3), dtype=np.uint8)
            return frame

    def process_single_frame(self, video_path, frame_index, config, is_preview=False):
        c = config

        # --- INITIALIZATION ---
        frame_ai = None
        blobs = []
        frame_clean = None

        det_hash = f"{c.detection_engine}_{c.yolo_model_file}_{c.threshold}_{c.min_blob_size}_{c.max_blob_size}_{c.max_blobs}_{c.track_mode}_{c.use_high_res}_{c.preprocess_enabled}_{c.preprocess_method}_{c.preprocess_strength}_{c.blob_shape}"

        # --- AUDIO QUEUE LOGIC (shared for cache hit and miss) ---
        def _update_audio_queue():
            if not (c.audio_enabled and c.audio_path):
                return
            try:
                fps = 30.0
                if self.cap:
                    cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if cap_fps and cap_fps > 0:
                        fps = cap_fps
                beat_frames = self.audio_proc.analyze_beats(
                    c.audio_path, fps,
                    band_focus=c.audio_band, sensitivity=c.audio_sensitivity
                )
                offset_frames = int(c.audio_offset * fps)
                target_frame = frame_index - offset_frames
                is_beat = target_frame in beat_frames
                if self.preview_cooldown > 0:
                    self.preview_cooldown -= 1
                if is_beat and self.preview_cooldown == 0:
                    next_idx = 0 if not self.preview_queue else self.preview_queue[-1] + 1
                    self.preview_queue.append(next_idx)
                    if len(self.preview_queue) > c.max_blobs:
                        self.preview_queue.pop(0)
                    self.preview_cooldown = 4
            except Exception as e:
                print(f"Audio error: {e}")

        # --- CHECK CACHE ---
        if (self.cache['frame_index'] == frame_index and
                self.cache['detection_hash'] == det_hash and
                self.cache['clean_frame'] is not None and is_preview):
            frame_clean = self.cache['clean_frame']
            blobs = self.cache['blobs']
            frame_ai = frame_clean
            _update_audio_queue()
        else:
            # --- FULL PROCESSING ---
            raw_frame = self.get_frame_from_video(video_path, frame_index)
            if raw_frame is None:
                return np.zeros((720, 1280, 3), dtype=np.uint8)

            _update_audio_queue()

            # Preprocess
            proc_config = {'preprocess_enabled': c.preprocess_enabled, 'preprocess_method': c.preprocess_method, 'preprocess_strength': c.preprocess_strength}
            processor = frame_processor.FrameProcessor(proc_config)
            frame_ai = processor.process(raw_frame)
            frame_clean = raw_frame.copy()

            # Detection (unified)
            yolo_model = self.load_model(c.yolo_model_file) if c.detection_engine == 'yolo' else None
            blobs = detect_blobs(frame_ai, c, c.blob_shape, yolo_model=yolo_model, persist_tracking=False)
            if not (c.audio_enabled and c.audio_path):
                blobs = blobs[:c.max_blobs]

            if is_preview:
                self.cache['frame_index'] = frame_index
                self.cache['detection_hash'] = det_hash
                self.cache['clean_frame'] = frame_clean
                self.cache['blobs'] = blobs

        # --- AUDIO FILTERING (QUEUE MAPPING) ---
        if c.audio_enabled and c.audio_path:
            filtered = [blobs[idx] for idx in self.preview_queue if idx < len(blobs)]
        else:
            filtered = blobs

        # --- AUDIO MODULATION ---
        preview_mod_energy = 0.0
        if c.audio_enabled and c.audio_path and (c.audio_modulate_size or c.audio_modulate_thickness or c.audio_modulate_glow):
            try:
                fps_prev = 30.0
                if self.cap:
                    cap_fps = self.cap.get(cv2.CAP_PROP_FPS)
                    if cap_fps and cap_fps > 0:
                        fps_prev = cap_fps
                total_f = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.cap else 1000
                cache_key = (c.audio_path, fps_prev, total_f)
                if self.audio_features_cache_key != cache_key:
                    self.audio_features_cache = self.audio_proc.analyze_full_features(c.audio_path, fps_prev, total_f)
                    self.audio_features_cache_key = cache_key
                    self.energy_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
                if self.audio_features_cache is not None:
                    fidx = min(frame_index, len(self.audio_features_cache['rms']) - 1)
                    raw_e = float(self.audio_features_cache['rms'][fidx])
                    if self.energy_filter is None:
                        self.energy_filter = OneEuroFilter(min_cutoff=1.0, beta=0.5)
                    preview_mod_energy = self.energy_filter.filter(frame_index / fps_prev, raw_e)
            except Exception as e:
                print(f"Preview audio mod error: {e}")

        # Compute effective mod_energy for render_frame
        mod_energy = 0.0
        if preview_mod_energy > 0:
            mod_energy = preview_mod_energy * c.audio_mod_intensity

        # --- TRAIL RESET on frame jump ---
        if c.trails_enabled and abs(frame_index - self._last_preview_frame) > 2:
            self.trail_history.clear()
        self._last_preview_frame = frame_index

        # --- CONVERT to RenderBlob ---
        render_blobs = [RenderBlob.from_preview(b) for b in filtered]

        # --- BUILD RenderConfig ---
        rc = RenderConfig.from_pydantic(c)
        rc.frame_center = (frame_clean.shape[1] // 2, frame_clean.shape[0] // 2)

        # Adjust for audio modulation flags
        if not (c.audio_modulate_size or c.audio_modulate_thickness or c.audio_modulate_glow):
            mod_energy = 0.0
        if c.audio_modulate_thickness and preview_mod_energy > 0:
            rc.blob_thickness = max(1, int(c.blob_thickness * (1.0 + preview_mod_energy * c.audio_mod_intensity)))
        if c.audio_modulate_glow and preview_mod_energy > 0:
            rc.glow_intensity = c.glow_intensity * (1.0 + preview_mod_energy * c.audio_mod_intensity)
        # Size modulation is handled inside render_frame via mod_energy
        size_mod_energy = preview_mod_energy * c.audio_mod_intensity if (c.audio_modulate_size and preview_mod_energy > 0) else 0.0

        # --- LAZY INIT BUFFER POOL ---
        h_img, w_img = frame_clean.shape[:2]
        if not hasattr(self, '_buffer_pool') or self._buffer_pool is None or not self._buffer_pool.fits(h_img, w_img):
            self._buffer_pool = BufferPool(h_img, w_img)

        return render_frame(frame_clean, frame_ai, render_blobs, rc, self.trail_history,
                            mod_energy=size_mod_energy, buffers=self._buffer_pool)

# --- 4. ENGINE PRINCIPALE PER EXPORT ---
def _track_blobs(blobs_raw, trackers, tracker_history, tracker_velocity, next_id,
                  frame_count, smoothing, match_radius, blob_shape):
    """
    Hungarian-like blob-to-tracker matching. Used by export and live.
    blobs_raw: list of 7-tuples (cx,cy,x1,y1,x2,y2,r) — from detect_blobs 'data' field.
    Returns updated (trackers, tracker_history, tracker_velocity, next_id).
    """
    updated_ids = set()
    for b in blobs_raw:
        b_area = (b[4] - b[2]) * (b[5] - b[3])
        match_id, min_cost = None, float('inf')
        for tid, tdata in trackers.items():
            if tid in updated_ids:
                continue
            pred_cx, pred_cy = tdata[0], tdata[1]
            if tid in tracker_velocity:
                vx, vy = tracker_velocity[tid]
                pred_cx += int(vx)
                pred_cy += int(vy)
            d = math.hypot(b[0] - pred_cx, b[1] - pred_cy)
            if d >= match_radius:
                continue
            t_area = max(1, (tdata[4] - tdata[2]) * (tdata[5] - tdata[3]))
            size_ratio = min(b_area, t_area) / max(b_area, t_area)
            cost = d * (2.0 - size_ratio)
            if cost < min_cost:
                min_cost = cost
                match_id = tid

        curr_id = match_id if match_id is not None else next_id
        if match_id is None:
            next_id += 1

        if curr_id in trackers:
            prev = trackers[curr_id]
            tracker_velocity[curr_id] = (b[0] - prev[0], b[1] - prev[1])
        else:
            tracker_velocity[curr_id] = (0, 0)

        final_pos = list(b) + [frame_count]
        if smoothing > 0:
            if curr_id not in tracker_history:
                tracker_history[curr_id] = deque(maxlen=smoothing)
            tracker_history[curr_id].append(b)
            hist = list(tracker_history[curr_id])
            avg = [int(sum(col) / len(col)) for col in zip(*hist)]
            final_pos = avg + [frame_count]
        trackers[curr_id] = tuple(final_pos)
        updated_ids.add(curr_id)

    return trackers, tracker_history, tracker_velocity, next_id


def run_processing(config, progress_callback=None):
    print("\n>>> ENGINE EXPORT AVVIATO.")
    input_path = config['input_path']
    output_folder = config['output_folder']

    max_blobs = config['max_blobs']
    detection_engine = config.get('detection_engine', 'color')
    yolo_model_file = config.get('yolo_model_file', 'yolov8n.pt')
    blob_shape = config['blob_shape']
    smoothing = config['smoothing']
    persistence = config['persistence']
    frame_skip = config['frame_skip']
    blob_thickness = config['blob_thickness']

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input: {input_path}")
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w_frame, h_frame = int(cap.get(3)), int(cap.get(4))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    processor = frame_processor.FrameProcessor(config)

    yolo_model = None
    if detection_engine == 'yolo' and YOLO_AVAILABLE:
        print(f">>> Caricamento Modello YOLO ({yolo_model_file})...")
        yolo_model = YOLO(yolo_model_file)

    filename = os.path.basename(input_path)
    out_name = f"EXP_V18_{detection_engine}_{filename}"
    out_path = os.path.join(output_folder, out_name)
    try:
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
    except Exception:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(out_path, fourcc, fps, (w_frame, h_frame))

    next_id = 0
    trackers = {}
    tracker_history = {} if smoothing > 0 else {}
    frame_count = 0
    trail_history_export = {}
    tracker_velocity = {}

    # Build RenderConfig once (pre-converted colors)
    rc = RenderConfig.from_dict(config)
    rc.frame_center = (w_frame // 2, h_frame // 2)

    # Detection config as SimpleNamespace for detect_blobs
    det_cfg = SimpleNamespace(
        detection_engine=detection_engine,
        min_blob_size=config['min_blob_size'],
        max_blob_size=config.get('max_blob_size', 50000),
        use_high_res=config.get('use_high_res', False),
        track_mode=config['track_mode'],
        threshold=config['threshold'],
        threshold_mode=config.get('threshold_mode', 'adaptive'),
        morph_kernel_size=config.get('morph_kernel_size', 3),
        color_target_hex=config.get('color_target_hex', '#FF0000'),
        color_target_tolerance=config.get('color_target_tolerance', 30),
    )
    match_radius = config.get('tracker_match_radius', 150)

    # Buffer pool for zero-allocation rendering
    buffer_pool = BufferPool(h_frame, w_frame)

    # AUDIO SETUP
    audio_proc_export = audio_processor.AudioProcessor()
    beat_map = set()
    audio_mod_size = config.get('audio_modulate_size', False)
    audio_mod_thickness = config.get('audio_modulate_thickness', False)
    audio_mod_glow = config.get('audio_modulate_glow', False)
    audio_mod_intensity = config.get('audio_mod_intensity', 1.0)

    if config.get('audio_enabled') and config.get('audio_path'):
        try:
            print(">>> Avvio Analisi Audio Precisa...")
            beat_map = audio_proc_export.analyze_beats(
                config['audio_path'], fps,
                band_focus=config.get('audio_band', 'bass'),
                sensitivity=config.get('audio_sensitivity', 1.0)
            )
        except Exception as e:
            print(f"Audio Analysis Error: {e}")

    audio_features = None
    energy_filter_export = OneEuroFilter(min_cutoff=1.0, beta=0.5)
    if config.get('audio_enabled') and config.get('audio_path') and (audio_mod_size or audio_mod_thickness or audio_mod_glow):
        try:
            audio_features = audio_proc_export.analyze_full_features(config['audio_path'], fps, total_frames)
            print(">>> Audio features estratte per modulazione continua.")
        except Exception as e:
            print(f"Audio features error: {e}")

    export_queue = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Audio trigger logic
        is_beat = False
        if config.get('audio_enabled') and config.get('audio_path'):
            offset_frames = int(config.get('audio_offset', 0.0) * fps)
            check_frame = frame_count - offset_frames
            is_beat = check_frame in beat_map

        if is_beat:
            next_idx = 0 if not export_queue else export_queue[-1] + 1
            export_queue.append(next_idx)
            if len(export_queue) > max_blobs:
                export_queue.pop(0)

        frame_clean = frame.copy()
        frame_ai = processor.process(frame)

        frame_count += 1
        if progress_callback and frame_count % 5 == 0:
            progress_callback(frame_count / total_frames)
        if (frame_skip > 1) and (frame_count % frame_skip != 1):
            continue

        # --- DETECTION (unified) ---
        if detection_engine == 'yolo' and yolo_model is not None:
            # YOLO with persist=True for export tracker continuity
            blobs_raw = detect_blobs(frame_ai, det_cfg, blob_shape,
                                     yolo_model=yolo_model, persist_tracking=True)
            # YOLO provides its own IDs — update trackers directly
            for b in blobs_raw:
                tid = b['id']
                b_data = b['data']
                final_pos = list(b_data) + [frame_count]
                if smoothing > 0:
                    if tid not in tracker_history:
                        tracker_history[tid] = deque(maxlen=smoothing)
                    tracker_history[tid].append(b_data)
                    hist = list(tracker_history[tid])
                    avg = [int(sum(col) / len(col)) for col in zip(*hist)]
                    final_pos = avg + [frame_count]
                trackers[tid] = tuple(final_pos)
        else:
            # Color detection — needs manual tracking
            blobs_raw = detect_blobs(frame_ai, det_cfg, blob_shape)
            if not (config.get('audio_enabled') and config.get('audio_path')):
                blobs_raw = blobs_raw[:max_blobs]
            blobs_tuples = [b['data'] for b in blobs_raw]
            trackers, tracker_history, tracker_velocity, next_id = _track_blobs(
                blobs_tuples, trackers, tracker_history, tracker_velocity,
                next_id, frame_count, smoothing, match_radius, blob_shape
            )

        # Persistence cleanup
        max_persist = frame_skip if persistence == 0 else persistence
        trackers = {tid: v for tid, v in trackers.items() if frame_count - v[7] <= max_persist}
        dead_tids = [tid for tid in tracker_velocity if tid not in trackers]
        for tid in dead_tids:
            tracker_velocity.pop(tid, None)
            trail_history_export.pop(tid, None)
            if tracker_history and tid in tracker_history:
                del tracker_history[tid]

        # Sort active trackers by size
        active_list_pairs = sorted(trackers.items(),
                                   key=lambda item: (item[1][4] - item[1][2]) * (item[1][5] - item[1][3]),
                                   reverse=True)

        # Audio queue filtering
        if config.get('audio_enabled'):
            active_to_draw = [active_list_pairs[idx] for idx in export_queue if idx < len(active_list_pairs)]
        else:
            active_to_draw = active_list_pairs[:max_blobs]

        # Audio modulation
        mod_energy_raw = 0.0
        if audio_features is not None and frame_count > 0:
            fidx = min(frame_count - 1, len(audio_features['rms']) - 1)
            raw_energy = float(audio_features['rms'][fidx])
            mod_energy_raw = energy_filter_export.filter(frame_count / fps, raw_energy)

        # Apply modulation to rc for this frame
        rc.blob_thickness = blob_thickness
        rc.glow_intensity = config.get('glow_intensity', 1.0)
        if mod_energy_raw > 0:
            if audio_mod_thickness:
                rc.blob_thickness = max(1, int(blob_thickness * (1.0 + mod_energy_raw * audio_mod_intensity)))
            if audio_mod_glow:
                rc.glow_intensity = rc.glow_intensity * (1.0 + mod_energy_raw * audio_mod_intensity)

        size_mod_energy = mod_energy_raw * audio_mod_intensity if (audio_mod_size and mod_energy_raw > 0) else 0.0

        # Convert to RenderBlob
        render_blobs = [
            RenderBlob.from_tracker(tid, tdata, frame_count, max_persist, rc.persistence_fade)
            for tid, tdata in active_to_draw
        ]

        # --- RENDER (unified) ---
        out_frame = render_frame(frame_clean, frame_ai, render_blobs, rc, trail_history_export,
                                 mod_energy=size_mod_energy, buffers=buffer_pool)
        out_video.write(out_frame)

    cap.release()
    out_video.release()

    if config.get('audio_enabled') and config.get('audio_path'):
        final_video_path = merge_audio_with_ffmpeg(
            out_path, config['audio_path'],
            config.get('audio_offset', 0.0), out_path
        )
        return final_video_path

    return out_path


# --- 5. LIVE ENGINE ---
class LiveEngine:
    """Real-time blob tracking with persistent tracker state.

    Combines detection, tracking, and unified rendering for live camera feeds.
    """

    def __init__(self):
        self.yolo_model = None
        self.current_model_name = None

        # Tracker state (same as export)
        self.trackers = {}
        self.tracker_history = {}
        self.tracker_velocity = {}
        self.next_id = 0

        # Trail state
        self.trail_history = {}

        # Buffer pool (lazy init)
        self._buffer_pool = None

        # Cached render config
        self._render_config = None
        self._frame_count = 0

        # MediaPipe state
        self._mp_state = {}
        self._mp_confidence = None
        self._mp_num_poses = None
        self._mp_num_faces = None

        # Gesture state
        self._gesture_scale = 1.0
        self._gesture_alpha = 0.3

    def _ensure_mp_solutions(self, config):
        """Lazy init/teardown MediaPipe Task objects based on config flags.

        Recreates landmarkers when confidence or num settings change.
        """
        if not MP_AVAILABLE:
            return
        c = config
        conf = c.mp_confidence
        num_poses = c.mp_num_poses
        num_faces = c.mp_num_faces

        # Detect parameter changes that require recreation
        conf_changed = conf != self._mp_confidence
        poses_changed = num_poses != self._mp_num_poses
        faces_changed = num_faces != self._mp_num_faces

        if conf_changed or poses_changed:
            if 'pose' in self._mp_state:
                self._mp_state['pose'].close()
                del self._mp_state['pose']
            if 'hands' in self._mp_state:
                self._mp_state['hands'].close()
                del self._mp_state['hands']
        if conf_changed or faces_changed:
            if 'face_mesh' in self._mp_state:
                self._mp_state['face_mesh'].close()
                del self._mp_state['face_mesh']

        self._mp_confidence = conf
        self._mp_num_poses = num_poses
        self._mp_num_faces = num_faces

        # Pose
        if c.mp_pose_enabled:
            if 'pose' not in self._mp_state:
                opts = _mp_vision.PoseLandmarkerOptions(
                    base_options=_mp_BaseOptions(
                        model_asset_path=os.path.join(_MP_MODELS_DIR, "pose_landmarker_lite.task")),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    num_poses=num_poses,
                    min_pose_detection_confidence=conf,
                    min_tracking_confidence=conf,
                )
                self._mp_state['pose'] = _mp_vision.PoseLandmarker.create_from_options(opts)
        else:
            if 'pose' in self._mp_state:
                self._mp_state['pose'].close()
                del self._mp_state['pose']

        # Hands
        if c.mp_hands_enabled:
            if 'hands' not in self._mp_state:
                opts = _mp_vision.HandLandmarkerOptions(
                    base_options=_mp_BaseOptions(
                        model_asset_path=os.path.join(_MP_MODELS_DIR, "hand_landmarker.task")),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    num_hands=4,
                    min_hand_detection_confidence=conf,
                    min_tracking_confidence=conf,
                )
                self._mp_state['hands'] = _mp_vision.HandLandmarker.create_from_options(opts)
        else:
            if 'hands' in self._mp_state:
                self._mp_state['hands'].close()
                del self._mp_state['hands']

        # Face Mesh
        if c.mp_face_enabled:
            if 'face_mesh' not in self._mp_state:
                opts = _mp_vision.FaceLandmarkerOptions(
                    base_options=_mp_BaseOptions(
                        model_asset_path=os.path.join(_MP_MODELS_DIR, "face_landmarker.task")),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    num_faces=num_faces,
                    min_face_detection_confidence=conf,
                    min_tracking_confidence=conf,
                )
                self._mp_state['face_mesh'] = _mp_vision.FaceLandmarker.create_from_options(opts)
        else:
            if 'face_mesh' in self._mp_state:
                self._mp_state['face_mesh'].close()
                del self._mp_state['face_mesh']

        # Silhouette Segmenter
        if c.detection_engine == 'silhouette':
            if 'segmenter' not in self._mp_state:
                model_path = os.path.join(_MP_MODELS_DIR, 'selfie_segmenter.tflite')
                options = _mp_vision.ImageSegmenterOptions(
                    base_options=_mp_BaseOptions(model_asset_path=model_path),
                    running_mode=_mp_vision.RunningMode.IMAGE,
                    output_category_mask=True
                )
                self._mp_state['segmenter'] = _mp_vision.ImageSegmenter.create_from_options(options)
        else:
            if 'segmenter' in self._mp_state:
                self._mp_state['segmenter'].close()
                del self._mp_state['segmenter']

    def load_model(self, model_name):
        if not YOLO_AVAILABLE:
            return None
        if self.yolo_model is None or self.current_model_name != model_name:
            print(f"[LiveEngine] Loading YOLO: {model_name}")
            self.yolo_model = YOLO(model_name)
            self.current_model_name = model_name
        return self.yolo_model

    def reset(self):
        """Clear all tracker state."""
        self.trackers.clear()
        self.tracker_history.clear()
        self.tracker_velocity.clear()
        self.trail_history.clear()
        self.next_id = 0
        self._frame_count = 0
        # Close MediaPipe solutions
        for sol in self._mp_state.values():
            sol.close()
        self._mp_state.clear()

    def process_frame(self, frame, config):
        """
        Full pipeline: flip -> detect -> MP blobs -> track -> merge -> render.

        Args:
            frame: BGR numpy array from camera
            config: ProcessingConfig (pydantic model)
        Returns:
            rendered frame (numpy array)
        """
        c = config
        self._frame_count += 1
        frame_count = self._frame_count

        # 1. Camera flip
        if c.camera_flip:
            frame = cv2.flip(frame, 1)

        # Preprocess
        proc_config = {
            'preprocess_enabled': c.preprocess_enabled,
            'preprocess_method': c.preprocess_method,
            'preprocess_strength': c.preprocess_strength,
        }
        processor = frame_processor.FrameProcessor(proc_config)
        frame_ai = processor.process(frame)
        frame_clean = frame.copy()

        # 2. Detection (color/YOLO/silhouette)
        yolo_model = self.load_model(c.yolo_model_file) if c.detection_engine == 'yolo' else None

        # Shared RGB + mp_image (needed for silhouette and MP blobs)
        h, w = frame_clean.shape[:2]
        any_mp = MP_AVAILABLE and (c.mp_hands_enabled or c.mp_pose_enabled or c.mp_face_enabled)
        need_mp_image = any_mp or (c.detection_engine == 'silhouette' and MP_AVAILABLE)

        mp_image = None
        if need_mp_image:
            self._ensure_mp_solutions(c)
            rgb = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        segmenter = self._mp_state.get('segmenter') if c.detection_engine == 'silhouette' else None
        detection_blobs = detect_blobs(frame_ai, c, c.blob_shape,
                                       yolo_model=yolo_model, persist_tracking=True,
                                       segmenter=segmenter, mp_image=mp_image)

        # 3. MediaPipe blobs
        mp_blobs = []

        if any_mp:
            try:
                if mp_image is None:
                    self._ensure_mp_solutions(c)
                    rgb = cv2.cvtColor(frame_clean, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # 4. Hands + gesture
                if c.mp_hands_enabled:
                    hand_blobs, gesture_scale = detect_hand_blobs_with_gesture(
                        mp_image, self._mp_state, c.mp_hands_num_points, h, w,
                        c.mp_blob_size, c.mp_gesture_size, c.mp_gesture_min, c.mp_gesture_max
                    )
                    mp_blobs.extend(hand_blobs)

                    # 5. Gesture EMA smoothing
                    if gesture_scale is not None:
                        self._gesture_scale = (self._gesture_alpha * gesture_scale +
                                               (1 - self._gesture_alpha) * self._gesture_scale)

                size_mul = c.mp_blob_size
                if c.mp_gesture_size and c.mp_hands_enabled:
                    size_mul = c.mp_blob_size * self._gesture_scale

                # 6. Pose
                if c.mp_pose_enabled:
                    pose_blobs = detect_pose_blobs(
                        mp_image, self._mp_state, c.mp_pose_num_points,
                        c.mp_confidence, h, w, size_mul
                    )
                    mp_blobs.extend(pose_blobs)

                # 7. Face
                if c.mp_face_enabled:
                    face_blobs = detect_face_blobs(
                        mp_image, self._mp_state, c.mp_face_num_points, h, w, size_mul
                    )
                    mp_blobs.extend(face_blobs)
            except Exception as e:
                print(f"[MediaPipe] blob detection error: {e}")

        # 8. Combine: detection_blobs[:max_blobs] + all mp_blobs
        max_blobs = c.max_blobs
        blobs_raw = detection_blobs[:max_blobs] + mp_blobs

        # 9. Merge nearby blobs
        if c.mp_merge_distance > 0:
            blobs_raw = merge_nearby_blobs(blobs_raw, c.mp_merge_distance)

        # Tracking
        smoothing = c.smoothing
        match_radius = c.tracker_match_radius
        persistence = c.persistence

        if c.detection_engine == 'yolo' and yolo_model is not None:
            for b in blobs_raw:
                tid = b['id']
                b_data = b['data']
                final_pos = list(b_data) + [frame_count]
                if smoothing > 0:
                    if tid not in self.tracker_history:
                        self.tracker_history[tid] = deque(maxlen=smoothing)
                    self.tracker_history[tid].append(b_data)
                    hist = list(self.tracker_history[tid])
                    avg = [int(sum(col) / len(col)) for col in zip(*hist)]
                    final_pos = avg + [frame_count]
                self.trackers[tid] = tuple(final_pos)
        else:
            blobs_tuples = [b['data'] for b in blobs_raw]
            self.trackers, self.tracker_history, self.tracker_velocity, self.next_id = _track_blobs(
                blobs_tuples, self.trackers, self.tracker_history,
                self.tracker_velocity, self.next_id, frame_count,
                smoothing, match_radius, c.blob_shape
            )

        # Persistence cleanup
        frame_skip = c.frame_skip
        max_persist = frame_skip if persistence == 0 else persistence
        self.trackers = {tid: v for tid, v in self.trackers.items()
                         if frame_count - v[7] <= max_persist}
        dead_tids = [tid for tid in self.tracker_velocity if tid not in self.trackers]
        for tid in dead_tids:
            self.tracker_velocity.pop(tid, None)
            self.trail_history.pop(tid, None)
            if tid in self.tracker_history:
                del self.tracker_history[tid]

        # Sort by size — no max_blobs cap (MP blobs bypass it)
        active_list_pairs = sorted(
            self.trackers.items(),
            key=lambda item: (item[1][4] - item[1][2]) * (item[1][5] - item[1][3]),
            reverse=True
        )

        # Convert to RenderBlob
        render_blobs = [
            RenderBlob.from_tracker(tid, tdata, frame_count, max_persist, c.persistence_fade)
            for tid, tdata in active_list_pairs
        ]

        # Build RenderConfig
        rc = RenderConfig.from_pydantic(c)
        h_img, w_img = frame_clean.shape[:2]
        rc.frame_center = (w_img // 2, h_img // 2)

        # Buffer pool
        if self._buffer_pool is None or not self._buffer_pool.fits(h_img, w_img):
            self._buffer_pool = BufferPool(h_img, w_img)

        return render_frame(frame_clean, frame_ai, render_blobs, rc,
                            self.trail_history, mod_energy=0.0,
                            buffers=self._buffer_pool)