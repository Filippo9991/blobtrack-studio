"""blob_engine_core — primitive di blob detection/styling (solo OpenCV).

Funzioni estratte dal motore originale di BlobTrack (legacy/backend/blob_engine.py),
ripulite da ogni dipendenza pesante (YOLO/ultralytics/torch, MediaPipe, ffmpeg) così
da girare leggere nel free tier di Render (512 MB). Lavorano tutte su singole immagini
numpy in formato BGR di OpenCV.
"""
import math

import cv2
import numpy as np

# CLAHE: equalizzazione adattiva del contrasto, usata da get_channel()

_clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))



def hex_to_bgr(hex_code):
    h = hex_code.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (4, 2, 0))


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
