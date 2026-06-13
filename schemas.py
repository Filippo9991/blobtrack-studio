from pydantic import BaseModel
from typing import Optional, Tuple

class ProcessingConfig(BaseModel):
    # --- Detection ---
    input_path: Optional[str] = ""
    output_folder: Optional[str] = ""
    detection_engine: str = "color"
    yolo_model_file: str = "yolov8m.pt"
    use_high_res: bool = False
    track_mode: str = "luminance"  # luminance, red, green, blue, average, hsv_hue, hsv_saturation, hsv_value, lab_lightness, lab_a, lab_b, color_target
    threshold: int = 127
    threshold_mode: str = "adaptive"  # adaptive, fixed, otsu
    color_target_hex: str = "#FF0000"  # target color for color_target mode
    color_target_tolerance: int = 30   # tolerance in HSV hue (0-90)
    morph_kernel_size: int = 3         # morphology kernel size (3, 5, 7)
    edge_low: int = 50                   # Canny low threshold (10-200)
    edge_high: int = 150                  # Canny high threshold (50-300)
    min_blob_size: int = 100
    max_blob_size: int = 50000
    max_blobs: int = 20

    # --- Pre-processing ---
    preprocess_enabled: bool = False
    preprocess_method: str = "CrowdBoost"
    preprocess_strength: float = 1.0

    # --- Styling ---
    blob_shape: str = "circular"
    blob_color: str = "#FFFFFF"
    blob_thickness: int = 2
    blob_style: str = "solid"
    corner_radius: int = 0
    blob_dot_gap: int = 10
    
    # --- Wireframe ---
    wf_type: str = "linear"
    wf_color: str = "#FFFFFF"
    wf_thickness: int = 1
    wf_style: str = "solid"
    wf_dot_gap: int = 20
    wiring_density: int = 5
    end_cap: str = "none"

    # --- Center Point ---
    show_center: bool = False
    center_color: str = "#FFFF00"
    center_shape: str = "circle"
    center_style: str = "filled"
    center_size_level: int = 1

    # --- Text / Labels ---
    label_type: str = "none"
    text_color: str = "#FFFFFF"
    custom_text: str = "REC"
    font_weight: str = "regular"
    label_pos: str = "bottom"

    # --- Export & Scene ---
    inner_style: str = "normal"
    bg_mode: str = "original"
    opacity: float = 1.0
    smoothing: int = 5
    persistence: int = 30
    frame_skip: int = 1

    # --- Audio ---
    audio_enabled: bool = False
    audio_path: Optional[str] = None
    audio_band: str = "bass"    # 'bass' (0-250Hz), 'low_mids' (250-2k), 'high_mids' (2-6k), 'highs' (6-22k), 'full' (0-22k)
    audio_sensitivity: float = 1.0
    audio_offset: float = 0.0
    audio_sync_video: bool = True

    # --- Glow / Bloom ---
    glow_enabled: bool = False
    glow_intensity: float = 1.0        # 0.0-2.0
    glow_radius: int = 21              # odd int 5-51

    # --- Audio Modulation (Continua) ---
    audio_modulate_size: bool = False
    audio_modulate_thickness: bool = False
    audio_modulate_glow: bool = False
    audio_mod_intensity: float = 1.0    # 0.0-2.0

    # --- Motion Trails ---
    trails_enabled: bool = False
    trail_length: int = 20             # 5-60 frames
    trail_opacity: float = 0.6         # 0.1-1.0
    trail_style: str = "line"          # "line", "dots", "fade"

    # --- Text Extended ---
    text_size: float = 0.6             # 0.3-2.0
    text_outline: bool = False
    text_outline_color: str = "#000000"

    # --- Tracker Migliorato ---
    persistence_fade: bool = False
    tracker_match_radius: int = 150    # 50-500

    # --- Camera ---
    camera_flip: bool = False

    # --- MediaPipe ---
    mp_pose_enabled: bool = False
    mp_hands_enabled: bool = False
    mp_face_enabled: bool = False
    mp_confidence: float = 0.5         # 0.1-1.0
    mp_hands_num_points: int = 5      # 1-21 landmark per hand tracked as blob
    mp_num_poses: int = 4             # 1-6 people for pose
    mp_num_faces: int = 2             # 1-4 faces
    mp_pose_num_points: int = 6       # 1-33
    mp_face_num_points: int = 7       # 1-15
    mp_blob_size: float = 1.0         # 0.5-3.0 radius multiplier
    mp_merge_distance: int = 0        # 0=off, px threshold for merge
    mp_gesture_size: bool = False     # pinch to scale
    mp_gesture_min: float = 0.3
    mp_gesture_max: float = 2.5

    # --- Silhouette ---
    silhouette_threshold: float = 0.5    # 0.1-0.9, segmenter mask threshold

    # --- NDI ---
    ndi_enabled: bool = False
    ndi_name: str = "BlobTrack"

    # --- Syphon (macOS) ---
    syphon_enabled: bool = False
    syphon_name: str = "BlobTrack"

    # --- Spout (Windows) ---
    spout_enabled: bool = False
    spout_name: str = "BlobTrack"


class LiveCameraConfig(BaseModel):
    camera_index: int = 0
    target_fps: float = 30.0
    resolution: Tuple[int, int] = (1280, 720)
    jpeg_quality: int = 92