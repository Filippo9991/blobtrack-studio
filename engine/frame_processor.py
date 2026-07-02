import cv2
import numpy as np

import logging

logger = logging.getLogger("blobtrack.engine")

class FrameProcessor:
    def __init__(self, config):
        self.enabled = config.get('preprocess_enabled', False)
        self.method = config.get('preprocess_method', 'CrowdBoost')
        self.strength = config.get('preprocess_strength', 1.0)
        
        # Setup CLAHE con parametri ottimizzati per crowd
        clip_limit = 2.0 + (self.strength * 3.0)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        
        # Cache per kernel pre-calcolati
        self._kernel_cache = {}

    def _get_kernel(self, kernel_type, size=3):
        """Cache per kernel morfologici - evita ricalcoli"""
        key = f"{kernel_type}_{size}"
        if key not in self._kernel_cache:
            if kernel_type == 'ellipse':
                self._kernel_cache[key] = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            elif kernel_type == 'rect':
                self._kernel_cache[key] = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
        return self._kernel_cache[key]

    def process(self, frame):
        if not self.enabled:
            return frame

        try:
            # === CROWD BOOST ULTRA (OTTIMIZZATO PER 1080p) ===
            if self.method == 'CrowdBoost':
                # 1. Pre-Denoising intelligente (rimuove rumore video senza sfuocare)
                # Bilateral filter preserva i bordi mentre riduce il rumore
                d = 5  # Diametro piccolo per velocità
                sigma_color = 50 + (self.strength * 50)
                sigma_space = 50 + (self.strength * 50)
                frame = cv2.bilateralFilter(frame, d, sigma_color, sigma_space)

                # 2. Gamma Correction Adattiva (Schiarisce ombre in modo non lineare)
                gamma = 1.0 / (1.0 + (self.strength * 0.6))
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                frame = cv2.LUT(frame, table)

                # 3. CLAHE Avanzato su LAB (Separazione ottimale persone)
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                
                # CLAHE con parametri dinamici
                cl = 2.5 + (self.strength * 2.5)
                tile_size = 8  # 8x8 ottimale per crowd
                clahe_algo = cv2.createCLAHE(clipLimit=cl, tileGridSize=(tile_size, tile_size))
                l_eq = clahe_algo.apply(l)
                
                # Ricomponi
                updated_lab = cv2.merge((l_eq, a, b))
                frame = cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

                # 4. Unsharp Mask Potenziato (Bordi ultra-definiti)
                gaussian = cv2.GaussianBlur(frame, (0, 0), 2.5)
                amount = 0.8 + (self.strength * 1.2)
                frame = cv2.addWeighted(frame, 1.0 + amount, gaussian, -amount, 0)
                
                # 5. Contrast Stretching (Massimizza range dinamico)
                # Migliora separazione in scene con poco contrasto
                frame = cv2.normalize(frame, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                
                return frame

            # === METODO NUOVO: EDGE BOOST (Per folle dense) ===
            elif self.method == 'EdgeBoost':
                # Specifico per separare persone molto vicine
                
                # 1. Preprocessing base
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 2. Bilateral per smoothing intelligente
                denoised = cv2.bilateralFilter(gray, 5, 50, 50)
                
                # 3. CLAHE aggressivo
                clahe = cv2.createCLAHE(clipLimit=3.0 + self.strength * 2, tileGridSize=(8,8))
                enhanced = clahe.apply(denoised)
                
                # 4. Sobel edge detection multi-scala
                # Cattura sia bordi sottili che spessi
                sobelx = cv2.Sobel(enhanced, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(enhanced, cv2.CV_64F, 0, 1, ksize=3)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edges = np.uint8(np.clip(edges, 0, 255))
                
                # 5. Blend edges con originale
                edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
                blend_amount = 0.3 + (self.strength * 0.4)
                result = cv2.addWeighted(frame, 1.0, edges_bgr, blend_amount, 0)
                
                # 6. Final contrast boost
                result = cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
                
                return result

            # === METODO NUOVO: MOTION ENHANCE (Per video con movimento) ===
            elif self.method == 'MotionEnhance':
                # Ottimizzato per crowd in movimento (concerti, manifestazioni)
                
                # 1. Riduzione motion blur
                kernel_size = 5
                kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size**2)
                deblurred = cv2.filter2D(frame, -1, -kernel)
                sharpened = cv2.addWeighted(frame, 1.5, deblurred, -0.5, 0)
                
                # 2. CLAHE per contrasto
                lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = self.clahe.apply(l)
                result = cv2.merge((l_eq, a, b))
                result = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)
                
                # 3. High-pass filter (enfatizza dettagli)
                gaussian = cv2.GaussianBlur(result, (0, 0), 3)
                result = cv2.addWeighted(result, 1.5 + self.strength, gaussian, -0.5 - self.strength*0.5, 0)
                
                return result

            # === METODO NUOVO: THERMAL VISION (Massimo contrasto) ===
            elif self.method == 'ThermalVision':
                # Simula visione termica - massima separazione blob
                
                # 1. Converti in grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 2. CLAHE estremo
                clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4,4))
                enhanced = clahe.apply(gray)
                
                # 3. Applica colormap "caldo"
                # Le persone appaiono come "macchie calde"
                colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
                
                # 4. Boost contrasto finale
                colored = cv2.convertScaleAbs(colored, alpha=1.2 + self.strength*0.5, beta=10)
                
                return colored

            # --- METODI LEGACY OTTIMIZZATI ---
            elif self.method == 'CLAHE':
                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                l_eq = self.clahe.apply(l)
                updated_lab = cv2.merge((l_eq, a, b))
                return cv2.cvtColor(updated_lab, cv2.COLOR_LAB2BGR)

            elif self.method == 'DetailEnhance':
                # Ottimizzato con parametri migliori
                sigma_s = 15 + (self.strength * 40)
                sigma_r = 0.10 + (self.strength * 0.10)
                return cv2.detailEnhance(frame, sigma_s=sigma_s, sigma_r=sigma_r)

            elif self.method == 'Sharpen':
                # Unsharp mask invece di kernel fisso
                gaussian = cv2.GaussianBlur(frame, (0, 0), 2)
                amount = self.strength * 1.5
                return cv2.addWeighted(frame, 1.0 + amount, gaussian, -amount, 0)

            elif self.method == 'Gamma':
                gamma = 2.0 / (self.strength + 0.1)
                inv_gamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
                return cv2.LUT(frame, table)

            return frame
            
        except Exception as e:
            logger.warning(f"Preprocessing Error: {e}")
            return frame