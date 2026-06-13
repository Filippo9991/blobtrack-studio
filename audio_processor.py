
import librosa
import numpy as np
import os

class AudioProcessor:
    def __init__(self):
        self.audio_path = None
        self.beat_frames = set() 
        self.sr = 22050
        self.hop_length = 512
        self.cache_features = None # Cache per feature avanzate se usate

    def analyze_beats(self, path, fps, band_focus='full', sensitivity=1.0):
        """
        Analisi Onset basata su bande di frequenza.
        """
        # Se il path o i parametri non sono cambiati drasticamente, potremmo ottimizzare,
        # ma per sicurezza ricalcoliamo se cambia la banda.
        if not os.path.exists(path):
            print(f"Audio file not found: {path}")
            return set()

        # print(f"--- ANALISI AUDIO: Band '{band_focus}' ---")
        
        # 1. Caricamento (solo se cambiato file per risparmiare I/O)
        if self.audio_path != path:
            self.y, self.sr = librosa.load(path, sr=self.sr)
            self.audio_path = path
        
        y = self.y
        sr = self.sr
        
        # 2. Calcolo dello Spettrogramma (Energia per frequenza)
        S = np.abs(librosa.stft(y, n_fft=2048, hop_length=self.hop_length))
        fft_freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # 3. Mascheramento della Banda
        mask = np.ones_like(S, dtype=bool)
        
        ranges = {
            'sub': (0, 60),
            'bass': (0, 250),           
            'low_mids': (250, 2000),    
            'high_mids': (2000, 6000),  
            'highs': (6000, 22000),     
            'full': (0, 22000)
        }
        
        target_range = ranges.get(band_focus, ranges['full'])
        f_min, f_max = target_range
        
        valid_bins = (fft_freqs >= f_min) & (fft_freqs <= f_max)
        S_filtered = S * valid_bins[:, np.newaxis]
        
        # 4. Calcolo Onset Strength
        onset_env = librosa.onset.onset_strength(S=librosa.amplitude_to_db(S_filtered, ref=np.max), 
                                                 sr=sr, 
                                                 aggregate=np.median)
        
        # 5. Normalizzazione soglia
        delta_val = 0.07 / max(0.1, sensitivity)
        
        # 6. Peak Picking (CORRETTO PER LIBROSA 0.10+)
        # I parametri devono essere passati come keyword arguments
        peaks = librosa.util.peak_pick(onset_env, 
                                     pre_max=3, 
                                     post_max=3, 
                                     pre_avg=3, 
                                     post_avg=5, 
                                     delta=delta_val, 
                                     wait=int(fps/5)) 
        
        # 7. Conversione in Frame
        times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.hop_length)
        self.beat_frames = set([int(t * fps) for t in times])
        
        # print(f"Analisi completata: trovati {len(self.beat_frames)} eventi.")
        return self.beat_frames

    def analyze_full_features(self, path, fps, total_frames):
        """
        Metodo opzionale per estrarre features continue (RMS, Centroid, Flux)
        Usato se si vuole modulare parametri come raggio/spessore in modo continuo.
        """
        if not os.path.exists(path): return None

        # Cache: reuse result if same file/fps/total_frames
        cache_key = (path, fps, total_frames)
        if self.cache_features is not None and getattr(self, '_cache_features_key', None) == cache_key:
            return self.cache_features

        # Reuse loaded audio if same file
        if self.audio_path != path:
            self.y, self.sr = librosa.load(path, sr=self.sr)
            self.audio_path = path

        y = self.y
        sr = self.sr
        duration = librosa.get_duration(y=y, sr=sr)
        
        # RMS (Energia)
        rms = librosa.feature.rms(y=y, hop_length=self.hop_length)[0]
        
        # Centroid (Timbro)
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=self.hop_length)[0]
        
        # Flux (Onset strength come proxy)
        flux = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)

        # Helper per interpolare alla lunghezza dei frame video
        def resample(data, target_len):
            original_steps = np.linspace(0, duration, len(data))
            target_steps = np.linspace(0, duration, target_len)
            return np.interp(target_steps, original_steps, data)
        
        # Normalizzazione helper
        def norm(data):
            return (data - np.min(data)) / (np.max(data) - np.min(data) + 1e-6)

        result = {
            'rms': norm(resample(rms, total_frames)),
            'centroid': norm(resample(centroid, total_frames)),
            'flux': norm(resample(flux, total_frames))
        }
        self.cache_features = result
        self._cache_features_key = cache_key
        return result
