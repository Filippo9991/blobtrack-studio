"""Motore di computer vision di BlobTrack — package indipendente da Flask.

API pubblica (il layer web importa solo da qui):

    from engine import ProcessingConfig, options
    from engine import process_image_frame, run_video

- `process_image_frame(frame, config)` → modalità IMMAGINE: stato di tracking
  azzerato ad ogni chiamata (i modelli restano in cache), così richieste
  indipendenti non si contaminano.
- `run_video(config, progress_callback)` → modalità VIDEO (Pass 2): job con
  stato isolato per l'intera sequenza.
"""
import threading

from engine import options
from engine.schemas import ProcessingConfig

__all__ = ["ProcessingConfig", "options", "process_image_frame", "run_video"]

_engine = None
_engine_lock = threading.Lock()


def _get_live_engine():
    """Singleton lazy: carica torch/mediapipe solo alla prima richiesta."""
    global _engine
    if _engine is None:
        from engine.blob_engine import LiveEngine

        _engine = LiveEngine()
    return _engine


def process_image_frame(frame, config):
    """Elabora un singolo frame BGR e ritorna il frame renderizzato.

    Lo stato di tracking viene azzerato prima di ogni elaborazione; la chiamata
    è serializzata perché l'engine condiviso ha stato mutabile (modelli in cache).
    """
    engine = _get_live_engine()
    with _engine_lock:
        engine.reset_state()
        return engine.process_frame(frame, config)


def run_video(config, progress_callback=None):
    """Elaborazione video completa (trails/audio/tracker). Usata dal Pass 2."""
    from engine.blob_engine import run_processing

    return run_processing(config, progress_callback=progress_callback)
