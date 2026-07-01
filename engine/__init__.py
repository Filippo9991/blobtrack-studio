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
import importlib.util
import shutil
import threading
from functools import lru_cache

from engine import options
from engine.schemas import ProcessingConfig

__all__ = [
    "ProcessingConfig", "options", "process_image_frame", "run_video",
    "capabilities", "create_live_engine",
]


def create_live_engine():
    """Istanza dedicata di LiveEngine, con stato di tracking persistente.

    Per lo streaming live: il chiamante la tiene per la durata della sessione
    (scie e ID stabili si accumulano fra i frame) e chiama
    `engine.process_frame(frame, config, mod_energy=...)` per ogni frame.
    I modelli pesanti restano lazy: senza YOLO/MediaPipe si degrada da soli.
    """
    from engine.blob_engine import LiveEngine

    return LiveEngine()


@lru_cache(maxsize=1)
def capabilities():
    """Dipendenze opzionali disponibili nell'ambiente corrente.

    Il progetto ha due profili: `requirements.txt` (lite, deploy) e
    `requirements-local.txt` (full). Il layer web usa questa mappa per
    nascondere le funzioni non disponibili. `find_spec` controlla la presenza
    dei package senza importarli (niente caricamento di torch all'avvio).
    """
    return {
        "yolo": _has_module("ultralytics"),
        "mediapipe": _has_module("mediapipe"),
        "audio": _has_module("librosa"),
        "ffmpeg": shutil.which("ffmpeg") is not None,
    }


def _has_module(name):
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):  # package rovinato/bloccato = non disponibile
        return False

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
