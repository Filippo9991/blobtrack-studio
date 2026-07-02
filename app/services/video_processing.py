"""Adapter web → motore CV per l'elaborazione VIDEO.

Salva il video caricato, lo passa a `engine.run_video` (job con stato isolato per
l'intera sequenza: tracker, scie, smoothing) e ritorna il nome del file elaborato
nella cartella upload, servito dalla route protetta /uploads/<nome>.

L'elaborazione è CPU-bound e può durare minuti: un semaforo limita i job
concorrenti (VIDEO_MAX_JOBS, default 1) — chi arriva a slot pieni riceve
VideoBusyError e un invito a riprovare, invece di accodarsi fino al timeout.
"""
import os
import threading
import time
import uuid

from flask import current_app
from werkzeug.utils import secure_filename

from engine import ProcessingConfig, run_video

_slots_init_lock = threading.Lock()
_job_slots = None  # creato al primo job, dimensionato da VIDEO_MAX_JOBS


class VideoBusyError(RuntimeError):
    """Tutti gli slot di elaborazione video sono occupati."""


def _acquire_job_slot():
    global _job_slots
    with _slots_init_lock:
        if _job_slots is None:
            _job_slots = threading.Semaphore(int(current_app.config.get("VIDEO_MAX_JOBS", 1)))
    if not _job_slots.acquire(blocking=False):
        raise VideoBusyError(
            "Il server sta già elaborando un altro video: riprova tra qualche istante."
        )
    return _job_slots


def uploads_dir():
    """Cartella dei file elaborati (configurabile; default: static/uploads)."""
    path = current_app.config.get("UPLOAD_FOLDER") or os.path.join(
        current_app.static_folder, "uploads"
    )
    os.makedirs(path, exist_ok=True)
    return path


def _cleanup_old_uploads(uploads):
    """Rimuove output e temporanei più vecchi di UPLOADS_MAX_AGE_HOURS.

    Senza pulizia la cartella cresce per sempre (ogni job lascia un .mp4);
    eseguita all'avvio di ogni job, così non serve uno scheduler.
    """
    max_age = float(current_app.config.get("UPLOADS_MAX_AGE_HOURS", 24)) * 3600
    cutoff = time.time() - max_age
    try:
        for name in os.listdir(uploads):
            if name == ".gitkeep":
                continue
            path = os.path.join(uploads, name)
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
    except OSError:
        current_app.logger.warning("Pulizia uploads fallita", exc_info=True)


def process_video(file_storage, settings_dict, audio_storage=None):
    """Elabora il video caricato. Ritorna il nome del file output (in uploads_dir).

    Se `audio_storage` contiene una traccia, la reattività audio si attiva: il
    motore rileva i beat (spawn blob) ed esegue il mux con ffmpeg sul video finale.
    Solleva VideoBusyError se gli slot di elaborazione sono tutti occupati.
    """
    slots = _acquire_job_slot()
    uploads = uploads_dir()
    _cleanup_old_uploads(uploads)
    job = uuid.uuid4().hex

    in_name = secure_filename(file_storage.filename) or "input.mp4"
    in_path = os.path.join(uploads, f"_in_{job}_{in_name}")
    file_storage.save(in_path)

    # La traccia audio è opzionale: presenza del file = reattività attiva.
    audio_path = None
    if audio_storage and getattr(audio_storage, "filename", ""):
        a_name = secure_filename(audio_storage.filename) or "audio"
        audio_path = os.path.join(uploads, f"_aud_{job}_{a_name}")
        audio_storage.save(audio_path)

    try:
        config = ProcessingConfig(**settings_dict).model_dump()
        config["input_path"] = in_path
        config["output_folder"] = uploads
        config["audio_enabled"] = audio_path is not None
        config["audio_path"] = audio_path
        # Limiti di durata/risoluzione (attivi in produzione, 0 = illimitato)
        config["limit_max_seconds"] = current_app.config.get("VIDEO_MAX_SECONDS", 0)
        config["limit_max_dim"] = current_app.config.get("VIDEO_MAX_DIM", 0)

        out_path = run_video(config)

        # Nome prevedibile (niente nome originale esposto)
        final_name = f"{job}.mp4"
        os.replace(out_path, os.path.join(uploads, final_name))
        return final_name
    finally:
        slots.release()
        for path in (in_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)
