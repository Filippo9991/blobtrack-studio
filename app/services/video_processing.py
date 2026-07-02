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


def _save_inputs(file_storage, audio_storage, uploads, job):
    """Salva video (e traccia audio opzionale) su disco. Va fatto nel thread
    della richiesta: i FileStorage muoiono con la richiesta stessa."""
    in_name = secure_filename(file_storage.filename) or "input.mp4"
    in_path = os.path.join(uploads, f"_in_{job}_{in_name}")
    file_storage.save(in_path)

    audio_path = None
    if audio_storage and getattr(audio_storage, "filename", ""):
        a_name = secure_filename(audio_storage.filename) or "audio"
        audio_path = os.path.join(uploads, f"_aud_{job}_{a_name}")
        audio_storage.save(audio_path)
    return in_path, audio_path


def _execute(job, in_path, audio_path, settings_dict, uploads, progress_callback=None):
    """Corpo dell'elaborazione (richiede un application context attivo)."""
    try:
        config = ProcessingConfig(**settings_dict).model_dump()
        config["input_path"] = in_path
        config["output_folder"] = uploads
        config["audio_enabled"] = audio_path is not None
        config["audio_path"] = audio_path
        # Limiti di durata/risoluzione (attivi in produzione, 0 = illimitato)
        config["limit_max_seconds"] = current_app.config.get("VIDEO_MAX_SECONDS", 0)
        config["limit_max_dim"] = current_app.config.get("VIDEO_MAX_DIM", 0)

        out_path = run_video(config, progress_callback=progress_callback)

        # Nome prevedibile (niente nome originale esposto)
        final_name = f"{job}.mp4"
        os.replace(out_path, os.path.join(uploads, final_name))
        return final_name
    finally:
        for path in (in_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)


def process_video(file_storage, settings_dict, audio_storage=None):
    """Elaborazione SINCRONA (fallback senza JS): blocca fino al risultato.

    Se `audio_storage` contiene una traccia, la reattività audio si attiva: il
    motore rileva i beat (spawn blob) ed esegue il mux con ffmpeg sul video finale.
    Solleva VideoBusyError se gli slot di elaborazione sono tutti occupati.
    """
    slots = _acquire_job_slot()
    try:
        uploads = uploads_dir()
        _cleanup_old_uploads(uploads)
        job = uuid.uuid4().hex
        in_path, audio_path = _save_inputs(file_storage, audio_storage, uploads, job)
        return _execute(job, in_path, audio_path, settings_dict, uploads)
    finally:
        slots.release()


# --- Job asincroni (percorso normale, con barra di avanzamento) --------------
# Registry in-process: con 1 worker gunicorn (Procfile/render.yaml) ogni job è
# visibile a tutte le richieste. Il thread aggiorna progress via il
# progress_callback del motore; il client fa polling su /video/status/<id>.

JOB_TTL_SECONDS = 3600  # i job conclusi restano consultabili per un'ora
_jobs = {}
_jobs_lock = threading.Lock()


def get_job(job_id, user_id):
    """Il job, solo se esiste ed è dell'utente indicato (altrimenti None)."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job or job["user_id"] != user_id:
            return None
        return dict(job)


def _purge_old_jobs():
    cutoff = time.time() - JOB_TTL_SECONDS
    with _jobs_lock:
        for jid in [j for j, d in _jobs.items() if d["created"] < cutoff]:
            del _jobs[jid]


def start_video_job(file_storage, settings_dict, audio_storage, user_id):
    """Avvia l'elaborazione in un thread e ritorna subito l'id del job.

    Solleva VideoBusyError se gli slot sono occupati (prima di salvare i file).
    """
    slots = _acquire_job_slot()
    try:
        _purge_old_jobs()
        uploads = uploads_dir()
        _cleanup_old_uploads(uploads)
        job_id = uuid.uuid4().hex
        in_path, audio_path = _save_inputs(file_storage, audio_storage, uploads, job_id)
    except Exception:
        slots.release()
        raise

    with _jobs_lock:
        _jobs[job_id] = {
            "state": "running", "progress": 0.0, "result": None,
            "error": None, "user_id": user_id, "created": time.time(),
        }

    app = current_app._get_current_object()  # l'app vera, utilizzabile nel thread
    threading.Thread(
        target=_run_job,
        args=(app, job_id, in_path, audio_path, settings_dict, uploads, slots),
        daemon=True,
    ).start()
    return job_id


def _run_job(app, job_id, in_path, audio_path, settings_dict, uploads, slots):
    def on_progress(fraction):
        with _jobs_lock:
            if job_id in _jobs:
                _jobs[job_id]["progress"] = max(0.0, min(1.0, float(fraction)))

    try:
        with app.app_context():
            final_name = _execute(
                job_id, in_path, audio_path, settings_dict, uploads,
                progress_callback=on_progress,
            )
        with _jobs_lock:
            _jobs[job_id].update(state="done", progress=1.0, result=final_name)
    except Exception as exc:
        app.logger.exception("Job video %s fallito", job_id)
        with _jobs_lock:
            _jobs[job_id].update(state="error", error=str(exc))
    finally:
        slots.release()
