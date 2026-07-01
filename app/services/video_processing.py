"""Adapter web → motore CV per l'elaborazione VIDEO.

Salva il video caricato, lo passa a `engine.run_video` (job con stato isolato per
l'intera sequenza: tracker, scie, smoothing) e ritorna il nome del file elaborato
in static/uploads/, pronto per download/anteprima.
"""
import os
import uuid

from flask import current_app
from werkzeug.utils import secure_filename

from engine import ProcessingConfig, run_video


def _uploads_dir():
    path = os.path.join(current_app.static_folder, "uploads")
    os.makedirs(path, exist_ok=True)
    return path


def process_video(file_storage, settings_dict, audio_storage=None):
    """Elabora il video caricato. Ritorna il nome del file output (in static/uploads).

    Se `audio_storage` contiene una traccia, la reattività audio si attiva: il
    motore rileva i beat (spawn blob) ed esegue il mux con ffmpeg sul video finale.
    """
    uploads = _uploads_dir()
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
        for path in (in_path, audio_path):
            if path and os.path.exists(path):
                os.remove(path)
