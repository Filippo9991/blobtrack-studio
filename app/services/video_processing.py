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


def process_video(file_storage, settings_dict):
    """Elabora il video caricato. Ritorna il nome del file output (in static/uploads)."""
    uploads = _uploads_dir()
    job = uuid.uuid4().hex

    in_name = secure_filename(file_storage.filename) or "input.mp4"
    in_path = os.path.join(uploads, f"_in_{job}_{in_name}")
    file_storage.save(in_path)

    try:
        config = ProcessingConfig(**settings_dict).model_dump()
        config["input_path"] = in_path
        config["output_folder"] = uploads

        out_path = run_video(config)

        # Nome prevedibile (niente nome originale esposto)
        final_name = f"{job}.mp4"
        os.replace(out_path, os.path.join(uploads, final_name))
        return final_name
    finally:
        if os.path.exists(in_path):
            os.remove(in_path)
