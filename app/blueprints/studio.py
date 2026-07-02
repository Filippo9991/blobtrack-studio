"""Studio CV: upload immagine, detection avanzata, galleria creazioni e preset."""
import base64
import json
import threading

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    session,
    url_for,
)

from engine import capabilities

from app.decorators import login_required
from app.extensions import db
from app.forms import DeleteForm, LiveForm, MediaPipeForm, StudioForm
from app.models import Creation, Preset
from app.services.frame_engine import (
    render_frame_jpeg,
    render_image,
    render_live_frame_jpeg,
)
from app.services.video_processing import (
    VideoBusyError,
    get_job,
    process_video,
    start_video_job,
    uploads_dir,
)

studio_bp = Blueprint("studio", __name__)

# Campi del form che NON fanno parte del config del motore.
# I campi-config sono quindi "tutti gli altri": il form è l'unica fonte di verità
# (aggiungere un parametro al form lo include automaticamente nel config).
_NON_CONFIG_FIELDS = {
    "csrf_token", "source", "image", "video", "audio",
    "snapshot_raw", "preset_name", "submit",
}

IMAGE_EXTS = {"jpg", "jpeg", "png", "webp", "bmp"}
VIDEO_EXTS = {"mp4", "mov", "avi", "webm", "mkv"}


def _is_video_upload(file_storage):
    name = getattr(file_storage, "filename", "") or ""
    return "." in name and name.rsplit(".", 1)[1].lower() in VIDEO_EXTS


def _form_to_settings(form):
    return {
        name: field.data
        for name, field in form._fields.items()
        if name not in _NON_CONFIG_FIELDS
    }


def _apply_settings_to_form(form, settings):
    for name, field in form._fields.items():
        if name not in _NON_CONFIG_FIELDS and settings.get(name) is not None:
            field.data = settings[name]


@studio_bp.route("/studio", methods=["GET", "POST"])
@login_required
def studio():
    form = StudioForm()

    if request.method == "GET":
        preset_id = request.args.get("preset", type=int)
        if preset_id:
            preset = Preset.query.filter_by(id=preset_id, user_id=session["user_id"]).first()
            if preset:
                _apply_settings_to_form(form, json.loads(preset.config))
                flash(f"Preset «{preset.name}» caricato. Carica una sorgente ed elabora.", "success")
            else:
                flash("Preset non trovato.", "error")

        # Risultato di un job video asincrono concluso (il JS reindirizza qui)
        job_id = request.args.get("job")
        if job_id:
            job = get_job(job_id, session["user_id"])
            if job and job["state"] == "done":
                return render_template("studio.html", form=form, preview=None, result=job["result"])
            flash("Elaborazione non trovata o non ancora conclusa.", "warning")
        return render_template("studio.html", form=form, preview=None, result=None)

    # Il JS invia con questo header (per risposte JSON); i POST classici sono il
    # fallback senza JavaScript.
    is_fetch = request.headers.get("X-Requested-With") == "fetch"

    if not form.validate_on_submit():
        if is_fetch:
            msgs = [e for errs in form.errors.values() for e in errs]
            return {"error": " ".join(msgs) or "Dati non validi."}, 400
        return render_template("studio.html", form=form, preview=None, result=None)

    action = request.form.get("action", "preview")
    settings = _form_to_settings(form)

    # 1) Salvataggio preset: non serve una sorgente
    if action == "save_preset":
        name = (form.preset_name.data or "").strip()
        if not name:
            msg = "Dai un nome al preset per salvarlo."
            if is_fetch:
                return {"error": msg}, 400
            flash(msg, "warning")
            return render_template("studio.html", form=form, preview=None, result=None)
        preset = Preset(
            user_id=session["user_id"], name=name,
            config=json.dumps(settings), source="manual",
        )
        db.session.add(preset)
        db.session.commit()
        if is_fetch:
            return {"ok": True, "redirect": url_for("studio.presets")}
        flash(f"Preset «{name}» salvato.", "success")
        return redirect(url_for("studio.presets"))

    # 2) Render VIDEO: serve una sorgente video; elaborazione completa (asincrona)
    if action == "render_video":
        src = form.source.data
        if not _is_video_upload(src):
            msg = "Carica un video per l'elaborazione."
            if is_fetch:
                return {"error": msg}, 400
            flash(msg, "warning")
            return render_template("studio.html", form=form, preview=None, result=None)

        audio_file = form.audio.data
        if audio_file and not capabilities()["audio"]:
            audio_file = None  # profilo lite: la UI nasconde il campo, ma un POST può arrivare
            flash("Reattività audio non disponibile su questo server: elaboro senza traccia.", "warning")

        if is_fetch:
            try:
                job_id = start_video_job(src, settings, audio_file, session["user_id"])
            except VideoBusyError as exc:
                return {"error": str(exc)}, 429
            except Exception:
                current_app.logger.exception("Avvio job video fallito")
                return {"error": "Avvio dell'elaborazione fallito."}, 500
            return {"job": job_id}

        try:  # fallback sincrono senza JS
            result = process_video(src, settings, audio_file)
        except VideoBusyError as exc:
            flash(str(exc), "warning")
            return render_template("studio.html", form=form, preview=None, result=None)
        except Exception as exc:
            current_app.logger.exception("Errore di elaborazione video")
            flash(f"Elaborazione video fallita: {exc}", "error")
            return render_template("studio.html", form=form, preview=None, result=None)
        flash("Video elaborato.", "success")
        return render_template("studio.html", form=form, preview=None, result=result)

    # 3) IMMAGINE — save/preview. I pixel arrivano dal frame catturato dal client
    #    (snapshot_raw: immagine o fotogramma di un video) o dal file immagine.
    raw = _decode_data_url(form.snapshot_raw.data)
    src = form.source.data
    if raw is None and src and getattr(src, "filename", "") and not _is_video_upload(src):
        raw = src.read()
    if raw is None:
        msg = "Carica un'immagine, o un video da cui salvare un fotogramma."
        if is_fetch:
            return {"error": msg}, 400
        flash(msg, "warning")
        return render_template("studio.html", form=form, preview=None, result=None)

    try:
        png_bytes = render_image(raw, settings)
    except ValueError as exc:
        if is_fetch:
            return {"error": str(exc)}, 400
        flash(str(exc), "error")
        return render_template("studio.html", form=form, preview=None, result=None)
    except Exception as exc:  # YOLO/MediaPipe possono fallire: non esporre lo stacktrace
        current_app.logger.exception("Errore di elaborazione")
        if is_fetch:
            return {"error": "Elaborazione fallita."}, 500
        flash(f"Elaborazione fallita: {exc}", "error")
        return render_template("studio.html", form=form, preview=None, result=None)

    if action == "save":
        default_title = getattr(src, "filename", "") or "Creazione"
        title = (form.preset_name.data or default_title).strip()[:120]
        creation = Creation(
            user_id=session["user_id"], title=title,
            image_data=png_bytes, settings=json.dumps(settings),
        )
        db.session.add(creation)
        db.session.commit()
        if is_fetch:
            return {"ok": True, "redirect": url_for("studio.dashboard")}
        flash("Creazione salvata nella galleria.", "success")
        return redirect(url_for("studio.dashboard"))

    # preview (fallback no-JS per le immagini: con JS l'anteprima è live nel browser)
    preview = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    flash("Anteprima generata.", "success")
    return render_template("studio.html", form=form, preview=preview, result=None)


@studio_bp.route("/dashboard")
@login_required
def dashboard():
    creations = (
        Creation.query.filter_by(user_id=session["user_id"])
        .order_by(Creation.created_at.desc())
        .all()
    )
    return render_template("dashboard.html", creations=creations, delete_form=DeleteForm())


@studio_bp.route("/creation/<int:creation_id>/image")
@login_required
def creation_image(creation_id):
    """L'immagine di una creazione; con ?download=1 arriva come file scaricato."""
    from werkzeug.utils import secure_filename

    creation = Creation.query.filter_by(id=creation_id, user_id=session["user_id"]).first_or_404()
    headers = {}
    if request.args.get("download") == "1":
        name = secure_filename(creation.title) or f"creazione-{creation.id}"
        headers["Content-Disposition"] = f'attachment; filename="{name}.png"'
    return Response(creation.image_data, mimetype="image/png", headers=headers)


@studio_bp.route("/creation/<int:creation_id>/delete", methods=["POST"])
@login_required
def delete_creation(creation_id):
    if DeleteForm().validate_on_submit():
        creation = Creation.query.filter_by(id=creation_id, user_id=session["user_id"]).first_or_404()
        db.session.delete(creation)
        db.session.commit()
        flash("Creazione eliminata.", "success")
    return redirect(url_for("studio.dashboard"))


@studio_bp.route("/presets")
@login_required
def presets():
    items = (
        Preset.query.filter_by(user_id=session["user_id"])
        .order_by(Preset.created_at.desc())
        .all()
    )
    return render_template("presets.html", presets=items, delete_form=DeleteForm())


@studio_bp.route("/preset/<int:preset_id>/delete", methods=["POST"])
@login_required
def delete_preset(preset_id):
    if DeleteForm().validate_on_submit():
        preset = Preset.query.filter_by(id=preset_id, user_id=session["user_id"]).first_or_404()
        db.session.delete(preset)
        db.session.commit()
        flash("Preset eliminato.", "success")
    return redirect(url_for("studio.presets"))


@studio_bp.route("/video")
@login_required
def video():
    """Compat: la sezione Video è stata fusa nello Studio unico (immagine+video)."""
    preset = request.args.get("preset")
    job = request.args.get("job")
    return redirect(url_for("studio.studio", preset=preset, job=job))


@studio_bp.route("/video/status/<job_id>")
@login_required
def video_status(job_id):
    """Stato di un job asincrono (polling dal client, solo il proprietario)."""
    job = get_job(job_id, session["user_id"])
    if not job:
        return {"error": "Elaborazione non trovata."}, 404
    payload = {"state": job["state"], "progress": round(job["progress"], 3)}
    if job["state"] == "done":
        payload["redirect"] = url_for("studio.studio", job=job_id)
    elif job["state"] == "error":
        payload["error"] = job["error"]
    return payload


@studio_bp.route("/uploads/<filename>")
@login_required
def upload_file(filename):
    """Serve i video elaborati dalla cartella upload (route protetta).

    Sostituisce il vecchio link diretto a static/uploads: i file non sono più
    raggiungibili senza login e la cartella può stare fuori da static/
    (UPLOAD_FOLDER). send_from_directory blocca i path traversal.
    Con ?download=1 forza il download (Content-Disposition: attachment)."""
    if request.args.get("download") == "1":
        return send_from_directory(
            uploads_dir(), filename,
            as_attachment=True, download_name=f"blobtrack-{filename}",
        )
    return send_from_directory(uploads_dir(), filename)


def _decode_data_url(data_url):
    """Estrae i byte da un data URL ("data:image/jpeg;base64,..."). None se invalido."""
    if not data_url or "," not in data_url:
        return None
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


def _load_preset_into(form, snippet):
    """GET delle pagine webcam: precarica un preset se richiesto in querystring."""
    preset_id = request.args.get("preset", type=int)
    if not preset_id:
        return
    preset = Preset.query.filter_by(id=preset_id, user_id=session["user_id"]).first()
    if preset:
        _apply_settings_to_form(form, json.loads(preset.config))
        flash(f"Preset «{preset.name}» caricato. {snippet}", "success")
    else:
        flash("Preset non trovato.", "error")


def _handle_webcam_save(form, template):
    """POST condiviso da /live e /mediapipe: salva snapshot (in galleria) o preset.

    I frame elaborati passano da /live/frame; qui si gestisce solo il salvataggio
    (form classico con CSRF). Lo snapshot rielabora server-side il frame grezzo.
    """
    if not form.validate_on_submit():
        return render_template(template, form=form)

    action = request.form.get("action", "save_preset")
    settings = _form_to_settings(form)

    if action == "save_snapshot":
        raw = _decode_data_url(request.form.get("snapshot_raw"))
        if not raw:
            flash("Nessun frame da salvare: avvia la camera e riprova.", "warning")
            return render_template(template, form=form)
        try:
            png_bytes = render_image(raw, settings)
        except Exception as exc:  # YOLO/MediaPipe/IO: non esporre lo stacktrace
            current_app.logger.exception("Errore snapshot webcam")
            flash(f"Snapshot fallito: {exc}", "error")
            return render_template(template, form=form)
        title = (form.preset_name.data or "Snapshot").strip()[:120]
        creation = Creation(
            user_id=session["user_id"], title=title,
            image_data=png_bytes, settings=json.dumps(settings),
        )
        db.session.add(creation)
        db.session.commit()
        flash("Snapshot salvato nella galleria.", "success")
        return redirect(url_for("studio.dashboard"))

    # default: save_preset (interscambiabile con Studio: stessa config)
    name = (form.preset_name.data or "").strip()
    if not name:
        flash("Dai un nome al preset per salvarlo.", "warning")
        return render_template(template, form=form)
    preset = Preset(
        user_id=session["user_id"], name=name,
        config=json.dumps(settings), source="manual",
    )
    db.session.add(preset)
    db.session.commit()
    flash(f"Preset «{name}» salvato.", "success")
    return redirect(url_for("studio.presets"))


@studio_bp.route("/live", methods=["GET", "POST"])
@login_required
def live():
    """Live cam (color/YOLO): webcam dal browser, elaborazione su /live/frame."""
    form = LiveForm()
    if request.method == "GET":
        _load_preset_into(form, "Avvia la camera.")
        return render_template("live.html", form=form)
    return _handle_webcam_save(form, "live.html")


@studio_bp.route("/mediapipe", methods=["GET", "POST"])
@login_required
def mediapipe():
    """Pagina MediaPipe (webcam): pose/mani/volto senza i blob color.

    Stesso flusso del Live (frame su /live/frame, stato per-stream), ma
    `detection_engine='mediapipe'` → il motore salta la detection di base."""
    if not capabilities()["mediapipe"]:
        flash("MediaPipe non è disponibile su questo server (solo versione locale).", "warning")
        return redirect(url_for("studio.studio"))

    form = MediaPipeForm()
    if request.method == "GET":
        _load_preset_into(form, "Avvia la camera.")
        if not request.args.get("preset"):
            form.mp_pose_enabled.data = True  # parti con qualcosa da tracciare
        return render_template("mediapipe.html", form=form)
    return _handle_webcam_save(form, "mediapipe.html")


def _clamp_audio_level(value):
    """Livello mic dal client → float 0..1 (input non fidato)."""
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


# Guardia per-utente su /live/frame: il client corretto tiene UNA richiesta in
# volo, ma un client rotto (o ostile) potrebbe martellare l'endpoint più
# CPU-intensivo dell'app. Un solo frame in elaborazione per utente: gli altri
# ricevono 429 e il loop del client semplicemente salta il frame.
_live_busy_lock = threading.Lock()
_live_busy_users = set()


@studio_bp.route("/live/frame", methods=["POST"])
@login_required
def live_frame():
    """Riceve un frame (data URL JPEG) + config JSON, ritorna il frame elaborato.

    Endpoint di sola elaborazione (nessuna scrittura su DB): chiamato in loop dal
    client. Risponde con i byte JPEG così il browser li mostra direttamente.
    Se il client manda un id di `stream`, lo stato di tracking (scie, ID) persiste
    fra i frame di quello stream; `audio_level` è il livello del microfono."""
    data = request.get_json(silent=True) or {}
    raw = _decode_data_url(data.get("frame"))
    if not raw:
        return {"error": "Frame mancante o non valido."}, 400

    config = data.get("config") or {}
    stream = str(data.get("stream") or "")[:64]
    audio_level = _clamp_audio_level(data.get("audio_level"))

    uid = session["user_id"]
    with _live_busy_lock:
        if uid in _live_busy_users:
            return {"error": "Frame precedente ancora in elaborazione."}, 429
        _live_busy_users.add(uid)
    try:
        if stream:
            session_key = f"{uid}:{stream}"
            jpeg = render_live_frame_jpeg(raw, config, session_key, audio_level=audio_level)
        else:  # client senza stream id: elaborazione stateless come prima
            jpeg = render_frame_jpeg(raw, config)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception:  # YOLO/MediaPipe possono fallire: non esporre lo stacktrace
        current_app.logger.exception("Errore di elaborazione live")
        return {"error": "Elaborazione fallita."}, 500
    finally:
        with _live_busy_lock:
            _live_busy_users.discard(uid)

    return Response(jpeg, mimetype="image/jpeg")
