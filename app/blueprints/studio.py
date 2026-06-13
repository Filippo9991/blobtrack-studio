"""Studio CV: upload immagine, detection avanzata, galleria creazioni e preset."""
import base64
import json

from flask import (
    Blueprint,
    Response,
    current_app,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from app.decorators import login_required
from app.extensions import db
from app.forms import DeleteForm, LiveForm, StudioForm, VideoForm
from app.models import Creation, Preset
from app.services.frame_engine import render_frame_jpeg, render_image
from app.services.video_processing import process_video

studio_bp = Blueprint("studio", __name__)

# Campi dei form (Studio/Video) che NON fanno parte del config del motore.
# I campi-config sono quindi "tutti gli altri": il form è l'unica fonte di verità
# (aggiungere un parametro al form lo include automaticamente nel config).
_NON_CONFIG_FIELDS = {"csrf_token", "image", "video", "audio", "preset_name", "submit"}


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
                flash(f"Preset «{preset.name}» caricato. Carica un'immagine ed elabora.", "success")
            else:
                flash("Preset non trovato.", "error")
        return render_template("studio.html", form=form, preview=None)

    if not form.validate_on_submit():
        return render_template("studio.html", form=form, preview=None)

    action = request.form.get("action", "preview")
    settings = _form_to_settings(form)

    # Salvataggio preset: non richiede un'immagine
    if action == "save_preset":
        name = (form.preset_name.data or "").strip()
        if not name:
            flash("Dai un nome al preset per salvarlo.", "warning")
            return render_template("studio.html", form=form, preview=None)
        preset = Preset(
            user_id=session["user_id"], name=name,
            config=json.dumps(settings), source="manual",
        )
        db.session.add(preset)
        db.session.commit()
        flash(f"Preset «{name}» salvato.", "success")
        return redirect(url_for("studio.presets"))

    # preview / save: serve un'immagine
    file = form.image.data
    if not file or not getattr(file, "filename", ""):
        flash("Carica un'immagine per elaborarla.", "warning")
        return render_template("studio.html", form=form, preview=None)

    try:
        png_bytes = render_image(file.read(), settings)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("studio.html", form=form, preview=None)
    except Exception as exc:  # YOLO/MediaPipe possono fallire: non esporre lo stacktrace
        current_app.logger.exception("Errore di elaborazione")
        flash(f"Elaborazione fallita: {exc}", "error")
        return render_template("studio.html", form=form, preview=None)

    if action == "save":
        title = (form.preset_name.data or file.filename or "Creazione").strip()[:120]
        creation = Creation(
            user_id=session["user_id"], title=title,
            image_data=png_bytes, settings=json.dumps(settings),
        )
        db.session.add(creation)
        db.session.commit()
        flash("Creazione salvata nella galleria.", "success")
        return redirect(url_for("studio.dashboard"))

    preview = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    flash("Anteprima generata.", "success")
    return render_template("studio.html", form=form, preview=preview)


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
    creation = Creation.query.filter_by(id=creation_id, user_id=session["user_id"]).first_or_404()
    return Response(creation.image_data, mimetype="image/png")


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


@studio_bp.route("/video", methods=["GET", "POST"])
@login_required
def video():
    form = VideoForm()

    if request.method == "GET":
        preset_id = request.args.get("preset", type=int)
        if preset_id:
            preset = Preset.query.filter_by(id=preset_id, user_id=session["user_id"]).first()
            if preset:
                _apply_settings_to_form(form, json.loads(preset.config))
                flash(f"Preset «{preset.name}» caricato.", "success")
        return render_template("video.html", form=form, result=None)

    if not form.validate_on_submit():
        return render_template("video.html", form=form, result=None)

    settings = _form_to_settings(form)
    try:
        result = process_video(form.video.data, settings, form.audio.data)
    except Exception as exc:  # codec/IO/engine: non esporre lo stacktrace
        current_app.logger.exception("Errore di elaborazione video")
        flash(f"Elaborazione video fallita: {exc}", "error")
        return render_template("video.html", form=form, result=None)

    flash("Video elaborato.", "success")
    return render_template("video.html", form=form, result=result)


def _decode_data_url(data_url):
    """Estrae i byte da un data URL ("data:image/jpeg;base64,..."). None se invalido."""
    if not data_url or "," not in data_url:
        return None
    try:
        return base64.b64decode(data_url.split(",", 1)[1])
    except (ValueError, TypeError):
        return None


@studio_bp.route("/live", methods=["GET", "POST"])
@login_required
def live():
    """Live cam: la webcam la fornisce il browser; qui si gestiscono solo i
    parametri di stile e il salvataggio di preset/snapshot. I frame elaborati
    in tempo (quasi) reale passano dall'endpoint AJAX /live/frame."""
    form = LiveForm()

    if request.method == "GET":
        preset_id = request.args.get("preset", type=int)
        if preset_id:
            preset = Preset.query.filter_by(id=preset_id, user_id=session["user_id"]).first()
            if preset:
                _apply_settings_to_form(form, json.loads(preset.config))
                flash(f"Preset «{preset.name}» caricato. Avvia la camera.", "success")
            else:
                flash("Preset non trovato.", "error")
        return render_template("live.html", form=form)

    if not form.validate_on_submit():
        return render_template("live.html", form=form)

    action = request.form.get("action", "save_preset")
    settings = _form_to_settings(form)

    if action == "save_snapshot":
        raw = _decode_data_url(request.form.get("snapshot_raw"))
        if not raw:
            flash("Nessun frame da salvare: avvia la camera e riprova.", "warning")
            return render_template("live.html", form=form)
        try:
            png_bytes = render_image(raw, settings)
        except Exception as exc:  # YOLO/MediaPipe/IO: non esporre lo stacktrace
            current_app.logger.exception("Errore snapshot live")
            flash(f"Snapshot fallito: {exc}", "error")
            return render_template("live.html", form=form)
        title = (form.preset_name.data or "Live snapshot").strip()[:120]
        creation = Creation(
            user_id=session["user_id"], title=title,
            image_data=png_bytes, settings=json.dumps(settings),
        )
        db.session.add(creation)
        db.session.commit()
        flash("Snapshot salvato nella galleria.", "success")
        return redirect(url_for("studio.dashboard"))

    # default: save_preset
    name = (form.preset_name.data or "").strip()
    if not name:
        flash("Dai un nome al preset per salvarlo.", "warning")
        return render_template("live.html", form=form)
    preset = Preset(
        user_id=session["user_id"], name=name,
        config=json.dumps(settings), source="manual",
    )
    db.session.add(preset)
    db.session.commit()
    flash(f"Preset «{name}» salvato.", "success")
    return redirect(url_for("studio.presets"))


@studio_bp.route("/live/frame", methods=["POST"])
@login_required
def live_frame():
    """Riceve un frame (data URL JPEG) + config JSON, ritorna il frame elaborato.

    Endpoint di sola elaborazione (nessuna scrittura su DB): chiamato in loop dal
    client. Risponde con i byte JPEG così il browser li mostra direttamente."""
    data = request.get_json(silent=True) or {}
    raw = _decode_data_url(data.get("frame"))
    if not raw:
        return {"error": "Frame mancante o non valido."}, 400

    config = data.get("config") or {}
    try:
        jpeg = render_frame_jpeg(raw, config)
    except ValueError as exc:
        return {"error": str(exc)}, 400
    except Exception:  # YOLO/MediaPipe possono fallire: non esporre lo stacktrace
        current_app.logger.exception("Errore di elaborazione live")
        return {"error": "Elaborazione fallita."}, 500

    return Response(jpeg, mimetype="image/jpeg")
