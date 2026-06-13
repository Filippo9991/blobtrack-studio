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

from decorators import login_required
from extensions import db
from forms import DeleteForm, StudioForm
from models import Creation, Preset
from services.frame_engine import render_image

studio_bp = Blueprint("studio", __name__)

# Campi della StudioForm che compongono il config del motore (tutto tranne image/preset_name/submit)
SETTING_FIELDS = [
    "detection_engine", "yolo_model_file", "use_high_res", "track_mode", "threshold",
    "threshold_mode", "color_target_hex", "color_target_tolerance", "morph_kernel_size",
    "edge_low", "edge_high", "min_blob_size", "max_blob_size", "max_blobs",
    "preprocess_enabled", "preprocess_method", "preprocess_strength",
    "blob_shape", "blob_color", "blob_thickness", "blob_style", "corner_radius", "blob_dot_gap",
    "wf_type", "wf_color", "wf_thickness", "wf_style", "wf_dot_gap", "wiring_density", "end_cap",
    "show_center", "center_color", "center_shape", "center_style", "center_size_level",
    "label_type", "text_color", "custom_text", "font_weight", "label_pos", "text_size",
    "text_outline", "text_outline_color",
    "inner_style", "bg_mode", "opacity",
    "glow_enabled", "glow_intensity", "glow_radius",
    "mp_pose_enabled", "mp_hands_enabled", "mp_face_enabled", "mp_confidence",
    "mp_num_poses", "mp_pose_num_points", "mp_hands_num_points", "mp_num_faces",
    "mp_face_num_points", "mp_blob_size", "mp_merge_distance",
]


def _form_to_settings(form):
    return {field: getattr(form, field).data for field in SETTING_FIELDS}


def _apply_settings_to_form(form, settings):
    for field in SETTING_FIELDS:
        if field in settings and settings[field] is not None:
            getattr(form, field).data = settings[field]


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
        png_bytes, _ = render_image(file.read(), settings)
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
