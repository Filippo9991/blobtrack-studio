"""Studio CV: upload immagine, blob detection, galleria creazioni e preset."""
import base64
import json

from flask import (
    Blueprint,
    Response,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from decorators import login_required
from extensions import db
from forms import DeleteForm, ProcessForm
from models import Creation, Preset
from services.image_processing import normalize_settings, process_image

studio_bp = Blueprint("studio", __name__)

# Campi della ProcessForm che descrivono un preset di stile (senza l'immagine)
SETTING_FIELDS = [
    "track_mode", "threshold", "min_size", "max_blobs", "blob_shape",
    "blob_style", "blob_color", "blob_thickness", "corner_radius", "wf_type",
    "wf_style", "wf_color", "wiring_density", "inner_style", "bg_mode",
    "label_type", "show_center",
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
    form = ProcessForm()
    preview = None

    if request.method == "GET":
        # Precarica i parametri di un preset, se richiesto (?preset=<id>)
        preset_id = request.args.get("preset", type=int)
        if preset_id:
            preset = Preset.query.filter_by(
                id=preset_id, user_id=session["user_id"]
            ).first()
            if preset:
                _apply_settings_to_form(form, normalize_settings(json.loads(preset.config)))
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
        else:
            preset = Preset(
                user_id=session["user_id"],
                name=name,
                config=json.dumps(normalize_settings(settings)),
                source="manual",
            )
            db.session.add(preset)
            db.session.commit()
            flash(f"Preset «{name}» salvato.", "success")
            return redirect(url_for("studio.presets"))
        return render_template("studio.html", form=form, preview=None)

    # preview / save: serve un'immagine
    file = form.image.data
    if not file or not getattr(file, "filename", ""):
        flash("Carica un'immagine per elaborarla.", "warning")
        return render_template("studio.html", form=form, preview=None)

    try:
        png_bytes, n_blobs = process_image(file.read(), settings)
    except ValueError as exc:
        flash(str(exc), "error")
        return render_template("studio.html", form=form, preview=None)

    if action == "save":
        title = (form.preset_name.data or file.filename or "Creazione").strip()[:120]
        creation = Creation(
            user_id=session["user_id"],
            title=title,
            image_data=png_bytes,
            settings=json.dumps(normalize_settings(settings)),
        )
        db.session.add(creation)
        db.session.commit()
        flash(f"Creazione salvata ({n_blobs} blob rilevati).", "success")
        return redirect(url_for("studio.dashboard"))

    # preview
    preview = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
    flash(f"Anteprima generata: {n_blobs} blob rilevati.", "success")
    return render_template("studio.html", form=form, preview=preview)


@studio_bp.route("/dashboard")
@login_required
def dashboard():
    creations = (
        Creation.query.filter_by(user_id=session["user_id"])
        .order_by(Creation.created_at.desc())
        .all()
    )
    return render_template(
        "dashboard.html", creations=creations, delete_form=DeleteForm()
    )


@studio_bp.route("/creation/<int:creation_id>/image")
@login_required
def creation_image(creation_id):
    creation = Creation.query.filter_by(
        id=creation_id, user_id=session["user_id"]
    ).first_or_404()
    return Response(creation.image_data, mimetype="image/png")


@studio_bp.route("/creation/<int:creation_id>/delete", methods=["POST"])
@login_required
def delete_creation(creation_id):
    form = DeleteForm()
    if form.validate_on_submit():
        creation = Creation.query.filter_by(
            id=creation_id, user_id=session["user_id"]
        ).first_or_404()
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
    form = DeleteForm()
    if form.validate_on_submit():
        preset = Preset.query.filter_by(
            id=preset_id, user_id=session["user_id"]
        ).first_or_404()
        db.session.delete(preset)
        db.session.commit()
        flash("Preset eliminato.", "success")
    return redirect(url_for("studio.presets"))
