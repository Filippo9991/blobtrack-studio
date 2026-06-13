"""AI Preset Generator: genera un preset di stile da una descrizione testuale."""
import json

from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    session,
    url_for,
)

from decorators import login_required
from extensions import db
from forms import AIPresetForm, SaveAIPresetForm
from models import Preset
from services.ai_presets import AIPresetError, generate_preset

assistant_bp = Blueprint("assistant", __name__)


@assistant_bp.route("/assistant", methods=["GET", "POST"])
@login_required
def assistant():
    form = AIPresetForm()
    save_form = SaveAIPresetForm()
    generated = None

    if form.validate_on_submit():
        try:
            generated = generate_preset(form.prompt.data)
        except AIPresetError as exc:
            flash(str(exc), "error")
        else:
            if generated is None:
                flash(
                    "L'AI non ha risposto (limite di richieste o errore di rete). Riprova tra poco.",
                    "warning",
                )
            else:
                save_form.config.data = json.dumps(generated)
                flash("Preset generato! Dagli un nome e salvalo.", "success")

    return render_template(
        "assistant.html", form=form, save_form=save_form, generated=generated
    )


@assistant_bp.route("/assistant/save", methods=["POST"])
@login_required
def save_generated():
    save_form = SaveAIPresetForm()
    if save_form.validate_on_submit():
        try:
            settings = json.loads(save_form.config.data or "{}")
            if not isinstance(settings, dict):
                raise ValueError
        except ValueError:
            flash("Configurazione generata non valida.", "error")
            return redirect(url_for("assistant.assistant"))

        preset = Preset(
            user_id=session["user_id"],
            name=save_form.name.data.strip(),
            config=json.dumps(settings),
            source="ai",
        )
        db.session.add(preset)
        db.session.commit()
        flash(f"Preset «{preset.name}» salvato.", "success")
        return redirect(url_for("studio.presets"))

    flash("Dai un nome al preset per salvarlo.", "warning")
    return redirect(url_for("assistant.assistant"))
