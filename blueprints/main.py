"""Route pubbliche e generali: home, consenso cookie."""
from flask import (
    Blueprint,
    jsonify,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from extensions import db
from models import User

main_bp = Blueprint("main", __name__)


@main_bp.route("/")
def index():
    return render_template("index.html")


@main_bp.route("/consent", methods=["POST"])
def consent():
    """Registra la scelta dell'utente sul banner cookie (GDPR).

    - In sessione segniamo che la scelta è stata fatta, così il banner non riappare.
    - Se l'utente è loggato e accetta, salviamo il consenso sul DB (User.cookie_consent),
      così la scelta persiste anche nelle sessioni future.
    """
    accepted = request.form.get("action") == "accept"
    session["cookie_consent"] = True

    if accepted and session.get("user_id"):
        user = db.session.get(User, session["user_id"])
        if user:
            user.cookie_consent = True
            db.session.commit()

    # Richiesta via fetch (JS): rispondiamo in JSON e il banner sparisce senza reload
    if request.headers.get("X-Requested-With") == "fetch":
        return jsonify(ok=True, accepted=accepted)

    # Fallback senza JS: ricarica la pagina di provenienza
    return redirect(request.referrer or url_for("main.index"))
