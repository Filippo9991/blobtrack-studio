"""BlobTrack Studio — application factory (package `app`).

Avvio locale:   flask --app app run --debug
Avvio produzione (Render):  gunicorn wsgi:app
"""
import logging
import os

from flask import Flask, render_template, session

from config import config_by_name
from app.extensions import db

logging.basicConfig(level=logging.INFO)


def create_app(config_name=None):
    config_name = config_name or os.environ.get("FLASK_ENV", "development")
    if config_name not in config_by_name:
        config_name = "development"

    app = Flask(__name__, instance_relative_config=True)
    app.config.from_object(config_by_name[config_name])
    os.makedirs(app.instance_path, exist_ok=True)

    db.init_app(app)

    # --- Blueprints ---
    from app.blueprints.main import main_bp
    from app.blueprints.auth import auth_bp
    from app.blueprints.studio import studio_bp
    from app.blueprints.assistant import assistant_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(studio_bp)
    app.register_blueprint(assistant_bp)

    # --- variabili globali disponibili in tutti i template ---
    @app.context_processor
    def inject_globals():
        from engine import capabilities

        from app.models import User

        uid = session.get("user_id")
        user = db.session.get(User, uid) if uid else None

        # Il consenso può venire dalla sessione (utenti anonimi) o dal DB (utenti loggati)
        consent_given = bool(session.get("cookie_consent"))
        if user and user.cookie_consent:
            consent_given = True

        return {
            "current_user": user,
            "show_cookie_banner": not consent_given,
            # Profilo lite vs full: i template nascondono ciò che non è installato
            "caps": capabilities(),
        }

    # --- Pagine di errore custom ---
    @app.errorhandler(404)
    def not_found(error):
        return render_template("404.html"), 404

    @app.errorhandler(500)
    def server_error(error):
        db.session.rollback()  # evita di lasciare la sessione DB in stato sporco
        return render_template("500.html"), 500

    @app.errorhandler(413)
    def too_large(error):
        return render_template("413.html"), 413

    # Crea le tabelle se mancano (idempotente: sicuro ad ogni avvio, anche su Render)
    with app.app_context():
        db.create_all()

    return app
