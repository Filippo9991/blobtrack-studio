"""BlobTrack Studio — application factory.

Avvio locale:   flask --app app run --debug
Avvio produzione (Render):  gunicorn app:app
"""
import logging
import os

from flask import Flask, render_template, session

from config import config_by_name
from extensions import db

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
    from blueprints.main import main_bp
    from blueprints.auth import auth_bp
    from blueprints.studio import studio_bp
    from blueprints.assistant import assistant_bp

    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(studio_bp)
    app.register_blueprint(assistant_bp)

    # --- current_user disponibile in tutti i template ---
    @app.context_processor
    def inject_current_user():
        from models import User

        uid = session.get("user_id")
        user = db.session.get(User, uid) if uid else None
        return {"current_user": user}

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


app = create_app()
