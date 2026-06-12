"""Modelli del database (SQLAlchemy).

Relazioni:
    User 1───* Creation   (le immagini elaborate salvate dall'utente)
    User 1───* Preset      (le configurazioni di stile, manuali o generate dall'AI)

L'eliminazione di un User cancella in cascata tutti i suoi dati associati
(requisito: "Eliminazione account").
"""
from datetime import datetime, timezone

from werkzeug.security import check_password_hash, generate_password_hash

from extensions import db


def _utcnow():
    return datetime.now(timezone.utc)


class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    cookie_consent = db.Column(db.Boolean, default=False, nullable=False)
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    creations = db.relationship(
        "Creation", backref="user", cascade="all, delete-orphan", lazy=True
    )
    presets = db.relationship(
        "Preset", backref="user", cascade="all, delete-orphan", lazy=True
    )

    def set_password(self, password):
        """Salva la password come hash (mai in chiaro)."""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def __repr__(self):
        return f"<User {self.username}>"


class Creation(db.Model):
    """Un'immagine elaborata salvata dall'utente.

    L'immagine è salvata come BLOB nel database così da persistere anche su
    filesystem ephemeral (Render) senza bisogno di object storage esterno.
    """

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    title = db.Column(db.String(120), nullable=False)
    image_data = db.Column(db.LargeBinary, nullable=False)
    settings = db.Column(db.Text, nullable=False, default="{}")
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    def __repr__(self):
        return f"<Creation {self.id} '{self.title}'>"


class Preset(db.Model):
    """Una configurazione di stile riutilizzabile (parametri della blob detection)."""

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False)
    name = db.Column(db.String(80), nullable=False)
    config = db.Column(db.Text, nullable=False, default="{}")
    source = db.Column(db.String(20), default="manual", nullable=False)  # 'manual' | 'ai'
    created_at = db.Column(db.DateTime, default=_utcnow, nullable=False)

    def __repr__(self):
        return f"<Preset {self.id} '{self.name}' ({self.source})>"
