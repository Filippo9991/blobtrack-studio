"""Configurazione multi-ambiente dell'app BlobTrack Studio.

Nessun valore sensibile è hardcoded: SECRET_KEY e le API key vengono lette
dalle variabili d'ambiente (file .env in locale, pannello Environment su Render).
"""
import os

from dotenv import load_dotenv

# Carica le variabili dal file .env in locale (in produzione le inietta la piattaforma)
load_dotenv()


class Config:
    """Configurazione di base, condivisa da tutti gli ambienti."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-only-secret-change-me")

    # SQLite di default: il file vive nella cartella instance/ (path relativo)
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URI", "sqlite:///blobtrack.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    # Riconnette automaticamente se il database ha chiuso la connessione (utile su Postgres)
    SQLALCHEMY_ENGINE_OPTIONS = {"pool_pre_ping": True}

    # Provider LLM gratuito per l'AI Preset Generator
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

    # Limite dimensione upload: 8 MB
    MAX_CONTENT_LENGTH = 8 * 1024 * 1024


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    # Render/Heroku usano DATABASE_URL e il formato postgres:// che SQLAlchemy
    # non accetta più: va convertito in postgresql://.
    _db_url = os.environ.get("DATABASE_URL", "")
    if _db_url.startswith("postgres://"):
        _db_url = _db_url.replace("postgres://", "postgresql://", 1)
    SQLALCHEMY_DATABASE_URI = _db_url or Config.SQLALCHEMY_DATABASE_URI


class TestingConfig(Config):
    TESTING = True
    WTF_CSRF_ENABLED = False
    SQLALCHEMY_DATABASE_URI = "sqlite:///:memory:"


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig,
}
