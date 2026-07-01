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

    # Provider LLM gratuito (Groq) per l'AI Preset Generator
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
    GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    # Limite dimensione upload: 64 MB (immagini piccole, video brevi)
    MAX_CONTENT_LENGTH = 64 * 1024 * 1024

    # Limiti dell'elaborazione video (0 = nessun limite, profilo locale).
    # In produzione (free tier: 0.1 CPU) vengono attivati per mantenere
    # i tempi di risposta accettabili.
    VIDEO_MAX_SECONDS = int(os.environ.get("VIDEO_MAX_SECONDS", "0"))
    VIDEO_MAX_DIM = int(os.environ.get("VIDEO_MAX_DIM", "0"))


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False

    # Server piccolo (Render free): upload più contenuti e video limitati
    # (durata/risoluzione ridotte prima dell'elaborazione). Override via env.
    MAX_CONTENT_LENGTH = 32 * 1024 * 1024
    VIDEO_MAX_SECONDS = int(os.environ.get("VIDEO_MAX_SECONDS", "15"))
    VIDEO_MAX_DIM = int(os.environ.get("VIDEO_MAX_DIM", "480"))

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
