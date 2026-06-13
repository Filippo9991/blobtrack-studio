"""Entrypoint WSGI per i server di produzione.

    gunicorn wsgi:app

L'app non viene creata all'import del package `app` (niente side-effect): la
factory `create_app()` viene invocata esplicitamente qui.
"""
from app import create_app

app = create_app()
