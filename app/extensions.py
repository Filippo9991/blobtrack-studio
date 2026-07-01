"""Estensioni Flask istanziate qui e inizializzate nella factory (create_app).

Tenerle separate evita import circolari fra app.py e models.py.
"""
from flask_sock import Sock
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
sock = Sock()  # WebSocket per il Live (le route sono in app/blueprints/live_ws.py)
