"""Estensioni Flask istanziate qui e inizializzate nella factory (create_app).

Tenerle separate evita import circolari fra app.py e models.py.
"""
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()
