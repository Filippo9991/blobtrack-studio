"""Fixture condivise per i test (pytest).

Forziamo l'ambiente 'testing' (SQLite in memoria, CSRF disabilitato) prima di
importare l'app, così il modulo crea direttamente un'app di test.
"""
import os

os.environ["FLASK_ENV"] = "testing"

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from app import create_app  # noqa: E402


@pytest.fixture
def app():
    return create_app("testing")


@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def sample_png():
    """Tre cerchi bianchi su sfondo nero, codificati in PNG."""
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    for x in (100, 200, 300):
        cv2.circle(img, (x, 150), 40, (255, 255, 255), -1)
    return cv2.imencode(".png", img)[1].tobytes()


def register_and_login(client, username="tester"):
    client.post(
        "/register",
        data=dict(
            username=username,
            email=f"{username}@test.com",
            password="secret1",
            confirm="secret1",
        ),
    )
    client.post("/login", data=dict(username=username, password="secret1"))
