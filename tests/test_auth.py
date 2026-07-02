"""Test di autenticazione e protezione delle route."""
from conftest import register_and_login

from app.models import User


def test_home_loads(client):
    assert client.get("/").status_code == 200


def test_register_hashes_password(client, app):
    client.post(
        "/register",
        data=dict(username="mario", email="mario@test.com",
                  password="secret1", confirm="secret1"),
    )
    with app.app_context():
        user = User.query.filter_by(username="mario").first()
        assert user is not None
        assert user.password_hash != "secret1"
        assert user.check_password("secret1")


def test_duplicate_username_rejected(client):
    # Primo utente (senza login, così il secondo register non viene reindirizzato)
    client.post(
        "/register",
        data=dict(username="anna", email="anna@test.com",
                  password="secret1", confirm="secret1"),
    )
    r = client.post(
        "/register",
        data=dict(username="anna", email="other@test.com",
                  password="secret1", confirm="secret1"),
    )
    assert "già in uso".encode() in r.data


def test_password_mismatch_rejected(client):
    r = client.post(
        "/register",
        data=dict(username="lucia", email="lucia@test.com",
                  password="secret1", confirm="nope"),
    )
    assert "non coincidono".encode() in r.data


def test_login_required_redirects(client):
    r = client.get("/dashboard")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_login_logout_cycle(client):
    register_and_login(client, "bob")
    assert client.get("/dashboard").status_code == 200
    client.get("/logout")
    assert client.get("/dashboard").status_code == 302


def test_delete_account_removes_data(client, app):
    register_and_login(client, "todelete")
    client.post("/profile/delete")
    with app.app_context():
        assert User.query.filter_by(username="todelete").first() is None


def test_custom_404(client):
    r = client.get("/nope")
    assert r.status_code == 404
    assert b"404" in r.data


def test_consent_accepted_before_login_persists_to_db(client, app):
    """Regressione: il consenso dato da anonimi deve sopravvivere a session.clear()
    del login e finire sul DB (prima il banner riappariva dopo il login)."""
    client.post(
        "/register",
        data=dict(username="gdpr", email="gdpr@test.com", password="secret1", confirm="secret1"),
    )
    client.post("/consent", data={"action": "accept"})          # da anonimo
    client.post("/login", data=dict(username="gdpr", password="secret1"))

    assert b"cookie-banner" not in client.get("/").data          # niente banner
    with app.app_context():
        assert User.query.filter_by(username="gdpr").first().cookie_consent is True


def test_consent_rejected_hides_banner_but_not_saved_as_accept(client, app):
    """Il rifiuto nasconde il banner in sessione ma NON marca il consenso sul DB."""
    client.post(
        "/register",
        data=dict(username="nogdpr", email="nogdpr@test.com", password="secret1", confirm="secret1"),
    )
    client.post("/consent", data={"action": "reject"})
    client.post("/login", data=dict(username="nogdpr", password="secret1"))

    assert b"cookie-banner" not in client.get("/").data          # scelta rispettata
    with app.app_context():
        assert User.query.filter_by(username="nogdpr").first().cookie_consent is False
