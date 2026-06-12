"""Autenticazione: registrazione, login, logout, profilo, eliminazione account."""
from flask import (
    Blueprint,
    flash,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

from decorators import login_required
from extensions import db
from forms import DeleteAccountForm, LoginForm, RegisterForm
from models import User

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if session.get("user_id"):
        return redirect(url_for("main.index"))

    form = RegisterForm()
    if form.validate_on_submit():
        if User.query.filter_by(username=form.username.data).first():
            flash("Username già in uso, scegline un altro.", "error")
        elif User.query.filter_by(email=form.email.data).first():
            flash("Esiste già un account con questa email.", "error")
        else:
            user = User(username=form.username.data, email=form.email.data)
            user.set_password(form.password.data)
            db.session.add(user)
            db.session.commit()
            flash("Registrazione completata! Ora puoi accedere.", "success")
            return redirect(url_for("auth.login"))
    return render_template("register.html", form=form)


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if session.get("user_id"):
        return redirect(url_for("main.index"))

    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and user.check_password(form.password.data):
            session.clear()
            session["user_id"] = user.id
            flash(f"Bentornato, {user.username}!", "success")
            next_url = request.args.get("next")
            # Evita open redirect: accetta solo path interni
            if next_url and next_url.startswith("/"):
                return redirect(next_url)
            return redirect(url_for("main.index"))
        flash("Username o password non corretti.", "error")
    return render_template("login.html", form=form)


@auth_bp.route("/logout")
def logout():
    session.clear()
    flash("Sei uscito dal tuo account.", "success")
    return redirect(url_for("main.index"))


@auth_bp.route("/profile")
@login_required
def profile():
    user = db.session.get(User, session["user_id"])
    return render_template(
        "profile.html", user=user, delete_form=DeleteAccountForm()
    )


@auth_bp.route("/profile/delete", methods=["POST"])
@login_required
def delete_account():
    form = DeleteAccountForm()
    if form.validate_on_submit():
        user = db.session.get(User, session["user_id"])
        if user:
            # cascade="all, delete-orphan" cancella anche creazioni e preset
            db.session.delete(user)
            db.session.commit()
        session.clear()
        flash("Account e tutti i dati associati sono stati eliminati.", "success")
        return redirect(url_for("main.index"))
    return redirect(url_for("auth.profile"))
