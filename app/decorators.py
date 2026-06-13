"""Decorator di autorizzazione."""
from functools import wraps

from flask import flash, redirect, request, session, url_for


def login_required(view):
    """Protegge una route: se l'utente non è loggato lo rimanda al login.

    Implementato con functools.wraps per preservare nome e docstring della view.
    """

    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if not session.get("user_id"):
            flash("Devi effettuare l'accesso per continuare.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)

    return wrapped_view
