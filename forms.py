"""Form dell'applicazione (Flask-WTF).

Ogni form eredita da FlaskForm e include automaticamente la protezione CSRF
(tramite {{ form.hidden_tag() }} nei template).
"""
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from wtforms import (
    BooleanField,
    HiddenField,
    IntegerField,
    PasswordField,
    SelectField,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import (
    DataRequired,
    Email,
    EqualTo,
    Length,
    NumberRange,
    Optional,
)

from services.image_processing import (
    BG_MODES,
    BLOB_SHAPES,
    BLOB_STYLES,
    INNER_STYLES,
    LABEL_TYPES,
    TRACK_MODES,
    WF_STYLES,
    WF_TYPES,
)


def _choices(values):
    """Trasforma una lista di stringhe in coppie (value, label) per i SelectField."""
    return [(v, v.replace("_", " ").capitalize()) for v in values]


# --- Autenticazione --------------------------------------------------------

class RegisterForm(FlaskForm):
    username = StringField(
        "Username", validators=[DataRequired(), Length(min=3, max=80)]
    )
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField(
        "Password", validators=[DataRequired(), Length(min=6, max=128)]
    )
    confirm = PasswordField(
        "Conferma password",
        validators=[
            DataRequired(),
            EqualTo("password", message="Le password non coincidono."),
        ],
    )
    submit = SubmitField("Registrati")


class LoginForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired()])
    password = PasswordField("Password", validators=[DataRequired()])
    submit = SubmitField("Accedi")


class DeleteAccountForm(FlaskForm):
    submit = SubmitField("Elimina definitivamente il mio account")


class DeleteForm(FlaskForm):
    """Form minimale (solo CSRF) per le azioni di eliminazione."""
    submit = SubmitField("Elimina")


# --- Studio CV -------------------------------------------------------------

class ProcessForm(FlaskForm):
    image = FileField(
        "Immagine",
        validators=[
            Optional(),
            FileAllowed(
                ["jpg", "jpeg", "png", "webp", "bmp"],
                "Sono ammesse solo immagini (jpg, png, webp, bmp).",
            ),
        ],
    )
    track_mode = SelectField("Canale di analisi", choices=_choices(TRACK_MODES))
    threshold = IntegerField("Soglia", validators=[NumberRange(0, 255)], default=127)
    min_size = IntegerField("Dimensione minima", validators=[NumberRange(1, 100000)], default=150)
    max_blobs = IntegerField("Numero massimo di blob", validators=[NumberRange(1, 200)], default=40)

    blob_shape = SelectField("Forma", choices=_choices(BLOB_SHAPES))
    blob_style = SelectField("Stile bordo", choices=_choices(BLOB_STYLES))
    blob_color = StringField("Colore blob", default="#00ff9d")
    blob_thickness = IntegerField("Spessore", validators=[NumberRange(1, 8)], default=2)
    corner_radius = IntegerField("Arrotondamento", validators=[NumberRange(0, 60)], default=0)

    wf_type = SelectField("Wireframe", choices=_choices(WF_TYPES))
    wf_style = SelectField("Tratteggio wireframe", choices=_choices(WF_STYLES))
    wf_color = StringField("Colore wireframe", default="#00ff9d")
    wiring_density = IntegerField("Densità collegamenti", validators=[NumberRange(1, 20)], default=3)

    inner_style = SelectField("Filtro interno", choices=_choices(INNER_STYLES))
    bg_mode = SelectField("Sfondo", choices=_choices(BG_MODES))
    label_type = SelectField("Etichetta", choices=_choices(LABEL_TYPES))
    show_center = BooleanField("Mostra centro")

    preset_name = StringField("Nome preset", validators=[Optional(), Length(max=80)])

    # Più azioni dallo stesso form (lette via request.form['action'])
    submit = SubmitField("Elabora")


class AIPresetForm(FlaskForm):
    prompt = TextAreaField(
        "Descrivi il look che vuoi",
        validators=[DataRequired(), Length(min=3, max=400)],
    )
    submit = SubmitField("Genera preset")


class SaveAIPresetForm(FlaskForm):
    """Salva un preset generato dall'AI (il config viaggia in un campo nascosto)."""
    name = StringField("Nome preset", validators=[DataRequired(), Length(max=80)])
    config = HiddenField()
    submit = SubmitField("Salva preset")
