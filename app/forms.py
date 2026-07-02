"""Form dell'applicazione (Flask-WTF).

Ogni form eredita da FlaskForm e include automaticamente la protezione CSRF
(tramite {{ form.hidden_tag() }} nei template).

I parametri di stile del motore vivono in `_ConfigFieldsForm`, condivisa da Studio
(immagine) e Video: così i preset sono interscambiabili fra i due.
"""
from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed, FileField
from wtforms import (
    BooleanField,
    FloatField,
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

from engine import capabilities
from engine import options as opt


# --- Autenticazione --------------------------------------------------------

class RegisterForm(FlaskForm):
    username = StringField("Username", validators=[DataRequired(), Length(min=3, max=80)])
    email = StringField("Email", validators=[DataRequired(), Email()])
    password = PasswordField("Password", validators=[DataRequired(), Length(min=6, max=128)])
    confirm = PasswordField(
        "Conferma password",
        validators=[DataRequired(), EqualTo("password", message="Le password non coincidono.")],
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


# --- Parametri di stile del motore (condivisi Studio + Video) --------------

class _ConfigFieldsForm(FlaskForm):
    def __init__(self, *args, **kwargs):
        """Adatta le scelte alle capability dell'ambiente (profilo lite vs full).

        Senza ultralytics l'opzione YOLO sparisce dal select: la validazione
        WTForms rifiuta così anche i POST costruiti a mano che la richiedono.
        """
        super().__init__(*args, **kwargs)
        if not capabilities()["yolo"]:
            self.detection_engine.choices = [
                c for c in self.detection_engine.choices if c[0] != "yolo"
            ]

    # Detection
    detection_engine = SelectField("Engine", choices=opt.choices(opt.DETECTION_ENGINES))
    yolo_model_file = SelectField("Modello YOLO", choices=opt.choices(opt.YOLO_MODELS), default="yolov8n.pt")
    use_high_res = BooleanField("Alta risoluzione")
    track_mode = SelectField("Canale", choices=opt.choices(opt.TRACK_MODES))
    threshold = IntegerField("Soglia", validators=[NumberRange(0, 255)], default=127)
    threshold_mode = SelectField("Modo soglia", choices=opt.choices(opt.THRESHOLD_MODES), default="fixed")
    color_target_hex = StringField("Colore target", default="#ff0000")
    color_target_tolerance = IntegerField("Tolleranza", validators=[NumberRange(1, 90)], default=30)
    morph_kernel_size = IntegerField("Morfologia", validators=[NumberRange(1, 9)], default=3)
    edge_low = IntegerField("Edge low", validators=[NumberRange(10, 200)], default=50)
    edge_high = IntegerField("Edge high", validators=[NumberRange(50, 300)], default=150)
    min_blob_size = IntegerField("Dim. minima", validators=[NumberRange(1, 200000)], default=100)
    max_blob_size = IntegerField("Dim. massima", validators=[NumberRange(1, 500000)], default=50000)
    max_blobs = IntegerField("Max blob", validators=[NumberRange(1, 300)], default=20)

    # Pre-processing
    preprocess_enabled = BooleanField("Pre-processing")
    preprocess_method = SelectField("Metodo", choices=opt.choices(opt.PREPROCESS_METHODS))
    preprocess_strength = FloatField("Intensità", validators=[NumberRange(0.1, 2.0)], default=1.0)

    # Blob
    blob_shape = SelectField("Forma", choices=opt.choices(opt.BLOB_SHAPES))
    blob_color = StringField("Colore blob", default="#ffffff")
    blob_thickness = IntegerField("Spessore", validators=[NumberRange(1, 10)], default=2)
    blob_style = SelectField("Stile bordo", choices=opt.choices(opt.BLOB_STYLES))
    corner_radius = IntegerField("Arrotondamento", validators=[NumberRange(0, 80)], default=0)
    blob_dot_gap = IntegerField("Gap punti", validators=[NumberRange(2, 100)], default=10)

    # Wireframe
    wf_type = SelectField("Tipo", choices=opt.choices(opt.WF_TYPES))
    wf_color = StringField("Colore wire", default="#ffffff")
    wf_thickness = IntegerField("Spessore wire", validators=[NumberRange(1, 10)], default=1)
    wf_style = SelectField("Tratteggio", choices=opt.choices(opt.WF_STYLES))
    wf_dot_gap = IntegerField("Gap wire", validators=[NumberRange(2, 100)], default=20)
    wiring_density = IntegerField("Densità", validators=[NumberRange(1, 20)], default=5)
    end_cap = SelectField("Terminali", choices=opt.choices(opt.END_CAPS))

    # Center
    show_center = BooleanField("Mostra centro")
    center_color = StringField("Colore centro", default="#ffff00")
    center_shape = SelectField("Forma centro", choices=opt.choices(opt.CENTER_SHAPES))
    center_style = SelectField("Stile centro", choices=opt.choices(opt.CENTER_STYLES))
    center_size_level = IntegerField("Dim. centro", validators=[NumberRange(1, 5)], default=1)

    # Text / labels
    label_type = SelectField("Etichetta", choices=opt.choices(opt.LABEL_TYPES))
    text_color = StringField("Colore testo", default="#ffffff")
    custom_text = StringField("Testo", default="REC", validators=[Optional(), Length(max=40)])
    font_weight = SelectField("Peso font", choices=opt.choices(opt.FONT_WEIGHTS))
    label_pos = SelectField("Posizione", choices=opt.choices(opt.LABEL_POSITIONS))
    text_size = FloatField("Dim. testo", validators=[NumberRange(0.3, 2.0)], default=0.6)
    text_outline = BooleanField("Contorno testo")
    text_outline_color = StringField("Colore contorno", default="#000000")

    # Inner / scene
    inner_style = SelectField("Filtro interno", choices=opt.choices(opt.INNER_STYLES))
    bg_mode = SelectField("Sfondo", choices=opt.choices(opt.BG_MODES))
    opacity = FloatField("Opacità", validators=[NumberRange(0.1, 1.0)], default=1.0)

    # Glow
    glow_enabled = BooleanField("Glow")
    glow_intensity = FloatField("Intensità glow", validators=[NumberRange(0.0, 2.0)], default=1.0)
    glow_radius = IntegerField("Raggio glow", validators=[NumberRange(5, 51)], default=21)

    # MediaPipe
    mp_pose_enabled = BooleanField("Pose")
    mp_hands_enabled = BooleanField("Mani")
    mp_face_enabled = BooleanField("Volto")
    mp_confidence = FloatField("Confidenza", validators=[NumberRange(0.1, 1.0)], default=0.5)
    mp_num_poses = IntegerField("N. pose", validators=[NumberRange(1, 6)], default=4)
    mp_pose_num_points = IntegerField("Punti pose", validators=[NumberRange(1, 33)], default=6)
    mp_hands_num_points = IntegerField("Punti mani", validators=[NumberRange(1, 21)], default=5)
    mp_num_faces = IntegerField("N. volti", validators=[NumberRange(1, 4)], default=2)
    mp_face_num_points = IntegerField("Punti volto", validators=[NumberRange(1, 15)], default=7)
    mp_blob_size = FloatField("Dim. blob MP", validators=[NumberRange(0.5, 3.0)], default=1.0)
    mp_merge_distance = IntegerField("Fusione (px)", validators=[NumberRange(0, 200)], default=0)


class _MotionFields:
    """Tracking fra frame + scie: condivisi da Studio (video) e Live (mixin WTForms)."""
    smoothing = IntegerField("Smoothing", validators=[NumberRange(0, 60)], default=5)
    persistence = IntegerField("Persistenza", validators=[NumberRange(0, 100)], default=30)
    persistence_fade = BooleanField("Fade persistenza")
    tracker_match_radius = IntegerField("Raggio tracker", validators=[NumberRange(50, 500)], default=150)
    trails_enabled = BooleanField("Scie (trails)")
    trail_length = IntegerField("Lunghezza scia", validators=[NumberRange(5, 60)], default=20)
    trail_opacity = FloatField("Opacità scia", validators=[NumberRange(0.1, 1.0)], default=0.6)
    trail_style = SelectField("Stile scia", choices=opt.choices(opt.TRAIL_STYLES), default="line")


class _AudioModFields:
    """Modulazione audio del rendering (video: traccia caricata; live: microfono)."""
    audio_modulate_size = BooleanField("Modula dimensione")
    audio_modulate_thickness = BooleanField("Modula spessore")
    audio_modulate_glow = BooleanField("Modula glow")
    audio_mod_intensity = FloatField("Intensità modulazione", validators=[NumberRange(0.0, 2.0)], default=1.0)


class StudioForm(_ConfigFieldsForm, _MotionFields, _AudioModFields):
    """Studio unificato: una sorgente che può essere IMMAGINE o VIDEO.

    Con un'immagine si genera/salva una creazione; con un video si avvia
    l'elaborazione completa (tracker, scie, audio) e si scarica l'MP4. I campi
    Motion/Audio hanno senso solo sul video ma restano nel form (default validi):
    la UI li mostra solo quando è caricato un video.
    """
    source = FileField(
        "Sorgente",
        validators=[
            Optional(),
            FileAllowed(
                ["jpg", "jpeg", "png", "webp", "bmp", "mp4", "mov", "avi", "webm", "mkv"],
                "Solo immagini (jpg, png, webp, bmp) o video (mp4, mov, avi, webm, mkv).",
            ),
        ],
    )
    # Parametro temporale che ha senso solo sul file video
    frame_skip = IntegerField("Frame skip", validators=[NumberRange(1, 10)], default=1)
    # Reattività audio (solo video): si attiva caricando una traccia; audio_enabled
    # e audio_path li imposta il service in base al file.
    audio = FileField(
        "Traccia audio",
        validators=[
            Optional(),
            FileAllowed(["mp3", "wav", "m4a", "aac", "ogg", "flac"], "Solo audio (mp3, wav, m4a, aac, ogg, flac)."),
        ],
    )
    audio_band = SelectField("Banda", choices=opt.choices(opt.AUDIO_BANDS), default="bass")
    audio_sensitivity = FloatField("Sensibilità beat", validators=[NumberRange(0.1, 3.0)], default=1.0)
    audio_offset = FloatField("Offset (s)", validators=[NumberRange(0.0, 60.0)], default=0.0)
    # Frame grezzo (data URL) catturato dal client per salvare un fotogramma
    # dell'anteprima (immagine o frame di un video) come creazione.
    snapshot_raw = HiddenField()
    preset_name = StringField("Nome", validators=[Optional(), Length(max=80)])
    submit = SubmitField("Elabora")


class LiveForm(_ConfigFieldsForm, _MotionFields, _AudioModFields):
    """Live cam: parametri di stile + motion (i frame arrivano dalla webcam).

    Lo stato di tracking è per-sessione di stream (frame_engine): scie e ID
    stabili si accumulano fra i frame. La modulazione audio usa il livello del
    microfono calcolato nel browser. `preset_name` per salvare preset/snapshot.
    """
    preset_name = StringField("Nome", validators=[Optional(), Length(max=80)])
    submit = SubmitField("Salva")


# --- AI Preset Generator ---------------------------------------------------

class AIPresetForm(FlaskForm):
    prompt = TextAreaField(
        "Descrivi il look che vuoi", validators=[DataRequired(), Length(min=3, max=400)]
    )
    submit = SubmitField("Genera preset")


class SaveAIPresetForm(FlaskForm):
    """Salva un preset generato dall'AI (il config viaggia in un campo nascosto)."""
    name = StringField("Nome preset", validators=[DataRequired(), Length(max=80)])
    config = HiddenField()
    submit = SubmitField("Salva preset")
