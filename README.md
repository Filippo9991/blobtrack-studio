# 💠 BlobTrack Studio

Web app per **blob detection artistica**: l'utente carica un'immagine, l'app rileva
forme e regioni con **OpenCV** lato server e le ristila con contorni, wireframe e
filtri creativi. Le creazioni si salvano in una galleria personale e i preset di
stile possono essere generati da un **assistente AI**.

> 🌐 **Live demo:** _(aggiungi qui l'URL Render dopo il deploy, es. https://blobtrack.onrender.com)_

Progetto d'esame: applicazione web completa con **Python + Flask**.

---

## ✨ Funzionalità

- 🔐 **Autenticazione completa** — registrazione, login, logout, eliminazione account (password hashate con `werkzeug.security`)
- 🍪 **Banner GDPR** — consenso al trattamento dati salvato sul database
- 🖼️ **Studio CV** — upload immagine → blob detection + styling con OpenCV, anteprima in tempo reale
- 🗂️ **Galleria personale** — le creazioni elaborate salvate come record sul database
- 🎚️ **Preset di stile** — salva e riapplica le tue configurazioni preferite
- 🤖 **AI Preset Generator** — descrivi un look a parole e l'AI (Groq) costruisce il preset
- ⚠️ **Pagine d'errore custom** — 404, 500, 413
- 📱 **Design responsive** — mobile-first, design system con CSS custom properties

---

## 🛠️ Tech Stack

| Ambito | Tecnologie |
|--------|-----------|
| Backend | Python 3.11, Flask, Flask-SQLAlchemy, Flask-WTF |
| Database | SQLite (sviluppo) · PostgreSQL (produzione) |
| Computer vision | OpenCV (`opencv-python-headless`), NumPy |
| Template | Jinja2 (con template inheritance) |
| API esterna | Groq (LLM gratuito, endpoint OpenAI-compatibile) |
| Deploy | Gunicorn, Render |
| Test | pytest |

---

## 🚀 Installazione locale

```bash
# 1. Clona il repository
git clone <url-del-repo>
cd blobTrack

# 2. Crea e attiva il virtual environment (Python 3.11)
python3.11 -m venv venv
source venv/bin/activate        # su Windows: venv\Scripts\activate

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Configura le variabili d'ambiente
cp .env.example .env
#   genera una SECRET_KEY:  python -c "import secrets; print(secrets.token_hex(32))"
#   aggiungi la tua GROQ_API_KEY (gratuita su https://console.groq.com)

# 5. Avvia l'app
flask --app app run --debug
```

L'app è su http://127.0.0.1:5000. Il database SQLite viene creato automaticamente in `instance/`.

### Eseguire i test

```bash
pytest
```

---

## 🔑 Variabili d'ambiente

| Variabile | Descrizione |
|-----------|-------------|
| `SECRET_KEY` | Chiave segreta Flask (sessioni, CSRF) |
| `GROQ_API_KEY` | Chiave API Groq per l'AI Preset Generator |
| `GROQ_MODEL` | Modello LLM (default `llama-3.3-70b-versatile`) |
| `FLASK_ENV` | `development` · `production` · `testing` |
| `DATABASE_URL` | Solo in produzione: connection string PostgreSQL (fornita da Render) |

Nessun valore sensibile è committato: `.env` è nel `.gitignore`, ed è presente `.env.example`.

---

## ☁️ Deploy su Render

1. Crea un database **PostgreSQL** (Free) e copia la *Internal Database URL*.
2. Crea un **Web Service** collegato a questo repository GitHub:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`
3. Imposta le **Environment Variables**: `SECRET_KEY`, `DATABASE_URL`, `GROQ_API_KEY`, `FLASK_ENV=production`.
4. Deploy. Le tabelle vengono create automaticamente all'avvio (`db.create_all()` idempotente).

---

## 📁 Struttura del progetto

```
app.py                 Application factory + error handlers
config.py              Configurazione multi-ambiente
extensions.py          Istanza SQLAlchemy
models.py              User, Creation, Preset
forms.py               Form Flask-WTF
decorators.py          @login_required
blob_engine_core.py    Primitive OpenCV (blob detection + styling)
services/              image_processing, ai_presets
blueprints/            main, auth, studio, assistant
templates/             Template Jinja2 (base + pagine)
static/                CSS (design system) e JS
tests/                 Suite pytest
```
