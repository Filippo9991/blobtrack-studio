"""AI Preset Generator — integrazione con un LLM gratuito (Groq).

Riceve una descrizione in linguaggio naturale e chiede al modello un preset di
stile in formato JSON. La chiamata è server-side, con timeout, gestione errori e
chiave letta dalle variabili d'ambiente (mai hardcoded).
"""
import json

import requests
from flask import current_app

import studio_options as opt

# Endpoint OpenAI-compatibile di Groq
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"


class AIPresetError(Exception):
    """Errore di configurazione (es. chiave mancante), distinto dai fallimenti di rete."""


def _system_prompt():
    return (
        "Sei un assistente che configura un'app di blob detection artistica su immagini. "
        "Dato un look descritto a parole, restituisci SOLO un oggetto JSON con un sottoinsieme "
        "di questi campi, usando esattamente questi nomi e rispettando i valori ammessi:\n"
        f"- blob_shape: uno tra {opt.BLOB_SHAPES}\n"
        f"- blob_style: uno tra {opt.BLOB_STYLES}\n"
        f"- wf_type: uno tra {opt.WF_TYPES}\n"
        f"- wf_style: uno tra {opt.WF_STYLES}\n"
        f"- inner_style: uno tra {opt.INNER_STYLES}\n"
        f"- bg_mode: uno tra {opt.BG_MODES}\n"
        f"- track_mode: uno tra {opt.TRACK_MODES}\n"
        "- threshold: intero 0-255\n"
        "- min_blob_size: intero 50-5000\n"
        "- max_blobs: intero 1-100\n"
        "- blob_color, wf_color, center_color: colori esadecimali, es #00ff9d\n"
        "- blob_thickness: intero 1-10\n"
        "- wiring_density: intero 1-20\n"
        "- corner_radius: intero 0-80\n"
        "- glow_enabled: true o false\n"
        "- show_center: true o false\n"
        "Rispondi solo con il JSON, senza testo aggiuntivo."
    )


def _validate(raw):
    """Tiene solo i campi noti al motore e ne coercizza/valida i valori."""
    from schemas import ProcessingConfig

    allowed = set(ProcessingConfig.model_fields)
    clean = {k: v for k, v in raw.items() if k in allowed}
    if not clean:
        return None
    validated = ProcessingConfig(**clean).model_dump()
    return {k: validated[k] for k in clean}


def generate_preset(prompt):
    """Chiede a Groq un preset a partire da una descrizione.

    Ritorna un dict di settings normalizzato, oppure None se la chiamata fallisce
    (timeout, errore HTTP, errore di connessione o risposta non valida).
    Solleva AIPresetError se la chiave API non è configurata.
    """
    api_key = current_app.config.get("GROQ_API_KEY")
    if not api_key:
        raise AIPresetError("L'assistente AI non è configurato: manca la variabile GROQ_API_KEY.")

    payload = {
        "model": current_app.config.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        "messages": [
            {"role": "system", "content": _system_prompt()},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.8,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    try:
        response = requests.post(GROQ_URL, json=payload, headers=headers, timeout=5)
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        raw_settings = json.loads(content)
    except requests.exceptions.Timeout:
        current_app.logger.warning("Groq: timeout della richiesta")
        return None
    except requests.exceptions.HTTPError as exc:
        current_app.logger.error("Groq: errore HTTP %s", exc)
        return None
    except requests.exceptions.ConnectionError as exc:
        current_app.logger.error("Groq: errore di connessione %s", exc)
        return None
    except (KeyError, ValueError) as exc:
        current_app.logger.error("Groq: risposta non valida %s", exc)
        return None

    return _validate(raw_settings)
