from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import whisper
import torch
import io
import numpy as np
from pydub import AudioSegment
from utils.summarizer import summarize_text
from utils.classifier import classify_text
import logging
import tempfile
import os
import subprocess
import json

# Configuration Flask
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
socketio = SocketIO(app, cors_allowed_origins="*", logger=True)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Variables globales
device = "cuda" if torch.cuda.is_available() else "cpu"
model = None
accumulated_text = ""  # Texte accumulé pour un meilleur résumé

def load_whisper_model():
    """Charge le modèle Whisper de manière sécurisée"""
    global model
    try:
        logger.info(f"Chargement de Whisper sur {device}...")
        model = whisper.load_model("small", device=device)
        logger.info("Whisper chargé avec succès !")
        return True
    except Exception as e:
        logger.error(f"Erreur chargement Whisper: {e}")
        return False

@app.route("/")
def index():
    """Page d'accueil"""
    return render_template("index.html")

@socketio.on("connect")
def handle_connect():
    """Client connecté"""
    logger.info("🔗 Client connecté")
    emit("status", {"message": "Connecté au serveur Whisper", "type": "success"})

@socketio.on("disconnect") 
def handle_disconnect():
    """Client déconnecté"""
    logger.info("❌ Client déconnecté")

@socketio.on("audio_chunk")
def handle_audio_chunk(data, mime_type=None):
    """
    Réception et traitement d'un chunk audio depuis le navigateur
    
    Args:
        data: Données audio en bytes (format WebM/Opus)
    """
    global accumulated_text
    temp_in_path = None
    temp_wav_path = None
    try:
        if model is None:
            emit("error", {"message": "Modèle Whisper non chargé"})
            return

        # Extraire audio et mimeType
        default_mime = "audio/webm;codecs=opus"
        if isinstance(data, dict):
            mime_type = data.get("mimeType", mime_type or default_mime)
            data = data.get("audio", b"")
        mime_type = mime_type or default_mime

        # Normaliser le type des données reçues
        if isinstance(data, list):
            data = bytes(data)
        if isinstance(data, bytearray):
            data = bytes(data)
        if not isinstance(data, (bytes, bytearray)):
            emit("error", {"message": "Format de données audio non supporté"})
            return

        received_size = len(data)
        logger.info(f"Chunk audio reçu: {received_size} bytes ({mime_type})")
        if received_size < 100:
            logger.warning("Chunk audio trop petit, ignoré")
            return

        # Ecrire l'entrée sur disque avec la bonne extension
        ext = '.ogg' if 'ogg' in mime_type else '.webm'
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_in:
            temp_in.write(data)
            temp_in_path = temp_in.name
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
            temp_wav_path = temp_wav.name

        # Conversion FFmpeg -> WAV mono 16k
        command = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
            '-i', temp_in_path,
            '-ac', '1', '-ar', '16000', '-acodec', 'pcm_s16le',
            '-f', 'wav', temp_wav_path
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            error_msg = f"Erreur FFmpeg: {result.stderr}"
            logger.error(error_msg)
            emit("error", {"message": error_msg})
            return

        if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
            error_msg = "Le fichier WAV de sortie est vide"
            logger.error(error_msg)
            emit("error", {"message": error_msg})
            return

        # Charger et valider l'audio
        audio = AudioSegment.from_wav(temp_wav_path)
        if len(audio) < 100:
            logger.warning("Chunk audio trop court après conversion, ignoré")
            return
        samples = np.array(audio.get_array_of_samples()).astype(np.float32) / (2**15)
        del audio
        if len(samples) < 8000:
            logger.warning("Chunk audio trop court pour la transcription, ignoré")
            return

        # Transcription Whisper
        try:
            logger.info("Début de la transcription avec Whisper...")
            result = model.transcribe(
                samples,
                language="fr",
                task="transcribe",
                temperature=0.0,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4
            )
            text = result.get("text", "").strip()
        except Exception as e:
            logger.error(f"Erreur Whisper: {e}")
            emit("error", {"message": f"Erreur Whisper: {str(e)}"})
            return

        if not text:
            logger.info("Aucune parole détectée dans ce chunk")
            emit("transcription", {
                "text": accumulated_text,
                "summary": "",
                "label": {"label": "En attente...", "confidence": 0}
            })
            return

        logger.info(f"Texte transcrit: '{text}'")
        accumulated_text = (accumulated_text + " " + text).strip()

        # Résumé
        summary = ""
        if len(accumulated_text) > 100:
            try:
                logger.info("Génération du résumé...")
                summary = summarize_text(accumulated_text)
            except Exception as e:
                logger.error(f"Erreur lors de la génération du résumé: {e}")
                summary = "Erreur lors de la génération du résumé"

        # Classification
        classification = {"label": "En attente...", "confidence": 0}
        if len(accumulated_text) > 50:
            try:
                logger.info("Classification du texte...")
                classification = classify_text(accumulated_text)
            except Exception as e:
                logger.error(f"Erreur lors de la classification: {e}")
                classification = {"label": "Erreur classification", "confidence": 0}

        emit("transcription", {"text": accumulated_text, "summary": summary, "label": classification})

    except Exception as e:
        logger.error(f"❌ Erreur traitement audio: {e}")
        emit("error", {"message": f"Erreur serveur: {str(e)}"})
    finally:
        # Nettoyage des fichiers temporaires
        for p in (temp_in_path, temp_wav_path):
            try:
                if p and os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass

@socketio.on("clear_session")
def handle_clear_session():
    """Effacer la session (reset du texte accumulé)"""
    global accumulated_text
    accumulated_text = ""
    logger.info("🗑️ Session effacée")
    emit("status", {"message": "Session réinitialisée", "type": "info"})

@socketio.on("get_status")
def handle_get_status():
    """Retourne le statut du serveur"""
    status = {
        "whisper_loaded": model is not None,
        "device": device,
        "accumulated_length": len(accumulated_text)
    }
    emit("server_status", status)

if __name__ == "__main__":
    # Whisper loading on startup
    print("🚀 Démarrage du serveur Voice Notes...")
    
    if not load_whisper_model():
        print("❌ Impossible de charger Whisper. Arrêt du serveur.")
        exit(1)
    
    print(f"✅ Serveur prêt sur http://0.0.0.0:5000")
    print(f"🎤 Modèle Whisper: small ({device})")
    print(f"🧠 Résumé et classification activés")
    
    # Run the server
    socketio.run(
        app, 
        host="0.0.0.0", 
        port=5000,
        debug=True,
        allow_unsafe_werkzeug=True  # For development only
    )