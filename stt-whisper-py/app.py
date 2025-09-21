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
def handle_audio_chunk(data):
    """
    Réception et traitement d'un chunk audio depuis le navigateur
    
    Args:
        data: Données audio en bytes (format WebM/Opus)
    """
    global accumulated_text
    
    try:
        if model is None:
            emit("error", {"message": "Modèle Whisper non chargé"})
            return
        
        logger.info(f"Chunk audio reçu: {len(data)} bytes")
        
        try:
            # Créer un fichier temporaire pour le chunk audio
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
                temp_webm.write(data)
                temp_webm_path = temp_webm.name
            
            try:
                # Charger l'audio avec pydub
                audio_segment = AudioSegment.from_file(temp_webm_path, format='webm')
                
                # Convertir en mono et 16kHz
                audio_segment = audio_segment.set_channels(1).set_frame_rate(16000)
                
                # Convertir en numpy array
                audio = np.array(audio_segment.get_array_of_samples())
                audio = audio.astype(np.float32) / (2**15)  # Convertir en float32 et normaliser
                
                # Supprimer le fichier temporaire
                os.unlink(temp_webm_path)
                
            except Exception as e:
                logger.error(f"❌ Erreur conversion audio: {e}")
                emit("error", {"message": f"Erreur conversion audio: {e}"})
                if os.path.exists(temp_webm_path):
                    os.unlink(temp_webm_path)
                return
            
        except Exception as e:
            logger.error(f"❌ Erreur manipulation fichier temporaire: {e}")
            emit("error", {"message": "Erreur lors du traitement audio"})
            return
        
        # Vérifier la longueur minimum (éviter les chunks trop courts)
        if len(audio) < 1600:  # Moins de 0.1 seconde à 16kHz
            logger.warning("⚠️ Chunk audio trop court, ignoré")
            return
        
        # Transcription avec Whisper
        logger.info("Transcription avec Whisper...")
        
        result = model.transcribe(
            audio,
            language="fr",  # Français
            task="transcribe",
            temperature=0.0,  # Déterministe
            no_speech_threshold=0.6,  # Seuil de détection de parole
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4
        )
        
        text = result["text"].strip()
        
        if not text:
            logger.info("🔇 Pas de parole détectée dans ce chunk")
            emit("transcription", {
                "text": accumulated_text,
                "summary": "",
                "label": {"label": "En attente...", "confidence": 0}
            })
            return
        
        logger.info(f"📝 Texte transcrit: '{text}'")
        
        # Accumuler le texte
        if text:
            accumulated_text += " " + text
            accumulated_text = accumulated_text.strip()
        
        # Summary (only if enough text)
        summary = ""
        if len(accumulated_text) > 100:
            try:
                logger.info("Génération du résumé...")
                summary = summarize_text(accumulated_text)
            except Exception as e:
                logger.error(f"Erreur résumé: {e}")
                summary = "Erreur lors de la génération du résumé"
        
        # Classification (only if enough text)
        classification = {"label": "En attente...", "confidence": 0}
        if len(accumulated_text) > 50:
            try:
                logger.info("🏷️ Classification du texte...")
                classification = classify_text(accumulated_text)
            except Exception as e:
                logger.error(f"Erreur classification: {e}")
                classification = {"label": "Erreur classification", "confidence": 0}
        
        # Send results to client
        emit("transcription", {
            "text": accumulated_text,
            "summary": summary,
            "label": classification
        })
        
        logger.info("✅ Résultats envoyés au client")
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement audio: {e}")
        emit("error", {"message": f"Erreur serveur: {str(e)}"})

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