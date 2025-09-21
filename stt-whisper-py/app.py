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
        
        if len(data) < 100:  # Chunk trop petit, probablement vide
            logger.warning("Chunk audio trop petit, ignoré")
            return
            
        try:
            # Créer un fichier temporaire pour le chunk audio
            with tempfile.NamedTemporaryFile(suffix='.webm', delete=False) as temp_webm:
                temp_webm.write(data)
                temp_webm_path = temp_webm.name
            
            # Créer un fichier WAV temporaire pour la sortie
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_wav:
                temp_wav_path = temp_wav.name
            
            try:
                # Convertir WebM en WAV avec FFmpeg
                command = [
                    'ffmpeg',
                    '-y',  # Overwrite output file if it exists
                    '-i', temp_webm_path,  # Input file
                    '-ac', '1',  # Mono
                    '-ar', '16000',  # 16kHz sample rate
                    '-acodec', 'pcm_s16le',  # 16-bit PCM
                    '-f', 'wav',  # WAV format
                    temp_wav_path  # Output file
                ]
                
                # Exécuter la commande FFmpeg avec timeout
                result = subprocess.run(
                    command, 
                    capture_output=True, 
                    text=True,
                    timeout=5  # 5 secondes de timeout
                )
                
                if result.returncode != 0:
                    error_msg = f"Erreur FFmpeg: {result.stderr}"
                    logger.error(error_msg)
                    emit("error", {"message": error_msg})
                    return
                
                # Vérifier que le fichier WAV a été créé
                if not os.path.exists(temp_wav_path) or os.path.getsize(temp_wav_path) == 0:
                    error_msg = "Le fichier WAV de sortie est vide"
                    logger.error(error_msg)
                    emit("error", {"message": error_msg})
                    return
                
                # Charger l'audio avec pydub
                audio = AudioSegment.from_wav(temp_wav_path)
                
                # Vérifier la durée minimale (au moins 100ms)
                if len(audio) < 100:  # 100ms
                    logger.warning("Chunk audio trop court après conversion, ignoré")
                    return
                
                # Convertir en array numpy
                samples = np.array(audio.get_array_of_samples())
                samples = samples.astype(np.float32) / (2**15)  # Normaliser entre -1 et 1
                
                # Libérer la mémoire
                del audio
                
                # Vérifier la longueur minimum (au moins 0.5 seconde à 16kHz)
                if len(samples) < 8000:  # 0.5 seconde à 16kHz
                    logger.warning("Chunk audio trop court pour la transcription, ignoré")
                    return
                
                try:
                    # Transcrire avec Whisper
                    logger.info("Début de la transcription avec Whisper...")
                    
                    result = model.transcribe(
                        samples,
                        language="fr",  # Français
                        task="transcribe",
                        temperature=0.0,  # Déterministe
                        no_speech_threshold=0.6,  # Seuil de détection de parole
                        logprob_threshold=-1.0,
                        compression_ratio_threshold=2.4
                    )
                    
                    text = result["text"].strip()
                    
                    if not text:
                        logger.info("Aucune parole détectée dans ce chunk")
                        emit("transcription", {
                            "text": accumulated_text,
                            "summary": "",
                            "label": {"label": "En attente...", "confidence": 0}
                        })
                        return
                    
                    logger.info(f"Texte transcrit: '{text}'")
                    
                    # Accumuler le texte
                    if text:
                        accumulated_text += " " + text
                        accumulated_text = accumulated_text.strip()
                    
                    # Générer un résumé si assez de texte
                    summary = ""
                    if len(accumulated_text) > 100:
                        try:
                            logger.info("Génération du résumé...")
                            summary = summarize_text(accumulated_text)
                        except Exception as e:
                            logger.error(f"Erreur lors de la génération du résumé: {e}")
                            summary = "Erreur lors de la génération du résumé"
                    
                    # Classification du texte si assez de contenu
                    classification = {"label": "En attente...", "confidence": 0}
                    if len(accumulated_text) > 50:
                        try:
                            logger.info("Classification du texte...")
                            classification = classify_text(accumulated_text)
                        except Exception as e:
                            logger.error(f"Erreur lors de la classification: {e}")
                            classification = {"label": "Erreur classification", "confidence": 0}
                    
                    # Envoyer les résultats au client
                    emit("transcription", {
                        "text": accumulated_text,
                        "summary": summary,
                        "label": classification
                    })
                    
                    logger.info("Résultats envoyés avec succès")
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la transcription: {str(e)}")
                    emit("error", {"message": f"Erreur de transcription: {str(e)}"})
                
            except subprocess.CalledProcessError as e:
                error_msg = f"Erreur FFmpeg: {e.stderr}"
                logger.error(error_msg)
                emit("error", {"message": error_msg})
                
            except Exception as e:
                error_msg = f"Erreur de traitement audio: {str(e)}"
                logger.error(error_msg)
                emit("error", {"message": error_msg})
                
            finally:
                # Nettoyer les fichiers temporaires
                for file_path in [temp_webm_path, temp_wav_path]:
                    try:
                        if file_path and os.path.exists(file_path):
                            os.unlink(file_path)
                    except Exception as e:
                        logger.error(f"Erreur lors de la suppression de {file_path}: {e}")
        
        except Exception as e:
            error_msg = f"Erreur lors du traitement du fichier temporaire: {str(e)}"
            logger.error(error_msg)
            emit("error", {"message": error_msg})
            return
            
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