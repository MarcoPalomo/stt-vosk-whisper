class VoiceRecorder {
    constructor() {
        this.socket = io();
        this.mediaRecorder = null;
        this.audioContext = null;
        this.analyser = null;
        this.dataArray = null;
        this.isRecording = false;
        this.recordedChunks = [];
        
        this.initializeElements();
        this.setupEventListeners();
        this.setupSocketListeners();
        this.setupVisualizer();
    }

    initializeElements() {
        this.startBtn = document.getElementById('startBtn');
        this.stopBtn = document.getElementById('stopBtn');
        this.clearBtn = document.getElementById('clearBtn');
        this.exportBtn = document.getElementById('exportBtn');
        this.copyBtn = document.getElementById('copyBtn');
        this.status = document.getElementById('status');
        this.transcription = document.getElementById('transcription');
        this.summary = document.getElementById('summary');
        this.classification = document.getElementById('classification');
        this.label = document.getElementById('label');
        this.canvas = document.getElementById('visualizer');
        this.canvasCtx = this.canvas.getContext('2d');
    }

    setupEventListeners() {
        this.startBtn.addEventListener('click', () => this.startRecording());
        this.stopBtn.addEventListener('click', () => this.stopRecording());
        this.clearBtn.addEventListener('click', () => this.clearResults());
        this.exportBtn.addEventListener('click', () => this.exportResults());
        this.copyBtn.addEventListener('click', () => this.copyTranscription());
    }

    setupSocketListeners() {
        this.socket.on('transcription', (data) => {
            this.updateTranscription(data.text);
            this.updateSummary(data.summary);
            this.updateClassification(data.label);
        });

        this.socket.on('connect', () => {
            this.updateStatus('Connecté au serveur', 'success');
        });

        this.socket.on('disconnect', () => {
            this.updateStatus('Déconnecté du serveur', 'error');
        });
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ 
                audio: {
                    sampleRate: 16000,
                    channelCount: 1,
                    echoCancellation: true,
                    noiseSuppression: true
                } 
            });

            this.setupAudioContext(stream);

            // Choisir dynamiquement un type MIME supporté
            const candidateTypes = [
                'audio/webm;codecs=opus',
                'audio/ogg;codecs=opus',
                'audio/webm',
                'audio/ogg'
            ];
            let selectedType = '';
            for (const t of candidateTypes) {
                if (MediaRecorder.isTypeSupported(t)) {
                    selectedType = t;
                    break;
                }
            }
            // Fallback sans préciser le type si rien n'est supporté
            const options = selectedType ? { mimeType: selectedType } : {};
            this.selectedMimeType = selectedType || 'audio/webm;codecs=opus';

            this.mediaRecorder = new MediaRecorder(stream, options);

            this.recordedChunks = [];

            this.mediaRecorder.ondataavailable = async (event) => {
                if (event.data && event.data.size > 0) {
                    // Accumuler tous les chunks depuis le début pour garantir un WebM valide
                    this.recordedChunks.push(event.data);
                    const blob = new Blob(this.recordedChunks, { type: this.selectedMimeType });
                    await this.sendAudioBlob(blob);
                }
            };

            this.mediaRecorder.start(1000); // Chunk toutes les 1 secondes
            this.isRecording = true;

            this.updateStatus('🎤 Enregistrement en cours...', 'recording');
            this.startBtn.disabled = true;
            this.stopBtn.disabled = false;

            this.visualize();

        } catch (error) {
            console.error('Erreur accès microphone:', error);
            this.updateStatus('❌ Erreur accès microphone', 'error');
        }
    }

    setupAudioContext(stream) {
        this.audioContext = new (window.AudioContext || window.webkitAudioContext)();
        this.analyser = this.audioContext.createAnalyser();
        const source = this.audioContext.createMediaStreamSource(stream);
        source.connect(this.analyser);
        
        this.analyser.fftSize = 256;
        this.dataArray = new Uint8Array(this.analyser.frequencyBinCount);
    }

    stopRecording() {
        if (this.mediaRecorder && this.isRecording) {
            this.mediaRecorder.stop();
            this.isRecording = false;
            
            this.updateStatus('⏹️ Enregistrement arrêté', 'stopped');
            this.startBtn.disabled = false;
            this.stopBtn.disabled = true;

            // Arrêter les tracks audio
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
    }

    async sendAudioBlob(blob) {
        try {
            // Envoyer directement l'ArrayBuffer (binaire) pour que Socket.IO transporte en binaire
            const arrayBuffer = await blob.arrayBuffer();
            // Émettre 2 arguments: (binaire, mimeType)
            this.socket.emit('audio_chunk', arrayBuffer, this.selectedMimeType);
        } catch (error) {
            console.error('Erreur lors de l\'envoi audio:', error);
            this.updateStatus(' Erreur d\'envoi audio', 'error');
        }
    }

    updateTranscription(text) {
        this.transcription.innerHTML = text.replace(/\n/g, '<br>');
        this.transcription.scrollTop = this.transcription.scrollHeight;
    }

    updateSummary(summary) {
        this.summary.innerHTML = summary.replace(/\n/g, '<br>');
    }

    updateClassification(labelData) {
        if (typeof labelData === 'object') {
            this.label.textContent = labelData.label || 'Non classé';
            this.label.className = `label ${labelData.label?.toLowerCase()}`;
            
            if (labelData.confidence) {
                const confidence = document.getElementById('confidence');
                confidence.textContent = `Confiance: ${(labelData.confidence * 100).toFixed(1)}%`;
            }
        } else {
            this.label.textContent = labelData || 'Non classé';
        }
    }

    updateStatus(message, type = 'info') {
        this.status.textContent = message;
        this.status.className = `status ${type}`;
    }

    setupVisualizer() {
        this.canvas.width = 800;
        this.canvas.height = 100;
    }

    visualize() {
        if (!this.isRecording || !this.analyser) return;

        requestAnimationFrame(() => this.visualize());

        this.analyser.getByteFrequencyData(this.dataArray);

        this.canvasCtx.fillStyle = '#1a1a1a';
        this.canvasCtx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        const barWidth = (this.canvas.width / this.dataArray.length) * 2.5;
        let barHeight;
        let x = 0;

        for (let i = 0; i < this.dataArray.length; i++) {
            barHeight = (this.dataArray[i] / 255) * this.canvas.height;

            const r = barHeight + 25 * (i / this.dataArray.length);
            const g = 250 * (i / this.dataArray.length);
            const b = 50;

            this.canvasCtx.fillStyle = `rgb(${r},${g},${b})`;
            this.canvasCtx.fillRect(x, this.canvas.height - barHeight, barWidth, barHeight);

            x += barWidth + 1;
        }
    }

    clearResults() {
        this.transcription.textContent = 'Votre transcription avec ponctuation automatique apparaîtra ici...';
        this.summary.textContent = 'Le résumé sera généré automatiquement...';
        this.label.textContent = 'Attente...';
        this.label.className = 'label';
        document.getElementById('confidence').textContent = '';
        this.updateStatus('Résultats effacés', 'info');
    }

    exportResults() {
        const data = {
            transcription: this.transcription.textContent,
            summary: this.summary.textContent,
            classification: this.label.textContent,
            timestamp: new Date().toISOString()
        };

        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `voice-notes-${new Date().toISOString().slice(0, 10)}.json`;
        a.click();
        URL.revokeObjectURL(url);

        this.updateStatus('Résultats exportés', 'success');
    }

    copyTranscription() {
        navigator.clipboard.writeText(this.transcription.textContent)
            .then(() => this.updateStatus('Texte copié !', 'success'))
            .catch(() => this.updateStatus('Erreur copie', 'error'));
    }
}

// Initialiser l'enregistreur au chargement de la page
document.addEventListener('DOMContentLoaded', () => {
    new VoiceRecorder();
});