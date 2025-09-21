using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Windows;
using Vosk;
using NAudio.Wave;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Microsoft.ML.Tokenizers;
using System.Collections.Generic;

namespace VoiceNotesApp
{
    public partial class MainWindow : Window
    {
        private VoskRecognizer? recognizer;
        private WaveInEvent? waveIn;
        private string transcript = "";
        private InferenceSession? summarizer;
        private Tokenizer? tokenizer;
        private bool isRecording = false;

        public MainWindow()
        {
            InitializeComponent();
            InitSummarizer();
            SummaryBox.IsReadOnly = true; // Résumé non éditable
        }

        // Initialisation du modèle de résumé (ONNX seulement pour l'instant)
        private void InitSummarizer()
        {
            try
            {
                // Charger seulement le modèle ONNX pour l'instant
                summarizer = new InferenceSession("Models/encoder_model.onnx");
                
                StatusLabel.Text = "✅ Modèle ONNX chargé (tokenizer désactivé temporairement).";
            }
            catch (Exception ex)
            {
                StatusLabel.Text = "❌ Erreur chargement modèles.";
                MessageBox.Show("Erreur lors du chargement des modèles :\n" + ex.Message);
            }
        }

        // Bouton Démarrer Réunion (toggle Start/Stop)
        private void StartBtn_Click(object sender, RoutedEventArgs e)
        {
            if (isRecording)
            {
                try
                {
                    waveIn?.StopRecording();
                    waveIn?.Dispose();
                    waveIn = null;
                    isRecording = false;
                    StatusLabel.Text = "⏹️ Enregistrement arrêté.";
                    StartBtn.Content = "🎤 Démarrer Réunion";
                }
                catch (Exception ex)
                {
                    StatusLabel.Text = "❌ Erreur à l'arrêt.";
                    MessageBox.Show("Erreur lors de l'arrêt de l'enregistrement :\n" + ex.Message);
                }
                return;
            }

            // Vérifier que le modèle Vosk existe
            if (!Directory.Exists("Models/vosk-model-fr"))
            {
                StatusLabel.Text = "❌ Modèle Vosk manquant.";
                MessageBox.Show("Le modèle Vosk français n'a pas été trouvé dans Models/vosk-model-fr/\n" +
                               "Veuillez télécharger le modèle d'abord.");
                return;
            }

            try
            {
                StatusLabel.Text = "🔄 Chargement du modèle Vosk...";
                
                Vosk.Vosk.SetLogLevel(0);
                var model = new Model("Models/vosk-model-fr");
                recognizer = new VoskRecognizer(model, 16000.0f);

                StatusLabel.Text = "🔄 Initialisation du microphone...";

                waveIn = new WaveInEvent
                {
                    WaveFormat = new WaveFormat(16000, 1)
                };
                
                waveIn.DataAvailable += (s, a) =>
                {
                    if (recognizer?.AcceptWaveform(a.Buffer, a.BytesRecorded) == true)
                    {
                        var json = recognizer.Result();
                        var voskResult = JsonSerializer.Deserialize<VoskResult>(json);
                        if (!string.IsNullOrWhiteSpace(voskResult?.text))
                        {
                            transcript += voskResult.text + " ";
                            Dispatcher.Invoke(() => TranscriptBox.Text = transcript);
                        }
                    }
                };
                
                waveIn.RecordingStopped += (s, a) =>
                {
                    waveIn?.Dispose();
                    waveIn = null;
                    isRecording = false;
                    Dispatcher.Invoke(() => {
                        StatusLabel.Text = "⏹️ Enregistrement arrêté.";
                        StartBtn.Content = "🎤 Démarrer Réunion";
                    });
                };
                
                waveIn.StartRecording();
                isRecording = true;
                StartBtn.Content = "⏹️ Arrêter";
                StatusLabel.Text = "🎤 Enregistrement en cours... (Cliquez à nouveau pour arrêter)";
            }
            catch (Exception ex)
            {
                StatusLabel.Text = "❌ Erreur démarrage micro.";
                MessageBox.Show("Erreur lors du démarrage de l'enregistrement :\n" + ex.Message + 
                               "\n\nVérifiez que :\n" +
                               "- Le modèle Vosk est dans Models/vosk-model-fr/\n" +
                               "- Votre microphone fonctionne\n" +
                               "- Les permissions microphone sont accordées");
            }
        }

        // Bouton Résumer
        private void SummarizeBtn_Click(object sender, RoutedEventArgs e)
        {
            if (string.IsNullOrWhiteSpace(transcript))
            {
                MessageBox.Show("Aucune transcription disponible.");
                return;
            }

            try
            {
                string summary = Summarize(transcript);
                SummaryBox.Text = summary;
                StatusLabel.Text = "✅ Résumé généré.";
            }
            catch (Exception ex)
            {
                StatusLabel.Text = "❌ Erreur résumé.";
                MessageBox.Show("Erreur lors du résumé :\n" + ex.Message);
            }
        }

        // Bouton Enregistrer
        private void SaveBtn_Click(object sender, RoutedEventArgs e)
        {
            try
            {
                string fileName = $"Notes_{DateTime.Now:yyyyMMdd_HHmm}.txt";
                File.WriteAllText(fileName,
                    "=== Transcription ===\n\n" + transcript +
                    "\n\n=== Résumé ===\n\n" + SummaryBox.Text);

                StatusLabel.Text = $"💾 Sauvegardé : {fileName}";
            }
            catch (Exception ex)
            {
                StatusLabel.Text = "❌ Erreur sauvegarde.";
                MessageBox.Show("Erreur lors de la sauvegarde :\n" + ex.Message);
            }
        }

        // Fonction de résumé simplifiée (sans tokenizer pour l'instant)
        private string Summarize(string text)
        {
            if (summarizer == null)
            {
                return "[Erreur: modèle non initialisé]";
            }

            try
            {
                // Version simplifiée temporaire - juste retourner un résumé basique
                // jusqu'à ce qu'on résolve le problème de tokenizer
                var sentences = text.Split('.', StringSplitOptions.RemoveEmptyEntries)
                                  .Take(3)
                                  .ToArray();
                
                return string.Join(". ", sentences) + ".";
            }
            catch (Exception ex)
            {
                MessageBox.Show("Erreur lors du résumé :\n" + ex.Message);
                return "[Erreur lors du résumé]";
            }
        }
    }

    // Classe utilitaire pour parser le JSON de Vosk
    public class VoskResult
    {
        public string text { get; set; } = string.Empty;
    }
}