# STT Whisper - Voice Notes AI

Real-time speech recognition application with **OpenAI Whisper**, automatic summarization, and intelligent classification.

## Features

- 🎤 **Real-time speech recognition** with automatic punctuation (Whisper)
- 📝 **Accurate punctuation** in transcriptions
- 📋 **Smart summarization** (T5/BART)
- 🏷️ **Automatic classification** into 8 categories (LightGBM)
- 🌊 **Real-time audio visualization**
- 💾 **JSON/Clipboard export**
- 📱 **Responsive web interface**

## Quick Start

```bash
# 1. Clone the project
git clone <your-repo>
cd stt-whisper-py

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the application
python app.py
```

**→ Open http://localhost:5000 in your browser**

## Project Structure

```
stt-whisper-py/
├── app.py                 # Flask + SocketIO server
├── requirements.txt       # Python dependencies
├── models/               # Auto-generated models
├── static/
│   ├── js/recorder.js    # Browser audio capture
│   └── css/style.css     # Modern interface
├── templates/
│   └── index.html        # Web interface
└── utils/
    ├── summarizer.py     # Smart summarization
    └── classifier.py     # LightGBM classification
```

## Usage

1. **Start** the application: `python app.py`
2. **Allow** microphone access
3. **Click** "🎤 Start" and speak
4. **Watch** the magic happen:
   - Real-time transcription with punctuation
   - Improving automatic summaries
   - Intelligent content classification

## Classification Categories

| Icon | Category | Description |
|------|-----------|-------------|
| 🏢 | Meeting | Agendas, decisions, discussions |
| 👥 | Interview | Recruitment, evaluations, Q&A |
| 📚 | Course | Training, lessons, learning |
| 📊 | Presentation | Slides, demonstrations, talks |
| 💬 | Discussion | Debates, opinion exchanges |
| 🎤 | Conference | Keynotes, symposiums |
| 💡 | Brainstorming | Creativity, innovation, ideation |
| 🔀 | Other | Informal conversations, miscellaneous |

## Configuration

### Whisper Model
- **Default**: `small` (good speed/accuracy balance)
- **Alternative**: Modify `model = whisper.load_model("base")` in `app.py`
- **Options**: `tiny`, `base`, `small`, `medium`, `large`

### Summarization
- **Model**: Multilingual mT5
- **Fallback**: BART if loading fails
- **Customizable**: Edit `utils/summarizer.py`

### Classification
- **Algorithm**: LightGBM with TF-IDF
- **Training**: Built-in synthetic data
- **Auto-save**: Model saves automatically

## Main Dependencies

```
flask==2.3.3
flask-socketio==5.3.6
openai-whisper==20231117
torch>=2.0.0
transformers==4.35.2
lightgbm==4.1.0
soundfile==0.12.1
```

## Troubleshooting

### Microphone Not Working
- Check browser permissions
- Use HTTPS in production
- Test with recent Chrome/Firefox

### Whisper Loading Slowly
- Normal on first run (model download)
- Use `tiny` or `base` for faster performance
- GPU recommended for larger models

### Dependency Errors
```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

## Performance

- **Whisper small**: ~244MB, fast transcription
- **Real-time**: < 2 seconds latency
- **GPU**: Automatic CUDA acceleration if available
- **Memory**: ~1-2GB RAM recommended

## License

MIT License - Free for personal and commercial use!

## Contributing

Contributions are welcome! Open an issue or submit a PR.

---

**Made with ❤️ using OpenAI Whisper, Flask & Python**