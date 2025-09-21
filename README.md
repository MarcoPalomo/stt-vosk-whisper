# 🎤 STT (Speech-To-Text) - Speech Recognition Projects

Comparison and exploration of different approaches for **automatic speech recognition** with **smart punctuation**.

## Overview

This repository contains **two implementations** of speech-to-text systems:

| 🏗️ **Approach** | 🔧 **Technology** | 🎯 **Advantage** | ⚡ **Performance** |
|------------------|-------------------|------------------|-------------------|
| **C# WPF Desktop** | Vosk + Microsoft.ML | Native Windows app | ⭐⭐⭐ |
| **Python Web** | Whisper + Flask | Automatic punctuation | ⭐⭐⭐⭐⭐ |

## Project Structure

```
STT/
├── README.md                    # 👈 This file
├── VoiceNoteAppByLud/          # 🖥️ C# WPF Application
│   ├── MainWindow.xaml
│   ├── MainWindow.xaml.cs
│   ├── App.xaml
│   ├── VoiceNotesApp.csproj
│   └── Models/
│       ├── vosk-model-fr/      # French Vosk model
│       └── encoder_model.onnx   # T5 for summarization
└── stt-whisper-py/             # 🌐 Python Web App
    ├── app.py                  # Flask + SocketIO
    ├── requirements.txt
    ├── static/                 # Web interface
    ├── templates/
    └── utils/                  # AI for summarization & classification
```

## Approach Comparison

### C# WPF Version (`VoiceNoteAppByLud/`)

**Advantages:**
- High-performance **native Windows** application
- **Modern WPF** interface with XAML
- No network dependency
- Complete system integration

**Disadvantages:**
- **No automatic punctuation** (Vosk limitation)
- .NET dependency complexity
- Windows-only
- Complex T5 tokenizer configuration

**Recommended for:** Enterprise Windows applications, offline environments

---

### Python Web Version (`stt-whisper-py/`)

**Advantages:**
- **Perfect automatic punctuation** (Whisper)
- **Responsive web** interface (mobile/desktop)
- **Real-time** via WebSocket
- **Automatic summarization + classification**
- Cross-platform (Windows/Mac/Linux)

**Disadvantages:**
- Requires Python server
- More resource-intensive
- Local network dependency

**Recommended for:** Modern applications, distributed teams, rapid prototyping

## Quick Start

### Option 1: Python Web (Recommended)
```bash
cd stt-whisper-py/
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```

### Option 2: C# Desktop
```bash
cd VoiceNoteAppByLud/
dotnet build
dotnet run
```

## Use Cases

### **Meeting Notes**
- ✅ **Python**: Transcription + automatic summarization + classification
- ⚠️ **C#**: Raw transcription, manual summarization needed

### **Interviews**  
- ✅ **Python**: Perfect punctuation, JSON export
- ⚠️ **C#**: Correct transcription but without punctuation

### **Courses & Training**
- ✅ **Python**: Automatic "course" classification, smart summarization
- ✅ **C#**: Stable local application, no network latency

### **Enterprise Environment**
- ✅ **C#**: Local security, no data on server
- ✅ **Python**: Modern interface, collaboration possible

## Technologies Used

| Component | C# WPF | Python Web |
|-----------|---------|------------|
| **STT Engine** | Vosk | OpenAI Whisper |
| **Interface** | WPF/XAML | HTML5/CSS3/JS |
| **Backend** | .NET 8 | Flask + SocketIO |
| **Summarization** | T5 ONNX | Transformers (mT5) |
| **Classification** | ❌ | LightGBM |
| **Audio** | NAudio | Web Audio API |
| **Real-time** | ❌ | WebSocket |

## Performance Metrics

| Criteria | C# WPF | Python Web | Winner |
|----------|---------|------------|------------|
| **Punctuation** | ❌ | ✅ Perfect | Python |
| **Speed** | ✅ Fast | ⚡ Real-time | Tie |
| **Accuracy** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | 🐍 Python |
| **Setup Ease** | ⚠️ Complex | ✅ Simple | 🐍 Python |
| **Stability** | ✅ Robust | ✅ Robust | 🤝 Tie |
| **Security** | ✅ Local | ⚠️ Server | 🖥️ C# |

## Recommendations

### **Choose Python Web if:**
- You want **automatic punctuation** (key feature)
- Need modern and responsive interface
- Require automatic summarization and classification
- Distributed team or collaboration needed
- Rapid prototyping

### **Choose C# WPF if:**
- Maximum security required (local data only)
- 100% Windows environment
- Traditional enterprise application
- Critical native performance
- No server allowed

## Future Development

### Upcoming Python Features:
- [ ] Extended multilingual support
- [ ] REST API for integration
- [ ] Offline mode with local Whisper
- [ ] Advanced analytics dashboard

### Possible C# Improvements:
- [ ] Azure Speech Services integration (punctuation)
- [ ] Whisper.NET plugin
- [ ] More modern interface (WinUI 3)

## 🤝 Contributing

**Pull Requests** are welcome for both projects!

- **Issues**: Report bugs and suggestions
- **Features**: Propose new features
- **Docs**: Improve documentation

## 📄 License

**MIT License** - Feel free to use for personal and commercial projects.

---

**🎤 Happy transcribing!** - *May the AI force be with your words*.