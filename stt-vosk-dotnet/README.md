# VoiceNoteAppByLud

A **Windows 11 desktop application (WPF, 100% C#)** that allows you to take notes **hands-free** using your built-in microphone.  
It runs completely **offline**, leveraging **Vosk** for Speech-to-Text (STT) in French and **T5 (ONNXRuntime)** for automatic text summarization.  

---

## Features

- **Real-time speech recognition** with [Vosk](https://alphacephei.com/vosk) (offline STT)  
- **Automatic summarization** using [T5 model](https://huggingface.co/t5-small) converted to ONNX  
- **100% C# / WPF** with a simple and smooth graphical interface  
- **Offline & private** → no internet required for transcription or summarization  
- **Portable setup** → just download the models and run  


## Project Structure

```

VoiceNoteAppByLud/
├── App.xaml
├── App.xaml.cs
├── MainWindow\.xaml
├── MainWindow\.xaml.cs
├── VoiceNoteAppByLud.csproj
├── Models/
│   ├── vosk-model-fr-0.6-linto-2.2.0/   # Vosk French STT model
│   └── t5-small/                        # T5 ONNX + tokenizer
│       ├── t5-small.onnx
│       └── tokenizer/
│           └── tokenizer.json
└── bin/                                 # Build output (.exe)

```



## Requirements

- **.NET 8 SDK** (or newer) → [Download here](https://dotnet.microsoft.com/download)  
- **NuGet Packages**:
  - `VoskSharp`
  - `NAudio`
  - `Microsoft.ML.OnnxRuntime`
  - `Tokenizer.NET`

Install with:

```powershell
dotnet add package VoskSharp
dotnet add package NAudio
dotnet add package Microsoft.ML.OnnxRuntime
dotnet add package Tokenizer.NET
```


## Models

1. **Download French Vosk model**:
   [vosk-model-fr-0.6-linto-2.2.0](https://alphacephei.com/vosk/models/vosk-model-fr-0.6-linto-2.2.0.zip)
   → Extract into `Models/vosk-model-fr-0.6-linto-2.2.0/`

2. **Export T5 (summarization) to ONNX**:
   Use Hugging Face Transformers to export `t5-small` into ONNX format and save the tokenizer JSON.
   Place them in:

   ```
   Models/t5-small/t5-small.onnx
   Models/t5-small/tokenizer/tokenizer.json
   ```

---

## ▶️ Run the Application

```powershell
dotnet build VoiceNoteAppByLud.csproj
dotnet run --project VoiceNoteAppByLud.csproj
```

To generate a standalone **.exe**:

```powershell
dotnet publish VoiceNoteAppByLud.csproj -c Release -r win-x64 --self-contained true
```

The executable will be located in:

```
bin/Release/net8.0-windows/win-x64/publish/VoiceNoteAppByLud.exe
```

---

## Usage

1. Launch the app.
2. Click **Start Recording** → Speak naturally.
3. Live transcription appears in the app.
4. Click **Summarize** → Get a condensed version of your notes.

---

## Roadmap

* [ ] Multi-language support (English, Spanish, etc.)
* [x] Save transcripts as `.txt` or `.docx`
* [ ] Hotkey for quick recording during meetings
* [ ] Integration with cloud storage (optional)

---

## Author

**Ludovic Marco P.**
Platform Engineer | Cloud & AI Enthusiast

---

## 📜 License

MIT License.
Feel free to fork, modify, and use for your personal or professional needs.
