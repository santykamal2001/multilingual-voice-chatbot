# 🎤🤖 Multilingual Voice Chatbot

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![AI](https://img.shields.io/badge/AI-Powered-red.svg)

**A real-time multilingual voice chatbot with speech-to-speech capabilities**

*ASR → LLM → TTS Pipeline supporting Tamil, Hindi, Telugu, Kannada, and English*

</div>

---

## 🌟 Features

- 🎤 **Voice Input**: Speech-to-text using OpenAI Whisper
- 🧠 **Intelligent Responses**: Powered by Google Gemini 2.5 Pro LLM
- 🔍 **Real-time Search**: Google Search integration for current information
- 🗣️ **Voice Output**: Text-to-speech in multiple languages using Facebook MMS-TTS
- 🌐 **Multilingual Support**: Tamil, Hindi, Telugu, Kannada, and English
- 🕸️ **Web API**: RESTful Flask API for easy integration
- 🤖 **ReAct Agent**: Intelligent reasoning and action framework

## 📋 Table of Contents

- [Features](#-features)
- [Demo](#-demo)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)
- [Author](#-author)

## 🎬 Demo

The chatbot can handle questions like:
- "What's the weather today?" (English)
- "फ्रांस की राजधानी क्या है?" (Hindi)
- "ఈ రోజు వార్తలు ఏమిటి?" (Telugu)
- "ಇಂದಿನ ಸುದ್ದಿ ಏನು?" (Kannada)
- "இன்றைய செய்திகள் என்ன?" (Tamil)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- FFmpeg (required by Whisper)

### Install FFmpeg

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Clone the Repository

```bash
git clone https://github.com/santykamal2001/multilingual-voice-chatbot.git
cd multilingual-voice-chatbot
```

### Create Virtual Environment

```bash
python -m venv voice_chatbot_env
source voice_chatbot_env/bin/activate  # On Windows: voice_chatbot_env\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## ⚙️ Configuration

### 1. Get API Keys

#### Google Gemini API Key
1. Go to [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

#### Google Custom Search Engine
1. Go to [Google Custom Search Engine](https://cse.google.com/cse/)
2. Create a new search engine
3. Get your Search Engine ID (CSE ID)
4. Enable the Custom Search API in [Google Cloud Console](https://console.cloud.google.com/)

### 2. Environment Variables

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_CSE_ID=your_google_custom_search_engine_id_here
```

⚠️ **Never commit your `.env` file to version control!**

## 🏃 Usage

### Start the Flask Server

```bash
# Make sure your virtual environment is activated
source voice_chatbot_env/bin/activate  # On Windows: voice_chatbot_env\Scripts\activate

# Set the Flask app
export FLASK_APP=app.py  # On Windows: set FLASK_APP=app.py

# Run the server
flask run
```

The server will start at `http://127.0.0.1:5000`

### Using the API

Send a POST request to `/ask` endpoint:

```python
import requests
import base64

# Text input example
response = requests.post('http://127.0.0.1:5000/ask', 
    json={'text_input': 'What is the capital of France?'})
print(response.json())

# Audio input example (with base64 encoded audio)
with open('audio.wav', 'rb') as audio_file:
    audio_b64 = base64.b64encode(audio_file.read()).decode('utf-8')

response = requests.post('http://127.0.0.1:5000/ask',
    json={'audio_input': audio_b64})
print(response.json())
```

## 📡 API Reference

### POST `/ask`

Send text or audio input to get AI response with audio output.

#### Request Body

```json
{
  "text_input": "Your question here",
  "audio_input": "base64_encoded_audio"
}
```

*Note: Provide either `text_input` OR `audio_input`, not both*

#### Response

```json
{
  "answer": "AI response text",
  "audio_b64": "base64_encoded_response_audio",
  "sampling_rate": 16000,
  "language_code": "en",
  "language_name": "English"
}
```

#### Error Response

```json
{
  "error": "Error description"
}
```

## 📁 Project Structure

```
multilingual-voice-chatbot/
├── app.py                 # Main Flask application
├── chatbot.py            # Core chatbot implementation
├── asr_test.py           # ASR testing utilities
├── rag_setup.py          # RAG (Retrieval Augmented Generation) setup
├── demo_chatbot.py       # Demo/testing script
├── knowledge_base.txt    # Multilingual knowledge base
├── requirements.txt      # Python dependencies
├── .env                  # Environment variables (create this)
├── .gitignore           # Git ignore rules
├── README.md            # This file
├── chroma_db/           # Vector database (auto-created)
└── voice_chatbot_env/   # Virtual environment (auto-created)
```

## 🛠️ Technologies Used

| Component | Technology | Purpose |
|-----------|------------|---------|
| **ASR** | OpenAI Whisper | Convert speech to text |
| **LLM** | Google Gemini 2.5 Pro | Generate intelligent responses |
| **TTS** | Facebook MMS-TTS | Convert text to speech |
| **Search** | Google Custom Search API | Real-time information retrieval |
| **Framework** | LangChain | LLM orchestration and agents |
| **Web API** | Flask + Flask-CORS | RESTful API server |
| **Audio** | soundfile, scipy | Audio processing |
| **ML** | PyTorch, Transformers | Model inference |
| **Language** | langdetect, langcodes | Language detection and handling |

## 🔧 Troubleshooting

### Common Issues

1. **ModuleNotFoundError**: Make sure virtual environment is activated and dependencies are installed
2. **FFmpeg not found**: Install FFmpeg using the instructions above
3. **API Key errors**: Check your `.env` file and ensure API keys are correct
4. **Audio issues**: Ensure audio files are in WAV format for best compatibility
5. **Memory issues**: The models require significant RAM; consider using smaller Whisper models

### Model Loading Times

- First run will download models (may take 5-10 minutes)
- Whisper model: ~150MB
- TTS models: ~50MB each (downloaded on demand)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Santhos Kamal Arumugam Balamurugan**
- 🎓 MEng in Computer Science @ Oregon State University
- 💼 [LinkedIn](https://www.linkedin.com/in/santhos-kamal-arumugam-balamurugan-6915b41ba/)
- 🐙 [GitHub](https://github.com/santykamal2001)

---

<div align="center">

**If you found this project helpful, please give it a ⭐!**

</div>
