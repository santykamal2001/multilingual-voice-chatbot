#  Multilingual Voice Chatbot (ASR → RAG → TTS)

A real-time voice chatbot that supports multilingual interaction using **Whisper for voice input**, **Gemini LLM with LangChain for intelligent answers**, **Google Search for real-time info**, and **Facebook MMS-TTS for speech output** — all built in Python.

> Built by [Santhos Kamal A B](https://www.linkedin.com/in/santhoskamal), MS in Computer Science @ Oregon State University.

---

##  What This Project Does

- Takes voice or text input in Tamil, Hindi, Telugu, Kannada, or English
- Converts speech to text using **OpenAI's Whisper**
- Uses **Gemini Pro (LLM)** via LangChain to understand and generate answers
- Uses **Google Search** for real-time queries
- Converts answers to speech in the same language using **Facebook MMS-TTS**
- Plays the response audio and shows the answer as text

---

##  Technologies Used

| Component             | Technology                                        |
|----------------------|---------------------------------------------------|
| ASR (Voice Input)     | [OpenAI Whisper](https://github.com/openai/whisper) |
| LLM (Reasoning)       | Gemini 2.5 Pro via `langchain-google-genai`      |
| TTS (Voice Output)    | [facebook/mms-tts-*](https://huggingface.co/facebook) |
| Realtime Search       | LangChain + Google Programmable Search API       |
| Language Detection    | `langdetect`, `langcodes`                        |
| Audio Handling        | `sounddevice`, `soundfile`, `numpy`, `torch`     |
| Orchestration         | [LangChain](https://python.langchain.com/)       |
| Environment Config    | `python-dotenv`                                  |

---

##  Requirements

Your `requirements.txt` should contain:

# create an env
GEMINI_API_KEY=your_google_api_key_here
GOOGLE_CSE_ID=your_custom_search_engine_id_here
