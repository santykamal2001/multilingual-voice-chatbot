import os
import uuid
import torch
import numpy as np
import soundfile as sf
import tempfile
import whisper
import scipy.io.wavfile as wavfile
import base64 # For encoding/decoding audio data
from dotenv import load_dotenv
from langdetect import detect
from transformers import pipeline
from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS

# LangChain core
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langcodes import Language

# --- Flask App Initialization ---
app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not GEMINI_API_KEY:
    print("Error: GEMINI_API_KEY not found in .env")
    # In a production app, you might raise an exception or handle this more gracefully.
    # For now, we'll let the app start but it will fail if the key is used.
if not GOOGLE_CSE_ID:
    print("Error: GOOGLE_CSE_ID not found in .env")
    # Same as above for CSE_ID

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# === Load Whisper ASR ===
# ASR model loaded once when the Flask app starts
print("Loading Whisper ASR model...")
try:
    asr_model = whisper.load_model("base")
    print("Whisper model loaded.")
except Exception as e:
    print(f"Failed to load Whisper model: {e}")
    asr_model = None # Set to None if loading fails

# === Whisper Transcription Function (adapted for web) ===
def transcribe_speech_from_bytes(audio_bytes):
    """
    Transcribes audio from bytes data.
    Args:
        audio_bytes: Raw audio bytes (e.g., from a WAV file).
    Returns:
        Transcribed text.
    """
    if not asr_model:
        return "Error: ASR model not loaded."

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        tmpfile.write(audio_bytes)
        tmp_path = tmpfile.name
    try:
        print(f"Transcribing audio from {tmp_path}...")
        result = asr_model.transcribe(tmp_path)
        print(f"Transcribed: {result['text'].strip()}")
        return result['text'].strip()
    finally:
        os.remove(tmp_path) # Clean up the temporary file

# === Safe Output Parser ===
class SafeOutputParser(BaseOutputParser):
    def parse(self, text: str) -> str:
        if "Final Answer:" in text:
            return text.split("Final Answer:")[-1].strip()
        elif "Answer:" in text:
            return text.split("Answer:")[-1].strip()
        else:
            lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
            return lines[-1] if lines else text.strip()

# === Initialize Gemini LLM ===
print(f"Initializing Gemini LLM...")
try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=GEMINI_API_KEY,
        temperature=0.2,
    )
    print("LLM ready.")
except Exception as e:
    print(f"Failed to initialize LLM: {e}")
    llm = None

# === Google Search Tool ===
print("Setting up Google Search Tool...")
try:
    search = GoogleSearchAPIWrapper(google_api_key=GEMINI_API_KEY, google_cse_id=GOOGLE_CSE_ID)
    tools = [
        Tool(
            name="Google Search",
            func=search.run,
            description="Use for questions about current events or real-time information. Input should be a search query."
        )
    ]
    print("Google Search Tool ready.")
except Exception as e:
    print(f"Failed to set up Google Search Tool: {e}")
    tools = [] # No tools if setup fails

# === TTS Models ===
TTS_MODELS = {
    "ta": "facebook/mms-tts-tam",
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "kn": "facebook/mms-tts-kan",
    "en": "facebook/mms-tts-eng"
}
tts_pipelines = {}

def get_tts_pipeline(lang_code):
    if lang_code not in tts_pipelines:
        model_name = TTS_MODELS.get(lang_code, "facebook/mms-tts-eng")
        print(f"🔊 Loading TTS model: {model_name}")
        try:
            tts_pipelines[lang_code] = pipeline("text-to-speech", model=model_name, device=device)
        except Exception as e:
            print(f"⚠️ Failed to load {model_name}, falling back to English. Error: {e}")
            tts_pipelines[lang_code] = pipeline("text-to-speech", model="facebook/mms-tts-eng", device=device)
    return tts_pipelines[lang_code]

# === ReAct Agent Setup ===
print("Setting up Agent with ReAct...")
agent_executor = None
if llm and tools:
    REACT_PROMPT = """Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: {input}
    {agent_scratchpad}"""

    prompt = PromptTemplate.from_template(REACT_PROMPT)
    agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        output_parser=SafeOutputParser()
    )
    print("Agent ready.")
else:
    print("Agent not initialized due to missing LLM or tools.")

# --- API Endpoint ---
@app.route('/ask', methods=['POST'])
def ask_ai():
    data = request.json
    user_input = data.get('text_input')
    audio_data_b64 = data.get('audio_input') # Base64 encoded audio

    if not user_input and not audio_data_b64:
        return jsonify({"error": "No text or audio input provided"}), 400

    if audio_data_b64:
        try:
            # Decode base64 audio and transcribe
            audio_bytes = base64.b64decode(audio_data_b64)
            user_input = transcribe_speech_from_bytes(audio_bytes)
            if user_input.startswith("Error:"):
                return jsonify({"error": user_input}), 500
        except Exception as e:
            return jsonify({"error": f"Failed to process audio input: {e}"}), 400

    if not user_input: # If transcription failed or no text input was given
        return jsonify({"error": "Could not get valid text input from audio or text field"}), 400

    try:
        input_lang_code = detect(user_input)
        input_lang_name = Language.get(input_lang_code).display_name().capitalize()
    except Exception as e:
        print(f"Language detection failed: {e}. Defaulting to English.")
        input_lang_code = "en"
        input_lang_name = "English"

    modified_input = f"Please answer the following question in {input_lang_name}:\n{user_input}"

    if not agent_executor:
        return jsonify({"error": "AI agent not initialized. Check backend logs for errors."}), 500

    try:
        response = agent_executor.invoke({"input": modified_input})
        answer = response.get("output", "⚠️ Sorry, I couldn't generate an answer.").strip()

        # Truncate for TTS safety and efficiency
        tts_answer = answer[:500]

        # Generate TTS audio
        tts_pipeline = get_tts_pipeline(input_lang_code)
        audio_output = tts_pipeline(tts_answer)
        audio_data = np.squeeze(audio_output["audio"])
        sampling_rate = audio_output["sampling_rate"]

        # Save audio to a temporary file and then read its bytes for base64 encoding
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_file:
            sf.write(tmp_audio_file.name, audio_data, sampling_rate)
            tmp_audio_path = tmp_audio_file.name

        try:
            with open(tmp_audio_path, "rb") as f:
                audio_bytes_for_b64 = f.read()
            audio_b64 = base64.b64encode(audio_bytes_for_b64).decode('utf-8')
        finally:
            os.remove(tmp_audio_path) # Clean up the temporary audio file

        return jsonify({
            "answer": answer,
            "audio_b64": audio_b64,
            "sampling_rate": sampling_rate,
            "language_code": input_lang_code,
            "language_name": input_lang_name
        })

    except Exception as e:
        print(f"Error during agent invocation or TTS: {e}")
        return jsonify({"error": f"An internal server error occurred: {e}"}), 500

# To run the Flask app:
# Save this code as, e.g., `app.py`
# Make sure you have `flask`, `flask-cors`, `langchain`, `langchain-google-genai`,
# `langchain-google-community`, `python-dotenv`, `transformers`, `torch`, `numpy`,
# `soundfile`, `scipy`, `langdetect`, `langcodes`, `whisper` installed.
# You can install them via pip:
# pip install Flask Flask-Cors python-dotenv transformers torch numpy soundfile scipy langdetect langcodes-py google-search-results-api-wrapper langchain-google-genai openai-whisper
# Then run from your terminal: `flask run`
# By default, it will run on http://127.0.0.1:5000
