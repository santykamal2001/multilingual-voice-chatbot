import os
import torch
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from transformers import pipeline
from langdetect import detect # Keep for language detection
import numpy as np
import soundfile as sf

# Optional: For playing audio
try:
    import sounddevice as sd
    SOUND_ENABLED = True
except ImportError:
    SOUND_ENABLED = False

# Load API key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# 1. Define the device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Define paths and models
DOCUMENTS_PATH = "knowledge_base.txt"
CHROMA_DB_PATH = "chroma_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# 3. Load document
if not os.path.exists(DOCUMENTS_PATH):
    print(f"Error: Document not found at {DOCUMENTS_PATH}")
    exit()

print(f"Loading document from {DOCUMENTS_PATH}...")
loader = TextLoader(DOCUMENTS_PATH)
documents = loader.load()
print(f"Loaded {len(documents)} document(s).")

# 4. Split document
print("Splitting document into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
    separators=["\n\n", "\n", " ", ""]
)
texts = text_splitter.split_documents(documents)
print(f"Split into {len(texts)} chunks.")

# 5. Create embeddings
print(f"Creating embeddings with model: {EMBEDDING_MODEL_NAME}...")
# LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2
# and will be removed in 1.0. An updated version of the class exists in the `langchain-huggingface` package.
# For now, we'll keep the current import, but consider updating to:
# from langchain_huggingface import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME, model_kwargs={'device': device})

# 6. Create ChromaDB
print("Creating and saving ChromaDB...")
db = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=CHROMA_DB_PATH
)
# As noted in your output, db.persist() is deprecated in newer Chroma versions.
# Docs are automatically persisted. You can safely remove this line.
# db.persist()
print("ChromaDB ready.")

# 7. Initialize Gemini LLM
print("Initializing Gemini LLM...")
# ***** IMPORTANT CHANGE HERE: Updated model name *****
# Use 'gemini-2.5-pro' for the most capable model or 'gemini-2.5-flash' for speed/cost.
# It's recommended to first run a script to list available models to confirm names.
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro", # Changed from "gemini-pro"
    google_api_key=GEMINI_API_KEY,
    temperature=0
)
print("LLM ready.")

# 8. Define TTS model mapping
TTS_MODELS = {
    "ta": "facebook/mms-tts-tam",
    "hi": "facebook/mms-tts-hin",
    "te": "facebook/mms-tts-tel",
    "kn": "facebook/mms-tts-kan",
    "en": "facebook/mms-tts-eng" # Added English for completeness
}
tts_pipelines = {}

def get_tts_pipeline(lang_code):
    if lang_code not in tts_pipelines:
        model_name = TTS_MODELS.get(lang_code, "facebook/mms-tts-eng")
        print(f"Loading TTS model for {lang_code} → {model_name}")
        # Ensure that 'device' is correctly passed and handled by the pipeline if applicable
        # Some TTS models might not strictly use the device argument from pipeline if they handle it internally
        tts_pipelines[lang_code] = pipeline("text-to-speech", model=model_name, device=device)
    return tts_pipelines[lang_code]

# 9. RAG chain
print("Setting up RAG chain...")
template = """
You are a multilingual assistant. Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""

prompt = ChatPromptTemplate.from_template(template)

# The 'question_language' is not used directly by the prompt template,
# but it's part of the input to the chain.
# The prompt template focuses on using the context and question.
# The `detect` function will determine the language for TTS.
rag_chain = (
    {"context": db.as_retriever(k=5), "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
print("RAG chain ready.")

# 10. Test queries
print("\n--- Running Queries with TTS ---")
test_queries = [
    "பிரான்சின் தலைநகரம் என்ன?", # Tamil: What is the capital of France?
    "फ्रांस की राजधानी क्या है?", # Hindi: What is the capital of France?
    "ఫ్రాన్స్ రాజధాని ఏది?", # Telugu: What is the capital of France?
    "ಫ್ರಾನ್ಸ್ ರಾಜಧಾನಿ ಯಾವುದು?", # Kannada: What is the capital of France?
    "What are the primary colors?", # English (added for testing)
    "முதன்மை வண்ணங்கள் யாவை?", # Tamil: What are the primary colors?
    "प्राथमिक रंग क्या हैं?", # Hindi: What are the primary colors?
    "ప్రాథమిక రంగులు ఏమిటి?", # Telugu: What are the primary colors?
    "ಪ್ರಾಥಮಿಕ ಬಣ್ಣಗಳು ಯಾವುವು?", # Kannada: What are the primary colors?
    "பெர்லினைப் பற்றி சொல்லுங்கள்.", # Tamil: Tell me about Berlin.
    "बर्लिन के बारे में बताओ।", # Hindi: Tell me about Berlin.
    "బెర్లిన్ గురించి చెప్పండి.", # Telugu: Tell me about Berlin.
    "ಬರ್ಲಿನ್ ಬಗ್ಗೆ ಹೇಳಿ." # Kannada: Tell me about Berlin.
]

for i, query in enumerate(test_queries):
    print(f"\nQuery {i+1}: '{query}'")
    
    try:
        # Detect language of the query for TTS, before calling the LLM
        # This allows us to use the detected language for TTS even if the LLM fails.
        try:
            lang_code = detect(query)
            print(f"Detected query language: {lang_code}")
        except Exception as e:
            print(f"Could not detect language for query: {e}. Defaulting to 'en'.")
            lang_code = 'en' # Default to English if detection fails

        # Pass only 'question' to the RAG chain as per the prompt template.
        # The prompt itself ensures the LLM responds in the input language.
        response_text = rag_chain.invoke(query)
        
        print(f"LLM Response: {response_text}")
    except Exception as e:
        print(f"Failed to get response from LLM: {e}")
        continue # Skip TTS if LLM response failed

    # Get and use TTS pipeline
    try:
        tts = get_tts_pipeline(lang_code)
        audio_output = tts(response_text)
        audio_data = np.squeeze(audio_output["audio"])
        audio_path = f"response_audio_{i+1}.wav"
        sf.write(audio_path, audio_data, audio_output["sampling_rate"])
        print(f"Audio saved: {audio_path}")

        if SOUND_ENABLED:
            try:
                if audio_data.ndim == 1:
                    sd.play(audio_data, audio_output["sampling_rate"])
                    sd.wait()
                    print("Audio played.")
                else:
                    print(f"Audio shape not compatible for playback: {audio_data.shape}")
            except Exception as e:
                print(f"Error playing audio: {e}")
    except Exception as e:
        print(f"Failed to generate or play TTS audio: {e}")