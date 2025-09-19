import os
import uuid
import torch
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from langdetect import detect
from transformers import pipeline

# Optional: For playing audio
try:
    import sounddevice as sd
    SOUND_ENABLED = True
except ImportError:
    SOUND_ENABLED = False

# LangChain core
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.output_parsers import BaseOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langcodes import Language

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

if not GEMINI_API_KEY:
    print(" Error: GEMINI_API_KEY not found in .env")
    exit()
if not GOOGLE_CSE_ID:
    print(" Error: GOOGLE_CSE_ID not found in .env")
    exit()

# Set device
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f" Using device: {device}")

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

# === Initialize LLM ===
print(f" Initializing Gemini LLM...")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",
    google_api_key=GEMINI_API_KEY,
    temperature=0.2,
)
print(" LLM ready.")

# === Google Search Tool ===
print(" Setting up Google Search Tool...")
search = GoogleSearchAPIWrapper(google_api_key=GEMINI_API_KEY, google_cse_id=GOOGLE_CSE_ID)
tools = [
    Tool(
        name="Google Search",
        func=search.run,
        description="Use for questions about current events or real-time information. Input should be a search query."
    )
]
print(" Google Search Tool ready.")

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

# === ReAct Agent Prompt ===
print("🤖 Setting up Agent with ReAct...")

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
print(" Agent ready.")

# === Chat Loop ===
print("\n--- Ask me anything (type 'exit' to quit) ---")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print(" Goodbye!")
        break

    # Detect language and prepare multilingual-aware prompt
    try:
        input_lang_code = detect(user_input)
        lang_obj = Language.get(input_lang_code)
        input_lang_name = lang_obj.display_name().capitalize()
    except:
        input_lang_code = "en"
        input_lang_name = "English"

    modified_input = f"Please answer the following question in {input_lang_name}:\n{user_input}"

    try:
        response = agent_executor.invoke({"input": modified_input})
        answer = response.get("output", "⚠️ Sorry, I couldn't generate an answer.").strip()
        print(f"\n🧠 Agent ({input_lang_name}): {answer}")

        # Truncate if needed for TTS stability
        answer = answer[:500]

        tts = get_tts_pipeline(input_lang_code)
        audio_output = tts(answer)
        audio_data = np.squeeze(audio_output["audio"])
        audio_path = f"response_audio_{uuid.uuid4().hex}.wav"
        sf.write(audio_path, audio_data, audio_output["sampling_rate"])
        print(f"🔉 Audio saved to {audio_path}")

        if SOUND_ENABLED:
            sd.play(audio_data, audio_output["sampling_rate"])
            sd.wait()

    except Exception as e:
        print(f" Error: {e}")
