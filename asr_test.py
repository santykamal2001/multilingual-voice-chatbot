import torch
from transformers import pipeline
import soundfile as sf
import os

# 1. Define the device to use (MPS for your M2 GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device for ASR: {device}")

# 2. Load the Whisper ASR pipeline
# We'll start with a smaller model for quicker downloads and testing.
# You can change to "openai/whisper-base" or "openai/whisper-medium" later if needed.
# The 'device' argument pushes the model to your M2's GPU.
asr_pipeline = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=device)
print(f"Whisper-small model loaded on {device}.")

# 3. Prepare a test audio file
# For a quick test, let's create a dummy WAV file.
# In a real scenario, you'd have actual audio inputs.
sample_rate = 16000 # Whisper models typically expect 16kHz audio
duration = 3        # seconds
frequency = 440     # Hz (A4 note)
t = torch.linspace(0, duration, int(sample_rate * duration), device=device)
audio_data = torch.sin(2 * torch.pi * frequency * t).cpu().numpy() # Generate a sine wave, move to CPU for soundfile

test_audio_path = "test_audio.wav"
sf.write(test_audio_path, audio_data, sample_rate)
print(f"Generated a test audio file: {test_audio_path}")

# 4. Transcribe the audio
print("Transcribing audio...")
transcription = asr_pipeline(test_audio_path)

# 5. Print the result
print("\n--- ASR Transcription Result ---")
print(f"Transcribed Text: {transcription['text']}")

# 6. Clean up the test audio file
os.remove(test_audio_path)
print(f"Removed temporary audio file: {test_audio_path}")