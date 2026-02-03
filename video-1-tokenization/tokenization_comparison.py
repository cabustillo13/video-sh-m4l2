import soundfile as sf
from transformers import pipeline
from jiwer import wer


# Paths and models
AUDIO_PATH = "../audios/tokenization_demo.wav"
MODEL_A_NAME = "openai/whisper-small"
MODEL_B_NAME = "facebook/wav2vec2-base-960h"

# Load audio
audio, sr = sf.read(AUDIO_PATH)
print(f"Sample rate: {sr} Hz")

# Initialize ASR pipelines on CPU
asr_a = pipeline("automatic-speech-recognition", model=MODEL_A_NAME, device="cpu")
asr_b = pipeline("automatic-speech-recognition", model=MODEL_B_NAME, device="cpu")

# Transcribe audio
text_a = asr_a(audio)["text"]
text_b = asr_b(audio)["text"]

# Compute WER between outputs
wer_value = wer(text_a, text_b)

# Print results
print("\nModel A (Whisper):\n", text_a)
print("\nModel B (Wav2Vec2):\n", text_b)
print(f"\nWER: {wer_value:.4f}")

# Token counts
tokens_a = len(text_a.split())
tokens_b = len(text_b.split())

print(f"\nTokens A: {tokens_a}")
print(f"Tokens B: {tokens_b}")
