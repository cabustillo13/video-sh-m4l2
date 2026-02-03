import soundfile as sf
from transformers import pipeline
from jiwer import wer

# Paths and models
AUDIO_PATH = "../audios/tokenization_demo.wav"
MODEL_A_NAME = "openai/whisper-small"
MODEL_B_NAME = "facebook/wav2vec2-large-xlsr-53-spanish"

# Load audio
audio, sr = sf.read(AUDIO_PATH)
print(f"Sample rate: {sr} Hz")
print(f"Audio duration: {len(audio)/sr:.2f} seconds")

# Initialize ASR pipelines on CPU
# Whisper in spanish
asr_a = pipeline(
    "automatic-speech-recognition", 
    model=MODEL_A_NAME,
    device="cpu",
    generate_kwargs={
        "language": "spanish",
        "task": "transcribe"
    }
)

# Wav2Vec2 in spanish
asr_b = pipeline(
    "automatic-speech-recognition", 
    model=MODEL_B_NAME,
    device="cpu"
)

# Transcribe audio
print("\nTranscribe with Whisper...")
text_a = asr_a(audio)["text"]

print("Transcribe with Wav2Vec2...")
text_b = asr_b(audio)["text"]

# Normalize texts for WER calculation
text_a_norm = text_a.lower().strip()
text_b_norm = text_b.lower().strip()

# Compute WER between outputs
wer_value = wer(text_a_norm, text_b_norm)

# Print results
print("\nModel A (Whisper):\n", text_a)
print("\nModel B (Wav2Vec2):\n", text_b)
print(f"\nWER: {wer_value:.4f}")

# Token counts
tokens_a = len(text_a.split())
tokens_b = len(text_b.split())

print(f"\nTokens A: {tokens_a}")
print(f"Tokens B: {tokens_b}")
