import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import os
import json
import time
from sentence_transformers import SentenceTransformer, util

# Load embedding model
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load SQuAD dataset
def load_squad_data(file):
    with open(file, 'r') as f:
        data = json.load(f)

    contexts = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            contexts.append(paragraph['context'])

    return contexts

# Load ALL contexts for better retrieval coverage
squad_contexts = load_squad_data("train-v2.0.json")

print(f"Loaded {len(squad_contexts)} contexts from SQuAD dataset")

# Precompute embeddings (this will take a minute on first run)
print("Computing embeddings... please wait...")
context_embeddings = embedder.encode(squad_contexts, convert_to_tensor=True)
print("Embeddings ready!")

# Initialize recognizer
recognizer = sr.Recognizer()

# Switch to generative model — still uses SQuAD context
qa_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base"
)

# Record audio
def record_audio(filename="input.wav", duration=6, fs=44100):
    print("\nGet ready...")
    time.sleep(2)

    print("Speak now...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32')
    sd.wait()

    recording_int16 = np.int16(recording * 32767)
    wav.write(filename, fs, recording_int16)

    print("Recording complete!")

# Speech to text
def speech_to_text(file):
    with sr.AudioFile(file) as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio)
        print(f"Recognized: {text}")
        return text
    except sr.UnknownValueError:
        print("Could not understand audio")
        return None
    except sr.RequestError as e:
        print(f"Google API error: {e}")
        return None

# Retrieve best single context with confidence score
def get_context(question):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, context_embeddings)[0]

    # Get top 1 best context only
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()

    return squad_contexts[best_idx], best_score

# Answer using generative model with SQuAD context
def answer_question(question, context):
    prompt = (
        f"Read the context carefully and answer the question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )

    result = qa_pipeline(
        prompt,
        max_length=100,
        min_length=5,
        do_sample=False       # Deterministic output
    )

    return result[0]['generated_text'].strip()

# ── MAIN FLOW ──────────────────────────────────────────────────────────────────

record_audio()
question = speech_to_text("input.wav")

# Fallback to typed input if speech fails
if not question:
    question = input("Couldn't catch that. Type your question: ")

print(f"\nYou asked: {question}")

# Retrieve best matching SQuAD context
context, score = get_context(question)

print(f"\n[DEBUG] Retrieval confidence score: {score:.4f}")
print(f"[DEBUG] Selected context snippet:\n{context[:300]}...\n")

# Confidence threshold — warn if retrieval is weak
if score < 0.3:
    print("[WARNING] Low confidence context match — answer may not be accurate.\n")

# Generate answer using SQuAD context
answer = answer_question(question, context)

print(f"Answer: {answer}")

# Text-to-Speech output
tts = gTTS(text=answer, lang='en')
tts.save("response.mp3")

# Play audio — change to 'mpg321 response.mp3' on Linux or 'start response.mp3' on Windows
os.system("afplay response.mp3")