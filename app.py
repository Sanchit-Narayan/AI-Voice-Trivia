from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import numpy as np
import scipy.io.wavfile as wav
import speech_recognition as sr
from transformers import pipeline
from gtts import gTTS
import json
import os
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
CORS(app, origins=["http://127.0.0.1:5000", "http://localhost:5000"])

# ── Load Models & Data ────────────────────────────────────────────────────────

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def load_squad_data(file):
    with open(file, 'r') as f:
        data = json.load(f)
    contexts = []
    for article in data['data']:
        for paragraph in article['paragraphs']:
            contexts.append(paragraph['context'])
    return contexts

print("Loading SQuAD dataset...")
squad_contexts = load_squad_data("train-v2.0.json")  # Remove [:2000] for full dataset

print(f"Loaded {len(squad_contexts)} contexts. Computing embeddings...")
context_embeddings = embedder.encode(squad_contexts, convert_to_tensor=True)
print("Embeddings ready!")

print("Loading Flan-T5 model...")
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")
print("All models ready!")

recognizer = sr.Recognizer()

# ── Helper Functions ──────────────────────────────────────────────────────────

def get_context(question):
    question_embedding = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(question_embedding, context_embeddings)[0]
    best_idx = scores.argmax().item()
    best_score = scores[best_idx].item()
    return squad_contexts[best_idx], best_score

def answer_question(question, context):
    prompt = (
        f"Read the context carefully and answer the question.\n\n"
        f"Context: {context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )
    result = qa_pipeline(prompt, max_new_tokens=100, min_length=5, do_sample=False)
    return result[0]['generated_text'].strip()

# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/ask', methods=['POST'])
def ask():
    """Accept a typed question and return answer + context info"""
    data = request.get_json()
    question = data.get('question', '').strip()

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    context, score = get_context(question)
    answer = answer_question(question, context)

    # Generate TTS
    tts = gTTS(text=answer, lang='en')
    tts.save("response.mp3")

    return jsonify({
        'question': question,
        'answer': answer,
        'confidence': round(score, 4),
        'context_snippet': context[:300] + '...',
        'low_confidence': score < 0.3
    })

@app.route('/ask-audio', methods=['POST'])
def ask_audio():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file provided'}), 400

    audio_file = request.files['audio']
    audio_file.save("input_raw.webm")

    # Convert webm to wav using ffmpeg
    os.system("ffmpeg -y -i input_raw.webm input.wav")

    # Transcribe
    with sr.AudioFile("input.wav") as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        audio = recognizer.record(source)

    try:
        question = recognizer.recognize_google(audio)
    except sr.UnknownValueError:
        return jsonify({'error': 'Could not understand audio. Please speak clearly and try again.'}), 400
    except sr.RequestError as e:
        return jsonify({'error': f'Speech recognition service error: {e}'}), 500

    context, score = get_context(question)
    answer = answer_question(question, context)

    tts = gTTS(text=answer, lang='en')
    tts.save("response.mp3")

    return jsonify({
        'question': question,
        'answer': answer,
        'confidence': round(score, 4),
        'context_snippet': context[:300] + '...',
        'low_confidence': score < 0.3
    })

@app.route('/audio-response', methods=['GET'])
def audio_response():
    """Serve the generated TTS mp3"""
    return send_file("response.mp3", mimetype="audio/mpeg")

@app.route('/')
def home():
    return send_file('index.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

if __name__ == '__main__':
    app.run(debug=False, port=5000)