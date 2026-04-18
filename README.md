# 🎙️ AI Voice Trivia

An interactive **AI-powered voice-based trivia application** built using Flask, Speech Recognition, and Text-to-Speech technologies. This project allows users to engage in a fun quiz experience using **voice input and audio responses**, creating a seamless human-computer interaction.

---

## 🚀 Features

* 🎤 **Voice Input Recognition** – Capture and process user speech
* 🧠 **AI-Based Question Handling** – Smart response processing
* 🔊 **Text-to-Speech Output** – Audio responses using TTS
* 🌐 **Web Interface** – Simple and interactive frontend
* 📂 **Audio Processing** – Handles `.wav` and `.webm` formats
* ⚡ **Fast API Backend** – Powered by Flask

---

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Frontend:** HTML, CSS, JavaScript
* **Libraries Used:**

  * `speech_recognition`
  * `gTTS`
  * `transformers`
  * `numpy`
  * `scipy`
* **Other Tools:** Flask-CORS

Voice-based applications typically rely on combining **speech recognition and text-to-speech pipelines** to create interactive systems ([GitHub][1]).

---

## 📁 Project Structure

```
AI-Voice-Trivia/
│── app.py              # Main Flask backend
│── main.py             # Core logic / processing
│── index.html          # Frontend UI
│── input.wav           # Recorded input audio
│── input_raw.webm      # Raw browser audio
│── response.mp3        # Generated response audio
│── train-v2.0.json     # Dataset / questions
│── venv/               # Virtual environment (ignored)
```

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Sanchit-Narayan/AI-Voice-Trivia.git
cd AI-Voice-Trivia
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

*(If requirements.txt is not present, install manually:)*

```bash
pip install flask flask-cors numpy scipy speechrecognition gtts transformers
```

---

## ▶️ Running the Application

```bash
python app.py
```

Open your browser and go to:

```
http://localhost:5000
```

---

## 🎯 How It Works

1. User speaks into the microphone
2. Audio is recorded and sent to backend
3. Speech is converted to text
4. AI processes the query / trivia logic
5. Response is generated
6. Text is converted back to speech
7. Audio response is played to the user

---

## 📌 Future Improvements

* 🧩 Add more advanced NLP models
* 🌍 Multi-language support
* 📊 Score tracking and leaderboard
* 📱 Mobile responsiveness
* ☁️ Deployment on cloud (AWS / Render)

---
