# 📚 StudyMind — AI Academic Support Tool (Gemini Edition)

An AI-powered student study assistant powered by **Google Gemini**, with 5 core features:
- ✅ Answer subject doubts
- ✅ Summarize notes/text
- ✅ Generate quizzes (MCQ, True/False, Short Answer)
- ✅ Create interactive flashcards
- ✅ Build personalised study plans

---

## 📁 Project Structure

```
academic-tool/
├── backend/
│   ├── app.py            ← Flask API server (5 endpoints, Gemini-powered)
│   └── requirements.txt
└── frontend/
    └── index.html        ← Single-file UI (no build step needed)
```

---

## 🚀 Quick Start

### 1. Get a Gemini API Key

1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account
3. Click **"Create API Key"** — it's free to start!

### 2. Set up the backend

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Gemini API key
export GEMINI_API_KEY=AIza...    # Windows: set GEMINI_API_KEY=AIza...

# Start the server
python app.py
# → Running on http://localhost:5000
```

### 3. Open the frontend

Simply open `frontend/index.html` in your browser — no build or server needed.

---

## 🔌 API Endpoints

| Method | Endpoint          | Description                  |
|--------|-------------------|------------------------------|
| POST   | `/api/answer`     | Answer a subject doubt       |
| POST   | `/api/summarize`  | Summarize notes/text         |
| POST   | `/api/quiz`       | Generate a quiz              |
| POST   | `/api/flashcards` | Create flashcards            |
| POST   | `/api/studyplan`  | Generate a study plan        |
| GET    | `/api/health`     | Health check                 |

---

## 🛠 Tech Stack

| Layer    | Technology                          |
|----------|-------------------------------------|
| Backend  | Python 3.10+, Flask, flask-cors     |
| AI       | Google Gemini 2.0 Flash             |
| Frontend | Vanilla HTML/CSS/JS (zero deps)     |

---

## 💡 Tips

- **Ctrl + Enter** submits the current form.
- Flashcards are interactive — click to flip!
- The "Backend URL" field in the sidebar lets you point the frontend at any server.
