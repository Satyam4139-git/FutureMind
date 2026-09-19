from flask import Flask, request, jsonify
from flask_cors import CORS
from groq import Groq
from dotenv import load_dotenv
import os
import json
import re

# Load environment variables
load_dotenv()

app = Flask(__name__)
CORS(app)

# Load API key from .env
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found. Please set it in the .env file.")

# Initialize Groq client
client = Groq(api_key=api_key)
MODEL = "openai/gpt-oss-120b"


# -----------------------------
# Utility Functions
# -----------------------------
def ask_groq(system_prompt, user_message, max_tokens=1500):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error while calling Groq API: {str(e)}"


def parse_json_response(raw):
    """
    Cleans AI response and converts it into valid JSON.
    Removes markdown code blocks if present.
    """
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()
    return json.loads(clean)


# -----------------------------
# Root Route
# -----------------------------
@app.route("/", methods=["GET"])
def home():
    return "Academic Support Tool Backend is Live!"


# -----------------------------
# Health Check Route
# -----------------------------
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "message": "Backend is running successfully"
    })


# -----------------------------
# 1. Answer Subject Doubts
# -----------------------------
@app.route("/api/answer", methods=["POST"])
def answer_doubt():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        subject = data.get("subject", "General").strip()

        if not question:
            return jsonify({"error": "Question is required"}), 400

        system = (
            f"You are an expert tutor in {subject}. "
            "Give clear, concise, well-structured answers suitable for students. "
            "Use examples, analogies, and step-by-step explanations when helpful. "
            "Format using Markdown."
        )

        answer = ask_groq(system, question)
        return jsonify({"answer": answer})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 2. Summarize Notes
# -----------------------------
@app.route("/api/summarize", methods=["POST"])
def summarize_notes():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        detail = data.get("detail", "medium")

        if not text:
            return jsonify({"error": "Text is required"}), 400

        length_map = {
            "short": "in 3-5 bullet points",
            "medium": "in 1-2 paragraphs with key bullet points",
            "detailed": "in a structured outline with headings, sub-points, and a key-takeaways section"
        }

        system = (
            "You are an academic summarization assistant. "
            "Summarize clearly and accurately. Format using Markdown."
        )

        prompt = f"Summarize the following text {length_map.get(detail, length_map['medium'])}:\n\n{text}"
        summary = ask_groq(system, prompt, max_tokens=2000)

        return jsonify({"summary": summary})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 3. Generate Quiz
# -----------------------------
@app.route("/api/quiz", methods=["POST"])
def generate_quiz():
    try:
        data = request.get_json()
        topic = data.get("topic", "").strip()
        num_q = min(int(data.get("num_questions", 5)), 10)
        q_type = data.get("type", "mcq")

        if not topic:
            return jsonify({"error": "Topic is required"}), 400

        type_map = {
            "mcq": "multiple-choice questions (4 options each, mark correct answer clearly)",
            "truefalse": "true/false questions (state the answer after each)",
            "short": "short-answer questions (include a model answer)"
        }

        system = (
            "You are a quiz generator for academic students. "
            "Return ONLY valid JSON — no markdown fences, no preamble. "
            'Schema: {"quiz": [{"question": str, "options": [str] or null, "answer": str}]}'
        )

        prompt = f"Generate {num_q} {type_map.get(q_type, type_map['mcq'])} on the topic: {topic}"
        raw = ask_groq(system, prompt, max_tokens=2000)

        try:
            parsed = parse_json_response(raw)
            return jsonify(parsed)
        except Exception:
            return jsonify({
                "error": "Failed to parse AI response into JSON",
                "raw": raw
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 4. Make Flashcards
# -----------------------------
@app.route("/api/flashcards", methods=["POST"])
def make_flashcards():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        topic = data.get("topic", "").strip()
        count = min(int(data.get("count", 8)), 20)

        source = text or topic
        if not source:
            return jsonify({"error": "Text or topic is required"}), 400

        system = (
            "You are a flashcard creator for students. "
            "Return ONLY valid JSON — no markdown fences, no preamble. "
            'Schema: {"flashcards": [{"front": str, "back": str}]}'
        )

        prompt = (
            f"Create {count} study flashcards from:\n\n{source}"
            if text else
            f"Create {count} study flashcards on: {topic}"
        )

        raw = ask_groq(system, prompt, max_tokens=2000)

        try:
            parsed = parse_json_response(raw)
            return jsonify(parsed)
        except Exception:
            return jsonify({
                "error": "Failed to parse AI response into JSON",
                "raw": raw
            }), 500

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# 5. Generate Study Plan
# -----------------------------
@app.route("/api/studyplan", methods=["POST"])
def study_plan():
    try:
        data = request.get_json()
        subject = data.get("subject", "").strip()
        days = min(int(data.get("days", 7)), 30)
        goal = data.get("goal", "master the subject").strip()
        level = data.get("level", "intermediate").strip()

        if not subject:
            return jsonify({"error": "Subject is required"}), 400

        system = (
            "You are an academic coach. Format the plan using Markdown with a clear "
            "day-by-day schedule, daily time estimates, topics, resources, and revision tips."
        )

        prompt = (
            f"Create a {days}-day study plan for a {level} student "
            f"who wants to {goal} in {subject}. "
            "Include daily tasks, time allocations, and revision tips."
        )

        plan = ask_groq(system, prompt, max_tokens=2500)
        return jsonify({"plan": plan})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# -----------------------------
# Run App (Render Deployment Fix)
# -----------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
