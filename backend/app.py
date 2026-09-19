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
MODEL = "llama-3.3-70b-versatile"


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




def ask_vision(image_data_url, user_prompt):
    response = client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}}
                ]
            }
        ],
        max_tokens=1800
    )
    return response.choices[0].message.content

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
@app.route("/api/summarize-pdf", methods=["POST"])
def summarize_pdf():
    try:
        pdf = request.files.get("file")
        detail = request.form.get("detail", "medium")
        if not pdf:
            return jsonify({"error": "PDF file is required"}), 400
        if not pdf.filename.lower().endswith(".pdf"):
            return jsonify({"error": "Please upload a PDF file"}), 400
        raw = pdf.read()
        if len(raw) > 15 * 1024 * 1024:
            return jsonify({"error": "PDF is too large. Please keep it under 15 MB."}), 400
        reader = PdfReader(BytesIO(raw))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text.strip())
        text = "\n\n".join(pages).strip()
        if not text:
            return jsonify({"error": "No readable text was found in this PDF. Scanned/image-only PDFs are not supported yet."}), 400
        if len(text) > 50000:
            text = text[:50000]
        length_map = {
            "short": "in 5-8 concise bullet points",
            "medium": "with a short overview, key concepts, and important bullet points",
            "detailed": "as structured study notes with headings, sub-points, key terms, and exam takeaways"
        }
        prompt = f"Summarize these student notes {length_map.get(detail, length_map['medium'])}. Preserve important facts and definitions.\n\n{text}"
        summary = ask_groq("You are an academic summarization assistant. Be accurate, student-friendly, and use Markdown.", prompt, max_tokens=2500)
        return jsonify({"summary": summary, "pages": len(reader.pages)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/summary-pdf", methods=["POST"])
def summary_pdf():
    try:
        data = request.get_json() or {}
        summary = data.get("summary", "").strip()
        if not summary:
            return jsonify({"error": "Summary is required"}), 400
        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib import colors
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=42, leftMargin=42, topMargin=42, bottomMargin=42)
        styles = getSampleStyleSheet()
        body = ParagraphStyle("FutureMindBody", parent=styles["BodyText"], fontSize=10.5, leading=15, alignment=TA_LEFT, spaceAfter=8)
        story = [Paragraph("FutureMind — Study Summary", styles["Title"]), Spacer(1, 12)]
        for line in summary.splitlines():
            line = line.strip()
            if not line:
                story.append(Spacer(1, 6)); continue
            clean = re.sub(r"^#{1,6}\s*", "", line)
            clean = re.sub(r"^[-*]\s+", "• ", clean)
            clean = clean.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            story.append(Paragraph(clean, body))
        doc.build(story)
        buffer.seek(0)
        return send_file(buffer, mimetype="application/pdf", as_attachment=True, download_name="futuremind-summary.pdf")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain-diagram", methods=["POST"])
def explain_diagram():
    try:
        image = request.files.get("image")
        prompt = request.form.get("prompt", "Explain this image for a student.").strip()
        if not image:
            return jsonify({"error": "Image is required"}), 400
        allowed = {"image/png", "image/jpeg", "image/webp", "image/gif"}
        if image.mimetype not in allowed:
            return jsonify({"error": "Please upload a PNG, JPG, WEBP, or GIF image."}), 400
        raw = image.read()
        if len(raw) > 10 * 1024 * 1024:
            return jsonify({"error": "Image is too large. Please keep it under 10 MB."}), 400
        data_url = f"data:{image.mimetype};base64,{base64.b64encode(raw).decode('utf-8')}"
        instruction = (
            "You are FutureMind, a visual study tutor. Analyze the uploaded educational image. "
            "Automatically identify whether it is a flowchart, chart/graph, biology/science diagram, "
            "circuit, UML/architecture, table, math graph, or another educational visual. "
            "Do not invent unreadable values. Give a clear student-friendly response in Markdown with: "
            "1) Diagram type, 2) What it shows, 3) Key parts/data, 4) Important relationships, trends or steps, "
            "5) Simple explanation, and 6) Exam point. " + prompt
        )
        answer = ask_vision(data_url, instruction)
        return jsonify({"answer": answer, "model": VISION_MODEL})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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