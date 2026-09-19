from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from groq import Groq
from openai import OpenAI
from huggingface_hub import InferenceClient
from dotenv import load_dotenv
from io import BytesIO
from pypdf import PdfReader
import os
import json
import re
import base64

# ============================================================
# LOAD ENVIRONMENT VARIABLES
# ============================================================

load_dotenv()

app = Flask(__name__)
CORS(app)

# ============================================================
# API KEYS
# ============================================================

groq_api_key = os.getenv("GROQ_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
hf_token = os.getenv("HF_TOKEN")

if not groq_api_key:
    raise ValueError(
        "GROQ_API_KEY not found. Please set it in the .env file."
    )

if not openai_api_key:
    raise ValueError(
        "OPENAI_API_KEY not found. Please set it in the .env file."
    )

# ============================================================
# API CLIENTS
# ============================================================

groq_client = Groq(api_key=groq_api_key)
openai_client = OpenAI(api_key=openai_api_key)

hf_client = (
    InferenceClient(
        api_key=hf_token,
        provider="auto"
    )
    if hf_token
    else None
)

# ============================================================
# MODELS
# ============================================================

MODEL = "openai/gpt-oss-120b"
VISION_MODEL = "qwen/qwen3.6-27b"
IMAGE_MODEL = "gpt-image-2"
HF_IMAGE_MODEL = "black-forest-labs/FLUX.1-schnell"


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def ask_groq(system_prompt, user_message, max_tokens=1500):
    """
    Send a normal text request to Groq.
    """
    try:
        response = groq_client.chat.completions.create(
            model=MODEL,
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_message
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return f"Error while calling Groq API: {str(e)}"


def ask_vision(image_data_url, user_prompt):
    """
    Send an image + prompt to the vision model.
    """
    response = groq_client.chat.completions.create(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": user_prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url
                        }
                    }
                ]
            }
        ],
        max_tokens=1800
    )

    return response.choices[0].message.content


def parse_json_response(raw):
    """
    Clean AI response and convert it into valid JSON.
    Removes markdown code blocks if present.
    """

    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    return json.loads(clean)


# ============================================================
# ROOT ROUTE
# ============================================================

@app.route("/", methods=["GET"])
def home():
    return "FutureMind Academic Support Backend is Live!"


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL,
        "vision_model": VISION_MODEL,
        "image_model": IMAGE_MODEL,
        "message": "Backend is running successfully"
    })


# ============================================================
# 1. ASK FUTUREMIND / SUBJECT DOUBTS
# ============================================================

@app.route("/api/answer", methods=["POST"])
def answer_doubt():

    try:
        data = request.get_json() or {}

        question = data.get("question", "").strip()
        subject = data.get("subject", "General").strip()

        if not question:
            return jsonify({
                "error": "Question is required"
            }), 400

        system = (
            f"You are an expert tutor in {subject}. "
            "Give clear, concise, well-structured answers suitable "
            "for students. Use examples, analogies, and step-by-step "
            "explanations when helpful. Format using Markdown."
        )

        answer = ask_groq(
            system,
            question,
            max_tokens=1500
        )

        return jsonify({
            "answer": answer
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 2. SUMMARIZE TEXT / NOTES
# ============================================================

@app.route("/api/summarize", methods=["POST"])
def summarize_notes():

    try:
        data = request.get_json() or {}

        text = data.get("text", "").strip()
        detail = data.get("detail", "medium")

        if not text:
            return jsonify({
                "error": "Text is required"
            }), 400

        length_map = {
            "short":
                "in 3-5 bullet points",

            "medium":
                "in 1-2 paragraphs with key bullet points",

            "detailed":
                "in a structured outline with headings, "
                "sub-points, and a key-takeaways section"
        }

        system = (
            "You are an academic summarization assistant. "
            "Summarize clearly and accurately. "
            "Format using Markdown."
        )

        prompt = (
            f"Summarize the following text "
            f"{length_map.get(detail, length_map['medium'])}:\n\n"
            f"{text}"
        )

        summary = ask_groq(
            system,
            prompt,
            max_tokens=2000
        )

        return jsonify({
            "summary": summary
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 3. PDF → SUMMARY
# ============================================================

@app.route("/api/summarize-pdf", methods=["POST"])
def summarize_pdf():

    try:

        pdf = request.files.get("file")
        detail = request.form.get("detail", "medium")

        if not pdf:
            return jsonify({
                "error": "PDF file is required"
            }), 400

        if not pdf.filename.lower().endswith(".pdf"):
            return jsonify({
                "error": "Please upload a PDF file"
            }), 400

        raw = pdf.read()

        if len(raw) > 15 * 1024 * 1024:
            return jsonify({
                "error": "PDF is too large. Please keep it under 15 MB."
            }), 400

        reader = PdfReader(BytesIO(raw))

        pages = []

        for page in reader.pages:

            text = page.extract_text() or ""

            if text.strip():
                pages.append(text.strip())

        text = "\n\n".join(pages).strip()

        if not text:
            return jsonify({
                "error":
                    "No readable text was found in this PDF. "
                    "Scanned/image-only PDFs are not supported yet."
            }), 400

        if len(text) > 50000:
            text = text[:50000]

        length_map = {

            "short":
                "in 5-8 concise bullet points",

            "medium":
                "with a short overview, key concepts, "
                "and important bullet points",

            "detailed":
                "as structured study notes with headings, "
                "sub-points, key terms, and exam takeaways"
        }

        prompt = (
            f"Summarize these student notes "
            f"{length_map.get(detail, length_map['medium'])}. "
            "Preserve important facts and definitions.\n\n"
            f"{text}"
        )

        summary = ask_groq(
            "You are an academic summarization assistant. "
            "Be accurate, student-friendly, and use Markdown.",
            prompt,
            max_tokens=2500
        )

        return jsonify({
            "summary": summary,
            "pages": len(reader.pages)
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 4. SUMMARY → DOWNLOADABLE PDF
# ============================================================

@app.route("/api/summary-pdf", methods=["POST"])
def summary_pdf():

    try:

        data = request.get_json() or {}

        summary = data.get("summary", "").strip()

        if not summary:
            return jsonify({
                "error": "Summary is required"
            }), 400

        from reportlab.lib.pagesizes import A4
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer
        )
        from reportlab.lib.styles import (
            getSampleStyleSheet,
            ParagraphStyle
        )
        from reportlab.lib.enums import TA_LEFT

        buffer = BytesIO()

        doc = SimpleDocTemplate(
            buffer,
            pagesize=A4,
            rightMargin=42,
            leftMargin=42,
            topMargin=42,
            bottomMargin=42
        )

        styles = getSampleStyleSheet()

        body = ParagraphStyle(
            "FutureMindBody",
            parent=styles["BodyText"],
            fontSize=10.5,
            leading=15,
            alignment=TA_LEFT,
            spaceAfter=8
        )

        story = [
            Paragraph(
                "FutureMind - Study Summary",
                styles["Title"]
            ),
            Spacer(1, 12)
        ]

        for line in summary.splitlines():

            line = line.strip()

            if not line:
                story.append(Spacer(1, 6))
                continue

            clean = re.sub(
                r"^#{1,6}\s*",
                "",
                line
            )

            clean = re.sub(
                r"^[-*]\s+",
                "• ",
                clean
            )

            clean = (
                clean
                .replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
            )

            story.append(
                Paragraph(clean, body)
            )

        doc.build(story)

        buffer.seek(0)

        return send_file(
            buffer,
            mimetype="application/pdf",
            as_attachment=True,
            download_name="futuremind-summary.pdf"
        )

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 5. UPLOAD & EXPLAIN DIAGRAM
# ============================================================

@app.route("/api/explain-diagram", methods=["POST"])
def explain_diagram():

    try:

        image = request.files.get("image")

        prompt = request.form.get(
            "prompt",
            "Explain this image for a student."
        ).strip()

        if not image:
            return jsonify({
                "error": "Image is required"
            }), 400

        allowed = {
            "image/png",
            "image/jpeg",
            "image/webp",
            "image/gif"
        }

        if image.mimetype not in allowed:
            return jsonify({
                "error":
                    "Please upload a PNG, JPG, WEBP, or GIF image."
            }), 400

        raw = image.read()

        if len(raw) > 10 * 1024 * 1024:
            return jsonify({
                "error":
                    "Image is too large. Please keep it under 10 MB."
            }), 400

        data_url = (
            f"data:{image.mimetype};base64,"
            f"{base64.b64encode(raw).decode('utf-8')}"
        )

        instruction = (
            "You are FutureMind, a visual study tutor. "
            "Analyze the uploaded educational image. "

            "Automatically identify whether it is a "
            "flowchart, chart/graph, biology/science diagram, "
            "circuit, UML/architecture, table, math graph, "
            "or another educational visual. "

            "Do not invent unreadable values. "

            "Give a clear student-friendly response in Markdown "
            "with the following sections:\n\n"

            "1) Diagram type\n"
            "2) What it shows\n"
            "3) Key parts/data\n"
            "4) Important relationships, trends or steps\n"
            "5) Simple explanation\n"
            "6) Exam point\n\n"

            + prompt
        )

        answer = ask_vision(
            data_url,
            instruction
        )

        return jsonify({
            "answer": answer,
            "model": VISION_MODEL
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 6. GENERATE EDUCATIONAL DIAGRAM / IMAGE
# ============================================================

@app.route("/api/generate-image", methods=["POST"])
def generate_image():

    try:

        data = request.get_json() or {}

        prompt = data.get("prompt", "").strip()

        if not prompt:
            return jsonify({
                "error": "Image description is required"
            }), 400

        educational_prompt = f"""
Create a high-quality educational visual for FutureMind,
an academic study platform.

Student request:
{prompt}

IMPORTANT DESIGN REQUIREMENTS:

1. Make the visual educational and easy to understand.

2. Show the requested subject prominently.

3. If the request is a scientific, biological,
anatomical, engineering, mathematical, or academic
diagram, prioritize a clear educational representation.

4. Use clear labels when labels are appropriate.

5. Keep the layout organized and suitable for students.

6. Use a clean modern academic infographic style.

7. Avoid unnecessary decorative elements.

8. Do not add fake scientific data.

9. Do not invent measurements, statistics, or values
unless the user explicitly provides them.

10. If the request is a process, clearly show
the process and relationships between the parts.

11. If the request is an anatomical diagram, show
the relevant anatomical structures clearly and label
important parts.

12. If the request is a chart or graph, represent
the supplied values accurately.

13. Make text and labels readable.

14. The result should look like something a student
could actually use for studying or exam preparation.

15. Prefer a clean horizontal educational composition
when the subject benefits from an explanatory panel.

16. For complex academic topics, prefer a layout where
the main diagram or visual is on the left and concise
educational explanations are on the right.

17. Include useful sections such as:
- What is it?
- Main parts
- Key functions
- Important relationships
when appropriate to the subject.

18. Add a concise educational takeaway at the bottom
when appropriate.

19. Do not include a fake FutureMind logo or fake UI
elements unless specifically requested.

20. Create the actual image, not a written description.
"""

        # ----------------------------------------------------
        # PRIMARY: OPENAI IMAGE GENERATION
        # ----------------------------------------------------

        try:

            response = openai_client.images.generate(
                model=IMAGE_MODEL,
                prompt=educational_prompt,
                size="1536x1024"
            )

            if response.data:

                image_data = response.data[0].b64_json

                if image_data:

                    return jsonify({
                        "image": f"data:image/png;base64,{image_data}",
                        "model": IMAGE_MODEL,
                        "provider": "OpenAI"
                    })

        except Exception as openai_error:

            openai_error_text = str(openai_error).lower()

            # Only use Hugging Face automatically when
            # OpenAI has no available credit/quota.

            quota_error = (
                "429" in openai_error_text
                or "insufficient_quota" in openai_error_text
                or "credit_balance_exhausted" in openai_error_text
                or "quota" in openai_error_text
                or "credits" in openai_error_text
            )

            if not quota_error:

                return jsonify({
                    "error":
                        f"OpenAI image generation failed: "
                        f"{str(openai_error)}"
                }), 500

            print(
                "OpenAI image quota unavailable. "
                "Switching to Hugging Face..."
            )

        # ----------------------------------------------------
        # FALLBACK: HUGGING FACE
        # ----------------------------------------------------

        if not hf_client:

            return jsonify({
                "error":
                    "OpenAI image credits are unavailable and "
                    "Hugging Face is not configured."
            }), 500

        try:

            hf_image = hf_client.text_to_image(
                educational_prompt,
                model=HF_IMAGE_MODEL
            )

            # Convert the generated PIL image into PNG bytes
            image_buffer = BytesIO()

            hf_image.save(
                image_buffer,
                format="PNG"
            )

            image_buffer.seek(0)

            hf_image_data = base64.b64encode(
                image_buffer.read()
            ).decode("utf-8")

            return jsonify({
                "image":
                    f"data:image/png;base64,{hf_image_data}",
                "model": HF_IMAGE_MODEL,
                "provider": "Hugging Face"
            })

        except Exception as hf_error:

            return jsonify({
                "error":
                    "Both image generation services failed. "
                    f"Hugging Face error: {str(hf_error)}"
            }), 500

    except Exception as e:

        return jsonify({
            "error":
                f"Image generation failed: {str(e)}"
        }), 500# ============================================================
# 7. GENERATE QUIZ
# ============================================================

@app.route("/api/quiz", methods=["POST"])
def generate_quiz():

    try:

        data = request.get_json() or {}

        topic = data.get("topic", "").strip()

        num_q = min(
            int(data.get("num_questions", 5)),
            10
        )

        q_type = data.get(
            "type",
            "mcq"
        )

        if not topic:
            return jsonify({
                "error": "Topic is required"
            }), 400

        type_map = {

            "mcq":
                "multiple-choice questions "
                "(4 options each, mark correct answer clearly)",

            "truefalse":
                "true/false questions "
                "(state the answer after each)",

            "short":
                "short-answer questions "
                "(include a model answer)"
        }

        system = (
            "You are a quiz generator for academic students. "

            "Return ONLY valid JSON - no markdown fences, "
            "no preamble. "

            'Schema: {"quiz": [{"question": str, '
            '"options": [str] or null, "answer": str}]}'
        )

        prompt = (
            f"Generate {num_q} "
            f"{type_map.get(q_type, type_map['mcq'])} "
            f"on the topic: {topic}"
        )

        raw = ask_groq(
            system,
            prompt,
            max_tokens=2000
        )

        try:

            parsed = parse_json_response(raw)

            return jsonify(parsed)

        except Exception:

            return jsonify({
                "error":
                    "Failed to parse AI response into JSON",
                "raw": raw
            }), 500

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 8. MAKE FLASHCARDS
# ============================================================

@app.route("/api/flashcards", methods=["POST"])
def make_flashcards():

    try:

        data = request.get_json() or {}

        text = data.get(
            "text",
            ""
        ).strip()

        topic = data.get(
            "topic",
            ""
        ).strip()

        count = min(
            int(data.get("count", 8)),
            20
        )

        source = text or topic

        if not source:
            return jsonify({
                "error":
                    "Text or topic is required"
            }), 400

        system = (
            "You are a flashcard creator for students. "

            "Return ONLY valid JSON - no markdown fences, "
            "no preamble. "

            'Schema: {"flashcards": '
            '[{"front": str, "back": str}]}'
        )

        if text:

            prompt = (
                f"Create {count} study flashcards from:"
                f"\n\n{text}"
            )

        else:

            prompt = (
                f"Create {count} study flashcards on:"
                f" {topic}"
            )

        raw = ask_groq(
            system,
            prompt,
            max_tokens=2000
        )

        try:

            parsed = parse_json_response(raw)

            return jsonify(parsed)

        except Exception:

            return jsonify({
                "error":
                    "Failed to parse AI response into JSON",
                "raw": raw
            }), 500

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# 9. GENERATE STUDY PLAN
# ============================================================

@app.route("/api/studyplan", methods=["POST"])
def study_plan():

    try:

        data = request.get_json() or {}

        subject = data.get(
            "subject",
            ""
        ).strip()

        days = min(
            int(data.get("days", 7)),
            30
        )

        goal = data.get(
            "goal",
            "master the subject"
        ).strip()

        level = data.get(
            "level",
            "intermediate"
        ).strip()

        if not subject:
            return jsonify({
                "error":
                    "Subject is required"
            }), 400

        system = (
            "You are an academic coach. "

            "Format the plan using Markdown with a clear "
            "day-by-day schedule, daily time estimates, "
            "topics, resources, and revision tips."
        )

        prompt = (
            f"Create a {days}-day study plan for a "
            f"{level} student who wants to {goal} "
            f"in {subject}. "

            "Include daily tasks, time allocations, "
            "and revision tips."
        )

        plan = ask_groq(
            system,
            prompt,
            max_tokens=2500
        )

        return jsonify({
            "plan": plan
        })

    except Exception as e:

        return jsonify({
            "error": str(e)
        }), 500


# ============================================================
# RUN APP
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port
    )