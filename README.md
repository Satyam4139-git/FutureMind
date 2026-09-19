## 1. Get a Groq API Key

1. Go to [Groq API Keys](https://console.groq.com/keys?utm_source=chatgpt.com)
2. Sign in or create a Groq account.
3. Create a new API key.
4. Copy the key and keep it private.

> **Important:** Never upload your API key to GitHub. FutureMind uses the `GROQ_API_KEY` environment variable to keep the key out of the source code.

### 2. Set up the backend

Open PowerShell or a terminal in the project directory and run:

```bash
cd backend

# Create a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file inside the `backend` folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Then start the backend:

```bash
python app.py
```

The backend will run at:

```text
http://127.0.0.1:5000
```

FutureMind uses the Groq API for AI-powered responses. The application reads the API key from `GROQ_API_KEY` rather than storing it directly in the source code.
