from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import os
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow Chrome extension to access this server

# -------------------------------
# Hugging Face API configuration
# -------------------------------
HF_API_URL = "https://api-inference.huggingface.co/models/distilgpt2"
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Load from .env

# -------------------------------
# Test route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "📖 DeepReader Flask Server (Hugging Face) is running! Model: distilgpt2"

# -------------------------------
# Main analyze route
# -------------------------------
@app.route("/analyze", methods=["POST"])
def analyze_text():
    data = request.get_json()
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Simplified prompt for small models
    prompt = f"""
Analyze the following text for a literary reader:
Text: "{text}"
1. Simplified Meaning:
2. Metaphors/Symbols:
3. Philosophical Themes:
"""

    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "do_sample": True,
            "return_full_text": False
        }
    }

    try:
        response = requests.post(HF_API_URL, headers=headers, json=payload, timeout=45)
        response.raise_for_status()
        result = response.json()

        completion_text = result[0].get("generated_text", "").strip()
        analyzed_text = prompt + completion_text

        if not analyzed_text or "1. Simplified Meaning:" not in analyzed_text:
            analyzed_text = (
                f"1. Simplified Meaning:\n{text[:200]}... (Fallback)\n\n"
                "2. Metaphors/Symbols:\nNone\n\n"
                "3. Philosophical Themes:\nNone"
            )

    except Exception as e:
        print("=== HUGGING FACE API ERROR ===")
        print(e)
        analyzed_text = (
            f"1. Simplified Meaning:\n{text[:200]}... (API Request Failed)\n\n"
            "2. Metaphors/Symbols:\nNone\n\n"
            "3. Philosophical Themes:\nNone"
        )

    return jsonify({"analyzed": analyzed_text})


# -------------------------------
# Run the Flask server
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
