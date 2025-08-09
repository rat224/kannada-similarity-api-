import os
from flask import Flask, request, jsonify
from transformers import pipeline

app = Flask(__name__)

# Load model only once at startup to save memory
print("Loading model...")
transcriber = pipeline("automatic-speech-recognition", model="openai/whisper-small", device=-1)  # CPU

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Whisper API is running on Render!"})

@app.route("/transcribe", methods=["POST"])
def transcribe_audio():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    audio_file = request.files["file"]
    result = transcriber(audio_file.read())
    return jsonify({"transcript": result["text"]})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT
    app.run(host="0.0.0.0", port=port)
