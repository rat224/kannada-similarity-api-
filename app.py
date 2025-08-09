from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer, util

# Load model once at startup
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

app = Flask(__name__)

@app.route("/similarity", methods=["POST"])
def similarity():
    data = request.get_json()

    sentence1 = data.get("sentence1")
    sentence2 = data.get("sentence2")

    if not sentence1 or not sentence2:
        return jsonify({"error": "Both sentence1 and sentence2 are required"}), 400

    embeddings1 = model.encode(sentence1, convert_to_tensor=True)
    embeddings2 = model.encode(sentence2, convert_to_tensor=True)
    similarity_score = util.cos_sim(embeddings1, embeddings2).item()

    return jsonify({
        "sentence1": sentence1,
        "sentence2": sentence2,
        "similarity_score": round(similarity_score, 4)
    })

@app.route("/", methods=["GET"])
def home():
    return "Kannada Sentence Similarity API is running!"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
