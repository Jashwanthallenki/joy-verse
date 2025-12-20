import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import torch.nn.functional as F
from model import EmotionTransformer

app = Flask(__name__)
CORS(app)

# ---------- Load Model Safely ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = EmotionTransformer(input_dim=1404).to(device)

model.load_state_dict(
    torch.load("best_model.pth", map_location=device)
)

model.eval()

# ---------- Prediction Route ----------
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "landmarks" not in data:
            return jsonify({"error": "No landmarks data provided"}), 400

        landmarks = data["landmarks"]

        if not isinstance(landmarks, list):
            return jsonify({"error": "Landmarks must be a list"}), 400

        landmarks_array = np.array(landmarks).flatten()

        if landmarks_array.shape[0] != 1404:
            return jsonify({
                "error": "Invalid landmarks shape",
                "expected": 1404,
                "received": landmarks_array.shape[0]
            }), 400

        landmarks_tensor = torch.tensor(
            landmarks_array, dtype=torch.float32
        ).view(1, 1404).to(device)

        with torch.no_grad():
            output = model(landmarks_tensor)
            probabilities = F.softmax(output, dim=1)
            _, predicted = torch.max(probabilities, 1)

        emotion = get_emotion_label(predicted.item())

        return jsonify({"emotion": emotion})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_emotion_label(index):
    emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness']
    return emotion_labels[index] if index < len(emotion_labels) else "Unknown"
