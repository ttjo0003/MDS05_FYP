from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)

# -------------------------------
# Model settings
# -------------------------------
INPUT_SIZE = 225
HIDDEN_SIZE = 128
NUM_LAYERS = 2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "..", "model", "best_sign_lstm.pth")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "..", "model", "label_map.json")

# -------------------------------
# Define model
# -------------------------------
class SignLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.3
        )
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

# -------------------------------
# Load label map
# -------------------------------
if not os.path.exists(LABEL_MAP_PATH):
    raise FileNotFoundError(f"Cannot find {LABEL_MAP_PATH}")

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

# Expected format:
# {
#   "label2id": {"book": 0, "hello": 1},
#   "id2label": {"0": "book", "1": "hello"}
# }
id2label = {int(k): v for k, v in label_map["id2label"].items()}
num_classes = len(id2label)

# -------------------------------
# Load model
# -------------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Cannot find {MODEL_PATH}")

model = SignLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

@app.route("/")
def home():
    return jsonify({"message": "Sign language inference backend is running."})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in request body"}), 400

        sequence = np.array(data["sequence"], dtype=np.float32)

        # Expected shape: (30, 225)
        if sequence.shape != (30, 225):
            return jsonify({
                "error": f"Expected sequence shape (30, 225), got {sequence.shape}"
            }), 400

        x = torch.tensor(sequence).unsqueeze(0)  # (1, 30, 225)

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            pred_id = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_id].item())

        prediction = id2label[pred_id]

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)