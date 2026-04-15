from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import numpy as np
import json
import os

app = Flask(__name__)
CORS(app)
INPUT_SIZE = 225
HIDDEN_SIZE = 128
NUM_LAYERS = 2
TARGET_FRAMES = 30
MIN_NON_ZERO_FRAMES = 20
CONFIDENCE_THRESHOLD = 0.20
TOP_K = 3

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "best_sign_lstm.pth"),
    os.path.join(BASE_DIR, "..", "model", "best_sign_lstm.pth"),
]

LABEL_MAP_CANDIDATES = [
    os.path.join(BASE_DIR, "label_map.json"),
    os.path.join(BASE_DIR, "..", "model", "label_map.json"),
]


def resolve_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


MODEL_PATH = resolve_existing_path(MODEL_CANDIDATES)
LABEL_MAP_PATH = resolve_existing_path(LABEL_MAP_CANDIDATES)

if MODEL_PATH is None:
    raise FileNotFoundError(
        f"Cannot find model file. Tried: {MODEL_CANDIDATES}"
    )

if LABEL_MAP_PATH is None:
    raise FileNotFoundError(
        f"Cannot find label_map file. Tried: {LABEL_MAP_CANDIDATES}"
    )
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

with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
    label_map = json.load(f)

id2label = {int(k): v for k, v in label_map["id2label"].items()}
num_classes = len(id2label)
model = SignLSTM(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, num_classes)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()
def count_non_zero_frames(sequence: np.ndarray) -> int:
    # sequence shape: (30, 225)
    non_zero_mask = np.any(np.abs(sequence) > 1e-8, axis=1)
    return int(np.sum(non_zero_mask))


def count_frames_with_hand_signal(sequence: np.ndarray) -> int:
    # features:
    # pose: 0:99
    # left hand: 99:162
    # right hand: 162:225
    left = sequence[:, 99:162]
    right = sequence[:, 162:225]

    left_non_zero = np.any(np.abs(left) > 1e-8, axis=1)
    right_non_zero = np.any(np.abs(right) > 1e-8, axis=1)

    hand_non_zero = np.logical_or(left_non_zero, right_non_zero)
    return int(np.sum(hand_non_zero))


def build_top_k(probs: torch.Tensor, k: int = TOP_K):
    k = min(k, probs.shape[1])
    values, indices = torch.topk(probs, k=k, dim=1)

    results = []
    for score, idx in zip(values[0].tolist(), indices[0].tolist()):
        results.append({
            "label": id2label[int(idx)],
            "confidence": round(float(score), 4)
        })
    return results


@app.route("/")
def home():
    return jsonify({
        "message": "Sign language inference backend is running.",
        "model_path": MODEL_PATH,
        "label_map_path": LABEL_MAP_PATH
    })


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "sequence" not in data:
            return jsonify({"error": "Missing 'sequence' in request body"}), 400

        sequence = np.array(data["sequence"], dtype=np.float32)

        if sequence.shape != (TARGET_FRAMES, INPUT_SIZE):
            return jsonify({
                "error": f"Expected sequence shape ({TARGET_FRAMES}, {INPUT_SIZE}), got {sequence.shape}"
            }), 400

        non_zero_frames = count_non_zero_frames(sequence)
        hand_signal_frames = count_frames_with_hand_signal(sequence)

        if non_zero_frames < MIN_NON_ZERO_FRAMES:
            return jsonify({
                "error": (
                    f"Too few informative frames: {non_zero_frames}/{TARGET_FRAMES}. "
                    "Please perform the sign more clearly."
                )
            }), 400

        if hand_signal_frames < MIN_NON_ZERO_FRAMES:
            return jsonify({
                "error": (
                    f"Too few hand-signal frames: {hand_signal_frames}/{TARGET_FRAMES}. "
                    "Keep at least one hand visible while signing."
                )
            }), 400

        x = torch.tensor(sequence).unsqueeze(0)  

        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1)

            pred_id = int(torch.argmax(probs, dim=1).item())
            confidence = float(probs[0, pred_id].item())

            top_k = build_top_k(probs, k=TOP_K)

        prediction = id2label[pred_id]

        if confidence < CONFIDENCE_THRESHOLD:
            return jsonify({
                "prediction": "Uncertain - please repeat the sign",
                "confidence": round(confidence, 4),
                "accepted": False,
                "message": "Prediction confidence is low. Please sign again with clearer hand visibility.",
                "top_k": top_k,
                "quality": {
                    "non_zero_frames": non_zero_frames,
                    "hand_signal_frames": hand_signal_frames
                }
            })

        return jsonify({
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "accepted": True,
            "message": "Prediction received successfully.",
            "top_k": top_k,
            "quality": {
                "non_zero_frames": non_zero_frames,
                "hand_signal_frames": hand_signal_frames
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)