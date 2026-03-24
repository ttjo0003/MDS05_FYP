import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR = os.path.join(BASE_DIR, "..", "data", "videos")
JSON_PATH = os.path.join(BASE_DIR, "..", "data", "WLASL_v0.3.json")

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed")
METADATA_PATH = os.path.join(BASE_DIR, "..", "data", "metadata.csv")

NUM_FRAMES = 30
MIN_VALID_FRAMES = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MEDIAPIPE HOLISTIC
# =========================
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# =========================
# EXTRACT POSE + LEFT HAND + RIGHT HAND
# total per frame = 225
# =========================
def extract_pose_hand_keypoints(results):
    pose = np.zeros(33 * 3)
    left_hand = np.zeros(21 * 3)
    right_hand = np.zeros(21 * 3)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark]
        ).flatten()

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]
        ).flatten()

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]
        ).flatten()

    return np.concatenate([pose, left_hand, right_hand])

# =========================
# SAMPLE FRAMES UNIFORMLY
# =========================
def sample_frames(video_path, num_frames=30):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total_frames - 1, num_frames).astype(int)

    frames = []
    current = 0
    idx_set = set(indices)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if current in idx_set:
            frames.append(frame)

        current += 1

    cap.release()
    return frames

# =========================
# LOAD JSON
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

# =========================
# LOAD EXISTING METADATA
# =========================
if os.path.exists(METADATA_PATH):
    existing_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
    processed_ids = set(existing_df["video_id"].astype(str).tolist())
    print(f"Loaded existing metadata: {len(existing_df)} rows")
else:
    existing_df = pd.DataFrame(columns=["video_id", "gloss", "split", "path", "valid_frames"])
    processed_ids = set()
    print("No existing metadata found, starting fresh")

new_rows = []
processed_count = 0
skipped_existing_count = 0
failed_count = 0

# =========================
# MAIN LOOP
# =========================
for entry in tqdm(data):
    gloss = entry["gloss"]

    for inst in entry["instances"]:
        video_id = str(inst["video_id"])
        split = inst["split"]

        if video_id in processed_ids:
            skipped_existing_count += 1
            continue

        video_path = os.path.join(VIDEO_DIR, video_id + ".mp4")
        save_path = os.path.join(OUTPUT_DIR, video_id + ".npy")

        if not os.path.exists(video_path):
            failed_count += 1
            continue

        if os.path.exists(save_path):
            try:
                sequence = np.load(save_path)
                if sequence.shape == (NUM_FRAMES, 225):
                    valid_frames = int(np.sum(np.any(sequence != 0, axis=1)))

                    row = {
                        "video_id": video_id,
                        "gloss": gloss,
                        "split": split,
                        "path": save_path,
                        "valid_frames": valid_frames
                    }
                    new_rows.append(row)
                    processed_ids.add(video_id)

                    pd.DataFrame([row]).to_csv(
                        METADATA_PATH,
                        mode="a",
                        header=not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0,
                        index=False,
                        encoding="utf-8-sig"
                    )

                    print(f"{video_id} already has npy, metadata restored")
                    continue
            except Exception as e:
                print(f"Failed to load existing npy for {video_id}: {e}")

        frames = sample_frames(video_path, NUM_FRAMES)
        if len(frames) != NUM_FRAMES:
            failed_count += 1
            continue

        sequence = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            keypoints = extract_pose_hand_keypoints(results)
            sequence.append(keypoints)

        sequence = np.array(sequence)

        if sequence.shape != (NUM_FRAMES, 225):
            failed_count += 1
            continue

        valid_frames = int(np.sum(np.any(sequence != 0, axis=1)))
        print(f"{video_id} | valid_frames: {valid_frames}/{NUM_FRAMES}")

        if valid_frames < MIN_VALID_FRAMES:
            failed_count += 1
            continue

        np.save(save_path, sequence)

        row = {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "path": save_path,
            "valid_frames": valid_frames
        }

        new_rows.append(row)
        processed_ids.add(video_id)
        processed_count += 1

        pd.DataFrame([row]).to_csv(
            METADATA_PATH,
            mode="a",
            header=not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0,
            index=False,
            encoding="utf-8-sig"
        )

# =========================
# DONE
# =========================
holistic.close()

print("\nDone!")
print("Newly processed:", processed_count)
print("Skipped existing:", skipped_existing_count)
print("Failed / filtered:", failed_count)

final_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
final_df = final_df.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)
final_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

print("Final metadata rows:", len(final_df))