import os
import cv2
import json
import numpy as np
import pandas as pd
import mediapipe as mp
from tqdm import tqdm
from collections import Counter

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust the file path to point to your local dataset directory
VIDEO_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos"
JSON_PATH = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\WLASL_v0.3.json"

# save subset separately
OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "v1")
METADATA_PATH = os.path.join(BASE_DIR, "..", "data", "mdv1.csv")

NUM_FRAMES = 30
MIN_VALID_FRAMES = 5
NUM_CLASSES = 5
FEATURE_DIM = 214

# =========================
# CHOOSE TARGET WORDS
# =========================
# Option 1: set to None -> auto choose top NUM_CLASSES most frequent glosses
# Option 2: manually write your own list of words
TARGET_GLOSSES = ["before", "computer", "drink", "thin", "who"]

# Example manual setting:
# TARGET_GLOSSES = [
#     "book", "drink", "computer", "go", "help",
#     "love", "mother", "no", "yes", "please"
# ]

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
# total per frame = 214
# pose = 88, left hand = 63, right hand = 63
# =========================
def extract_pose_hand_keypoints(results):
    pose = np.zeros(88, dtype=np.float32)
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose_full = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
            dtype=np.float32
        ).flatten()

        # keep the same teammate-defined logic so pose length becomes 88
        pose = pose_full[11:]

        # safety check
        if pose.shape[0] != 88:
            pose = np.zeros(88, dtype=np.float32)

    if results.left_hand_landmarks:
        left_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    if results.right_hand_landmarks:
        right_hand = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark],
            dtype=np.float32
        ).flatten()

    keypoints = np.concatenate([pose, left_hand, right_hand])

    # final safety check
    if keypoints.shape[0] != FEATURE_DIM:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    return keypoints

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
# SELECT GLOSSES
# =========================
if TARGET_GLOSSES is None:
    gloss_counter = Counter()

    for entry in data:
        gloss = entry["gloss"]
        gloss_counter[gloss] += len(entry["instances"])

    selected_glosses = set(
        [gloss for gloss, _ in gloss_counter.most_common(NUM_CLASSES)]
    )
    print(f"Auto-selected top {NUM_CLASSES} glosses:")
else:
    selected_glosses = set(TARGET_GLOSSES)
    print("Using manually selected glosses:")

print(sorted(selected_glosses))

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
skipped_non_target_count = 0

# =========================
# FILTER DATA FIRST
# =========================
filtered_data = [entry for entry in data if entry["gloss"] in selected_glosses]
print(f"Filtered gloss entries: {len(filtered_data)}")

# =========================
# MAIN LOOP
# =========================
for entry in tqdm(filtered_data):
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
            print(f"Missing video: {video_path}")
            continue

        if os.path.exists(save_path):
            try:
                sequence = np.load(save_path)

                if sequence.shape == (NUM_FRAMES, FEATURE_DIM):
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
            print(f"{video_id} skipped: expected {NUM_FRAMES} frames, got {len(frames)}")
            continue

        sequence = []

        for frame in frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)
            keypoints = extract_pose_hand_keypoints(results)
            sequence.append(keypoints)

        sequence = np.array(sequence, dtype=np.float32)

        if sequence.shape != (NUM_FRAMES, FEATURE_DIM):
            failed_count += 1
            print(f"{video_id} skipped: wrong shape {sequence.shape}")
            continue

        valid_frames = int(np.sum(np.any(sequence != 0, axis=1)))
        print(f"{video_id} | {gloss} | valid_frames: {valid_frames}/{NUM_FRAMES}")

        if valid_frames < MIN_VALID_FRAMES:
            failed_count += 1
            print(f"{video_id} skipped: valid_frames < {MIN_VALID_FRAMES}")
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
print("Selected glosses:", sorted(selected_glosses))
print("Newly processed:", processed_count)
print("Skipped existing:", skipped_existing_count)
print("Skipped non-target gloss videos:", skipped_non_target_count)
print("Failed / filtered:", failed_count)

if os.path.exists(METADATA_PATH) and os.path.getsize(METADATA_PATH) > 0:
    final_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
    final_df = final_df.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)
    final_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print("Final metadata rows:", len(final_df))

    # show class distribution
    final_counts = final_df["gloss"].value_counts()
    print("\nFinal gloss distribution:")
    print(final_counts)
else:
    print("No metadata file was created.")