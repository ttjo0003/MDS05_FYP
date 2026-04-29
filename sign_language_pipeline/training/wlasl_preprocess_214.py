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

# VIDEO_DIR = os.path.join(BASE_DIR, "..", "data", "videos")
# JSON_PATH = os.path.join(BASE_DIR, "..", "data", "WLASL_v0.3.json")
VIDEO_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos"
JSON_PATH = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\WLASL_v0.3.json"

OUTPUT_DIR = os.path.join(BASE_DIR, "..", "data", "processed_holistic_pose_hand_214_semantic")
METADATA_PATH = os.path.join(BASE_DIR, "..", "data", "metadata_holistic_pose_hand_214_semantic.csv")

NUM_FRAMES = 30
MIN_VALID_FRAMES = 5
FEATURE_DIM = 214

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
# 214 dims total:
# pose landmarks 11-32 only: 22 * 4 = 88
# left hand: 21 * 3 = 63
# right hand: 21 * 3 = 63
# total = 88 + 63 + 63 = 214
# =========================
def extract_pose_hand_keypoints(results):
    pose = np.zeros(22 * 4, dtype=np.float32)
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose_landmarks = results.pose_landmarks.landmark[11:33]
        pose = np.array(
            [[lm.x, lm.y, lm.z, lm.visibility] for lm in pose_landmarks],
            dtype=np.float32
        ).flatten()

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

    if keypoints.shape[0] != FEATURE_DIM:
        return np.zeros(FEATURE_DIM, dtype=np.float32)

    return keypoints


# =========================
# READ VIDEO FRAMES
# =========================
def read_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        frames.append(frame)

    cap.release()
    return frames


# =========================
# SEMANTIC FRAME SELECTION
# Select frames where hands are detected.
# If fewer than 30 hand frames exist, keep all hand frames
# and only a few idle frames from start/end.
# =========================
def select_useful_frame_indices(frames, num_frames=30):
    important_indices = []
    idle_indices = []

    for i, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        has_hand = (
            results.left_hand_landmarks is not None or
            results.right_hand_landmarks is not None
        )

        if has_hand:
            important_indices.append(i)
        else:
            idle_indices.append(i)

    if len(important_indices) >= num_frames:
        selected_pos = np.linspace(
            0,
            len(important_indices) - 1,
            num_frames
        ).astype(int)

        return np.array(important_indices)[selected_pos]

    selected = important_indices.copy()

    # Keep only a few idle frames, not many repeated no-hand frames
    if len(idle_indices) > 0:
        start_idle = idle_indices[:2]
        end_idle = idle_indices[-2:]
        selected += start_idle + end_idle

    selected = sorted(list(set(selected)))

    if len(selected) == 0:
        return np.array([])

    if len(selected) < num_frames:
        selected = np.array(selected)
        selected_pos = np.linspace(
            0,
            len(selected) - 1,
            num_frames
        ).astype(int)

        return selected[selected_pos]

    selected = np.array(selected)
    selected_pos = np.linspace(
        0,
        len(selected) - 1,
        num_frames
    ).astype(int)

    return selected[selected_pos]


# =========================
# PROCESS SINGLE VIDEO
# =========================
def process_video(video_path):
    frames = read_video_frames(video_path)

    if len(frames) == 0:
        return None, 0, 0, 0

    if len(frames) < NUM_FRAMES:
        return None, 0, len(frames), 0

    selected_indices = select_useful_frame_indices(frames, NUM_FRAMES)

    if len(selected_indices) != NUM_FRAMES:
        return None, 0, len(frames), 0

    sequence = []
    hand_frames = 0

    for frame_index in selected_indices:
        frame = frames[int(frame_index)]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        if results.left_hand_landmarks or results.right_hand_landmarks:
            hand_frames += 1

        keypoints = extract_pose_hand_keypoints(results)
        sequence.append(keypoints)

    sequence = np.array(sequence, dtype=np.float32)

    if sequence.shape != (NUM_FRAMES, FEATURE_DIM):
        return None, 0, len(frames), hand_frames

    valid_frames = int(np.sum(np.any(sequence != 0, axis=1)))

    return sequence, valid_frames, len(frames), hand_frames


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
    existing_df = pd.DataFrame(
        columns=[
            "video_id",
            "gloss",
            "split",
            "path",
            "valid_frames",
            "total_frames",
            "hand_frames"
        ]
    )
    processed_ids = set()
    print("No existing metadata found, starting fresh")

processed_count = 0
skipped_existing_count = 0
failed_count = 0

# =========================
# MAIN LOOP - PROCESS ALL GLOSSES
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

        # If .npy already exists, restore metadata if shape is correct
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
                        "valid_frames": valid_frames,
                        "total_frames": "",
                        "hand_frames": ""
                    }

                    pd.DataFrame([row]).to_csv(
                        METADATA_PATH,
                        mode="a",
                        header=not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0,
                        index=False,
                        encoding="utf-8-sig"
                    )

                    processed_ids.add(video_id)
                    skipped_existing_count += 1
                    continue

            except Exception as e:
                print(f"Failed to load existing npy for {video_id}: {e}")

        sequence, valid_frames, total_frames, hand_frames = process_video(video_path)

        if sequence is None:
            failed_count += 1
            continue

        print(
            f"{video_id} | {gloss} | "
            f"valid_frames: {valid_frames}/{NUM_FRAMES} | "
            f"hand_frames: {hand_frames}/{NUM_FRAMES}"
        )

        if valid_frames < MIN_VALID_FRAMES:
            failed_count += 1
            continue

        np.save(save_path, sequence)

        row = {
            "video_id": video_id,
            "gloss": gloss,
            "split": split,
            "path": save_path,
            "valid_frames": valid_frames,
            "total_frames": total_frames,
            "hand_frames": hand_frames
        }

        pd.DataFrame([row]).to_csv(
            METADATA_PATH,
            mode="a",
            header=not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0,
            index=False,
            encoding="utf-8-sig"
        )

        processed_ids.add(video_id)
        processed_count += 1

# =========================
# DONE
# =========================
holistic.close()

print("\nDone!")
print("Newly processed:", processed_count)
print("Skipped existing:", skipped_existing_count)
print("Failed / filtered:", failed_count)

if os.path.exists(METADATA_PATH) and os.path.getsize(METADATA_PATH) > 0:
    final_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
    final_df = final_df.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)
    final_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print("Final metadata rows:", len(final_df))

    print("\nFinal gloss distribution:")
    print(final_df["gloss"].value_counts())
else:
    print("No metadata file was created.")