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

VIDEO_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos"
JSON_PATH = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\WLASL_v0.3.json"

# =========================
# GLOSS SPLIT CONFIG
# =========================
START_GLOSS_IDX = 0
END_GLOSS_IDX = 599

OUTPUT_DIR = os.path.join(
    BASE_DIR, "..", "data",
    f"processed_holistic_pose_hand_semantic_{START_GLOSS_IDX}-{END_GLOSS_IDX}"
)

METADATA_PATH = os.path.join(
    BASE_DIR, "..", "data",
    f"metadata_holistic_pose_hand_semantic_{START_GLOSS_IDX}-{END_GLOSS_IDX}.csv"
)

NUM_FRAMES = 25
MIN_VALID_FRAMES = 5
MIN_HAND_FRAMES = 5
FEATURE_DIM = 225

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# MEDIAPIPE
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
# KEYPOINT EXTRACTION
# =========================
def extract_pose_hand_keypoints(results):
    pose = np.zeros(33 * 3, dtype=np.float32)
    left_hand = np.zeros(21 * 3, dtype=np.float32)
    right_hand = np.zeros(21 * 3, dtype=np.float32)

    if results.pose_landmarks:
        pose = np.array(
            [[lm.x, lm.y, lm.z] for lm in results.pose_landmarks.landmark],
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
# READ VIDEO
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
# =========================
def select_useful_frame_indices(frames, num_frames=25):
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

    selected = important_indices.copy()

    if len(selected) == 0:
        selected = list(range(len(frames)))
    else:
        selected += idle_indices[:2] + idle_indices[-2:]

    selected = sorted(list(set(selected)))

    selected_pos = np.linspace(
        0,
        len(selected) - 1,
        num_frames
    ).astype(int)

    return np.array(selected)[selected_pos]

# =========================
# PROCESS VIDEO
# =========================
def process_video(video_path):
    frames = read_video_frames(video_path)

    # if len(frames) < NUM_FRAMES:
    #     return None, 0, len(frames), 0
    if len(frames) == 0:
        return None, 0, 0, 0

    selected_indices = select_useful_frame_indices(frames, NUM_FRAMES)

    if len(selected_indices) != NUM_FRAMES:
        return None, 0, len(frames), 0

    sequence = []
    hand_frames = 0

    for idx in selected_indices:
        frame = frames[int(idx)]
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
# LOAD DATA
# =========================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data = data[START_GLOSS_IDX:END_GLOSS_IDX + 1]

print(f"Processing gloss index from {START_GLOSS_IDX} to {END_GLOSS_IDX}")
print(f"Total glosses in this run: {len(data)}")
print(f"Output folder: {OUTPUT_DIR}")
print(f"Metadata file: {METADATA_PATH}")

# =========================
# LOAD EXISTING METADATA FOR RESUME
# =========================
if os.path.exists(METADATA_PATH) and os.path.getsize(METADATA_PATH) > 0:
    existing_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
    existing_df = existing_df.drop_duplicates(subset=["video_id"], keep="last")

    processed_ids = set(existing_df["video_id"].astype(str).tolist())

    print(f"Resume enabled: found {len(processed_ids)} processed videos in CSV.")
else:
    processed_ids = set()
    print("No existing metadata found. Starting from scratch for this range.")

processed_count = 0
skipped_count = 0
failed_count = 0
missing_video_count = 0

# =========================
# MAIN LOOP
# =========================
for gloss_idx, entry in enumerate(tqdm(data), start=START_GLOSS_IDX):
    gloss = entry["gloss"]

    print(f"\n===== GLOSS {gloss_idx}: {gloss} =====")

    for inst in entry["instances"]:
        video_id = str(inst["video_id"])
        split = inst.get("split", "")

        video_path = os.path.join(VIDEO_DIR, video_id + ".mp4")
        save_path = os.path.join(OUTPUT_DIR, video_id + ".npy")

        # Resume logic based on CSV only
        if video_id in processed_ids:
            print(f"↪ SKIPPED CSV: {video_id} | {gloss} | already in metadata")
            skipped_count += 1
            continue

        if not os.path.exists(video_path):
            print(f"✖ MISSING VIDEO: {video_id} | {gloss}")
            missing_video_count += 1
            failed_count += 1
            continue

        sequence, valid_frames, total_frames, hand_frames = process_video(video_path)

        print(
            f"{video_id} | {gloss} | "
            f"total_frames: {total_frames} | "
            f"valid_frames: {valid_frames}/{NUM_FRAMES} | "
            f"hand_frames: {hand_frames}/{NUM_FRAMES}"
        )

        if sequence is None:
            print(f"✖ FAILED: {video_id} | {gloss} | could not create valid sequence")
            failed_count += 1
            continue
            
        if valid_frames < MIN_VALID_FRAMES:
            print(
                f"✖ FILTERED: {video_id} | {gloss} | "
                f"valid_frames {valid_frames} < MIN_VALID_FRAMES {MIN_VALID_FRAMES}"
            )
            failed_count += 1
            continue
        
        if hand_frames < MIN_HAND_FRAMES:
            print(
                f"✖ FILTERED: {video_id} | {gloss} | "
                f"hand_frames {hand_frames} < MIN_HAND_FRAMES {MIN_HAND_FRAMES}"
            )
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

        pd.DataFrame([row]).to_csv(
            METADATA_PATH,
            mode="a",
            header=not os.path.exists(METADATA_PATH) or os.path.getsize(METADATA_PATH) == 0,
            index=False,
            encoding="utf-8-sig"
        )

        processed_ids.add(video_id)
        processed_count += 1

        print(f"✔ SAVED: {video_id} | {gloss}")

# =========================
# DONE
# =========================
holistic.close()

print("\nDone!")
print("Newly processed:", processed_count)
print("Skipped from CSV:", skipped_count)
print("Missing videos:", missing_video_count)
print("Failed / filtered:", failed_count)

if os.path.exists(METADATA_PATH) and os.path.getsize(METADATA_PATH) > 0:
    final_df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
    final_df = final_df[["video_id", "gloss", "split", "path", "valid_frames"]]
    final_df = final_df.drop_duplicates(subset=["video_id"], keep="last").reset_index(drop=True)
    final_df.to_csv(METADATA_PATH, index=False, encoding="utf-8-sig")

    print("Final metadata rows:", len(final_df))
    print("\nFinal gloss distribution:")
    print(final_df["gloss"].value_counts())
else:
    print("No metadata file was created.")