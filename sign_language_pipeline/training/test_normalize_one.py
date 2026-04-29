import os
import numpy as np
import pandas as pd

# =========================
# CONFIG
# =========================
METADATA_PATH = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\metadata_holistic_pose_hand_semantic_0-599.csv"

INPUT_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\processed_holistic_pose_hand_semantic_0-599"

OUTPUT_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\processed_holistic_pose_hand_semantic_0-599_normalized_5gloss"

TARGET_GLOSSES = ["drink", "computer", "thin", "who", "before"]

NUM_FRAMES = 25
FEATURE_DIM = 225

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# NORMALIZATION FUNCTION
# =========================
def normalize_sequence(seq):
    pose = seq[:, 0:99].reshape(NUM_FRAMES, 33, 3)
    left = seq[:, 99:162].reshape(NUM_FRAMES, 21, 3)
    right = seq[:, 162:225].reshape(NUM_FRAMES, 21, 3)

    normalized_frames = []

    for t in range(NUM_FRAMES):
        pose_t = pose[t]
        left_t = left[t]
        right_t = right[t]

        left_shoulder = pose_t[11]
        right_shoulder = pose_t[12]

        if np.all(left_shoulder == 0) or np.all(right_shoulder == 0):
            normalized_frames.append(seq[t])
            continue

        origin = (left_shoulder + right_shoulder) / 2.0
        scale = np.linalg.norm(left_shoulder - right_shoulder)

        if scale < 1e-6:
            scale = 1.0

        pose_norm = (pose_t - origin) / scale
        left_norm = (left_t - origin) / scale
        right_norm = (right_t - origin) / scale

        frame_norm = np.concatenate([
            pose_norm.reshape(-1),
            left_norm.reshape(-1),
            right_norm.reshape(-1)
        ])

        normalized_frames.append(frame_norm)

    return np.array(normalized_frames, dtype=np.float32)


# =========================
# LOAD METADATA
# =========================
df = pd.read_csv(METADATA_PATH, dtype={"video_id": str})
df["video_id"] = df["video_id"].astype(str).str.zfill(5)

# Filter only target glosses
df_filtered = df[df["gloss"].isin(TARGET_GLOSSES)]

print("Total selected samples:", len(df_filtered))
print("Gloss distribution:")
print(df_filtered["gloss"].value_counts())

# =========================
# PROCESS ALL FILES
# =========================
converted = 0
skipped = 0

for _, row in df_filtered.iterrows():
    video_id = row["video_id"]
    gloss = row["gloss"]

    input_path = os.path.join(INPUT_DIR, video_id + ".npy")
    output_path = os.path.join(OUTPUT_DIR, video_id + ".npy")

    if not os.path.exists(input_path):
        print(f"❌ Missing file: {video_id} ({gloss})")
        skipped += 1
        continue

    seq = np.load(input_path)

    if seq.shape != (NUM_FRAMES, FEATURE_DIM):
        print(f"❌ Wrong shape: {video_id} ({gloss}) -> {seq.shape}")
        skipped += 1
        continue

    norm_seq = normalize_sequence(seq)
    np.save(output_path, norm_seq)

    converted += 1

    if converted % 50 == 0:
        print(f"Processed {converted} files...")

print("\n✅ Done")
print("Converted:", converted)
print("Skipped:", skipped)
print("Saved folder:", OUTPUT_DIR)