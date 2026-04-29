import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

INPUT_NPY_DIR = os.path.join(DATA_DIR, "processed_holistic_pose_hand_semantic_0-599")
INPUT_METADATA = os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_0-599.csv")

# =========================
# SPLIT CONFIG
# Change these for each teammate
# =========================
START_ROW_IDX = 0
END_ROW_IDX = 4787

OUTPUT_NPY_DIR = os.path.join(
    DATA_DIR,
    f"processed_holistic_pose_hand_semantic_0-599_normalized_{START_ROW_IDX}-{END_ROW_IDX}"
)

OUTPUT_METADATA = os.path.join(
    DATA_DIR,
    f"metadata_holistic_pose_hand_semantic_0-599_normalized_{START_ROW_IDX}-{END_ROW_IDX}.csv"
)

os.makedirs(OUTPUT_NPY_DIR, exist_ok=True)

NUM_FRAMES = 25
FEATURE_DIM = 225


def normalize_sequence(seq):
    seq = seq.copy().astype(np.float32)

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

        if np.allclose(left_shoulder, 0) or np.allclose(right_shoulder, 0):
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


df = pd.read_csv(INPUT_METADATA, dtype={"video_id": str})
df["video_id"] = df["video_id"].astype(str).str.zfill(5)

# =========================
# SLICE CSV ROWS
# =========================
df = df.iloc[START_ROW_IDX:END_ROW_IDX + 1].reset_index(drop=True)

print(f"Processing rows {START_ROW_IDX} to {END_ROW_IDX}")
print("Total rows in this run:", len(df))

new_rows = []
converted = 0
skipped = 0

for _, row in df.iterrows():
    video_id = row["video_id"]

    input_path = os.path.join(INPUT_NPY_DIR, video_id + ".npy")
    output_path = os.path.join(OUTPUT_NPY_DIR, video_id + ".npy")

    if not os.path.exists(input_path):
        skipped += 1
        print(f"Missing npy: {input_path}")
        continue

    seq = np.load(input_path)

    if seq.shape != (NUM_FRAMES, FEATURE_DIM):
        skipped += 1
        print(f"Wrong shape: {video_id}, {seq.shape}")
        continue

    norm_seq = normalize_sequence(seq)

    np.save(output_path, norm_seq)

    new_row = row.copy()
    new_row["path"] = output_path
    new_rows.append(new_row)

    converted += 1

new_df = pd.DataFrame(new_rows)
new_df.to_csv(OUTPUT_METADATA, index=False, encoding="utf-8-sig")

print("\nDone!")
print("Converted:", converted)
print("Skipped:", skipped)
print("Saved folder:", OUTPUT_NPY_DIR)
print("Saved metadata:", OUTPUT_METADATA)