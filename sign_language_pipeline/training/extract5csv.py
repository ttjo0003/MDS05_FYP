import os
import pandas as pd

# =========================
# CONFIG
# =========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

INPUT_CSV = os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_0-599.csv")
OUTPUT_CSV = os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_5gloss.csv")

TARGET_GLOSSES = ["drink", "computer", "thin", "who", "before"]

# =========================
# LOAD & FILTER
# =========================
df = pd.read_csv(INPUT_CSV, dtype={"video_id": str})
df["video_id"] = df["video_id"].astype(str).str.zfill(5)

df_filtered = df[df["gloss"].isin(TARGET_GLOSSES)].copy()

# (Optional) drop duplicates just in case
df_filtered = df_filtered.drop_duplicates(subset=["video_id"], keep="first")

# =========================
# SAVE
# =========================
df_filtered.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")

print("✅ Saved:", OUTPUT_CSV)
print("Total rows:", len(df_filtered))
print("Gloss distribution:")
print(df_filtered["gloss"].value_counts())