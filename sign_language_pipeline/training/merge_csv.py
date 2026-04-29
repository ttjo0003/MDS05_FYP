import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

csv_files = [
    os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_0-249.csv"),
    os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_250-382.csv"),
    os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_383-399.csv"),
    os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_400-599.csv"),
]

dfs = []

for file in csv_files:
    df = pd.read_csv(file, dtype={"video_id": str})

    # keep leading zeros / restore to 5 digits
    df["video_id"] = df["video_id"].astype(str).str.zfill(5)

    print(f"{file}: {len(df)} rows")
    dfs.append(df)

combined_df = pd.concat(dfs, ignore_index=True)

# make sure video_id stays as 5-digit string
combined_df["video_id"] = combined_df["video_id"].astype(str).str.zfill(5)

# remove duplicates
combined_df = combined_df.drop_duplicates(subset=["video_id"], keep="first")

# save back to data folder
output_path = os.path.join(DATA_DIR, "metadata_holistic_pose_hand_semantic_0-599.csv")
combined_df.to_csv(output_path, index=False, encoding="utf-8-sig")

print("\n✅ Saved to:", output_path)
print("Total rows:", len(combined_df))
print("Sample video_id:")
print(combined_df["video_id"].head(10).tolist())