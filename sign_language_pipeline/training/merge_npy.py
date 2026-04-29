import os
import shutil

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "..", "data")

SOURCE_FOLDERS = [
    "processed_holistic_pose_hand_semantic_0-249",
    "processed_holistic_pose_hand_semantic_250-382",
    "processed_holistic_pose_hand_semantic_383-399",
    "processed_holistic_pose_hand_semantic_400-599",
]

TARGET_FOLDER = "processed_holistic_pose_hand_semantic_0-599"
TARGET_PATH = os.path.join(DATA_DIR, TARGET_FOLDER)

os.makedirs(TARGET_PATH, exist_ok=True)

copied = 0
skipped = 0

for folder in SOURCE_FOLDERS:
    src_path = os.path.join(DATA_DIR, folder)

    if not os.path.exists(src_path):
        print(f"⚠️ Missing folder: {src_path}")
        continue

    files = os.listdir(src_path)

    for file in files:
        if not file.endswith(".npy"):
            continue

        src_file = os.path.join(src_path, file)
        dst_file = os.path.join(TARGET_PATH, file)

        if os.path.exists(dst_file):
            skipped += 1
            continue

        shutil.copy2(src_file, dst_file)
        copied += 1

    print(f"✔ Finished folder: {folder}")

print("\n✅ Merge complete")
print(f"Copied: {copied}")
print(f"Skipped (duplicates): {skipped}")
print(f"Total files in merged folder: {len(os.listdir(TARGET_PATH))}")