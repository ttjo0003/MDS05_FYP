import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
VIDEO_DIR = r"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos"

TEST_VIDEO_ID = "69419"

NUM_FRAMES = 30
FEATURE_DIM = 214
MIN_VALID_FRAMES = 5

PREVIEW_DIR = f"preview_landmarks_{TEST_VIDEO_ID}"
NPY_OUTPUT_PATH = f"test_sequence_{TEST_VIDEO_ID}.npy"

os.makedirs(PREVIEW_DIR, exist_ok=True)

# =========================
# MEDIAPIPE
# =========================
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


# =========================
# EXTRACT 214 FEATURES
# Same as your wlasl_preprocess.py
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
# READ ALL FRAMES
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
# MOTION-BASED FRAME SELECTION
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

    print("Total frames:", len(frames))
    print("Important (hand) frames:", len(important_indices))
    print("Idle frames:", len(idle_indices))

    # Case 1: enough important frames
    if len(important_indices) >= num_frames:
        selected_pos = np.linspace(
            0, len(important_indices) - 1, num_frames
        ).astype(int)
        return np.array(important_indices)[selected_pos]

    # Case 2: not enough → mix with idle
    selected = important_indices.copy()

    # keep only few idle frames (2 start + 2 end)
    if len(idle_indices) > 0:
        start_idle = idle_indices[:2]
        end_idle = idle_indices[-2:]
        selected += start_idle + end_idle

    # remove duplicates & sort
    selected = sorted(list(set(selected)))

    # still less than 30 → resample
    if len(selected) < num_frames:
        selected = np.array(selected)
        selected_pos = np.linspace(
            0, len(selected) - 1, num_frames
        ).astype(int)
        return selected[selected_pos]

    # more than 30 → downsample
    selected = np.array(selected)
    selected_pos = np.linspace(
        0, len(selected) - 1, num_frames
    ).astype(int)

    return selected[selected_pos]


# =========================
# DRAW LANDMARKS
# =========================
def draw_landmarks_on_frame(frame, results):
    annotated = frame.copy()

    # Draw only pose landmarks 11-32
    if results.pose_landmarks:
        h, w, _ = annotated.shape

        pose_points = results.pose_landmarks.landmark

        for idx in range(11, 33):
            lm = pose_points[idx]
            x = int(lm.x * w)
            y = int(lm.y * h)

            cv2.circle(annotated, (x, y), 4, (0, 255, 0), -1)

        # Only draw pose connections where both points are between 11 and 32
        for connection in mp_holistic.POSE_CONNECTIONS:
            start_idx, end_idx = connection

            if 11 <= start_idx <= 32 and 11 <= end_idx <= 32:
                start_lm = pose_points[start_idx]
                end_lm = pose_points[end_idx]

                start_point = (int(start_lm.x * w), int(start_lm.y * h))
                end_point = (int(end_lm.x * w), int(end_lm.y * h))

                cv2.line(annotated, start_point, end_point, (255, 255, 255), 2)

    # Draw left hand
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    # Draw right hand
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            annotated,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    return annotated


# =========================
# MAIN TEST
# =========================
def main():
    video_path = os.path.join(VIDEO_DIR, TEST_VIDEO_ID + ".mp4")

    print("Testing video:", video_path)

    if not os.path.exists(video_path):
        print("Video not found.")
        return

    frames = read_video_frames(video_path)

    if len(frames) == 0:
        print("Cannot read video frames.")
        return

    selected_indices = select_useful_frame_indices(
        frames,
        num_frames=NUM_FRAMES
        )

    if len(selected_indices) != NUM_FRAMES:
        print("Failed to select 30 frames.")
        return

    print("Selected frame indices:")
    print(selected_indices)

    sequence = []
    valid_frames = 0

    for i, frame_index in enumerate(selected_indices):
        frame = frames[frame_index]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb)

        keypoints = extract_pose_hand_keypoints(results)
        sequence.append(keypoints)

        has_keypoints = np.any(keypoints != 0)

        if has_keypoints:
            valid_frames += 1

        annotated = draw_landmarks_on_frame(frame, results)

        status_text = f"sample {i:02d} | original frame {frame_index} | valid={has_keypoints}"

        if results.left_hand_landmarks:
            status_text += " | LH"
        if results.right_hand_landmarks:
            status_text += " | RH"
        if results.pose_landmarks:
            status_text += " | POSE"

        cv2.putText(
            annotated,
            status_text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        save_path = os.path.join(PREVIEW_DIR, f"sample_{i:02d}.jpg")
        cv2.imwrite(save_path, annotated)

    sequence = np.array(sequence, dtype=np.float32)

    print("\nSequence shape:", sequence.shape)
    print("Valid frames:", valid_frames, "/", NUM_FRAMES)

    if sequence.shape == (NUM_FRAMES, FEATURE_DIM):
        np.save(NPY_OUTPUT_PATH, sequence)
        print("Saved test npy:", NPY_OUTPUT_PATH)
    else:
        print("Wrong sequence shape. NPY not saved.")

    if valid_frames < MIN_VALID_FRAMES:
        print("Warning: this video would be filtered out by your preprocessing.")
    else:
        print("This video passes MIN_VALID_FRAMES.")

    print("Preview images saved in:", PREVIEW_DIR)

    holistic.close()


if __name__ == "__main__":
    main()