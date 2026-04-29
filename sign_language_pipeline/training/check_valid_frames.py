import os
import cv2
import numpy as np
import mediapipe as mp

# =========================
# CONFIG
# =========================
VIDEO_ID = "14885"

VIDEO_PATH = rf"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos\{VIDEO_ID}.mp4"

OUTPUT_DIR = rf"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\debug_video_{VIDEO_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

FEATURE_DIM = 225

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
# EXTRACT 225 FEATURES
# pose: 33 * 3 = 99
# left hand: 21 * 3 = 63
# right hand: 21 * 3 = 63
# total = 225
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

    return np.concatenate([pose, left_hand, right_hand])


# =========================
# MAIN DEBUG
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("Video:", VIDEO_ID)
print("FPS:", fps)
print("Total frames:", total_frames)

frame_idx = 0
valid_frames = 0
hand_frames = 0
pose_frames = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(rgb)

    keypoints = extract_pose_hand_keypoints(results)

    has_pose = results.pose_landmarks is not None
    has_lh = results.left_hand_landmarks is not None
    has_rh = results.right_hand_landmarks is not None
    has_hand = has_lh or has_rh
    valid = np.any(keypoints != 0)

    if valid:
        valid_frames += 1
    if has_hand:
        hand_frames += 1
    if has_pose:
        pose_frames += 1

    # draw MediaPipe landmarks directly
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_holistic.POSE_CONNECTIONS
        )

    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.left_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            frame,
            results.right_hand_landmarks,
            mp_holistic.HAND_CONNECTIONS
        )

    status_text = (
        f"frame {frame_idx}/{total_frames - 1} | "
        f"valid={valid} | pose={has_pose} | LH={has_lh} | RH={has_rh}"
    )

    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0) if valid else (0, 0, 255),
        2
    )

    save_path = os.path.join(OUTPUT_DIR, f"frame_{frame_idx:03d}.jpg")
    cv2.imwrite(save_path, frame)

    print(status_text)

    frame_idx += 1

cap.release()
holistic.close()

print("\n===== SUMMARY =====")
print("Total frames:", total_frames)
print("Checked frames:", frame_idx)
print("Valid frames:", valid_frames)
print("Pose frames:", pose_frames)
print("Hand frames:", hand_frames)
print("Preview saved to:", OUTPUT_DIR)