import cv2
import numpy as np
import os

# =========================
# CONFIG
# =========================
VIDEO_ID = "08919"

# Correct way to insert VIDEO_ID into path
VIDEO_PATH = rf"D:\Monash University\Monash 2026 Sem 1\FIT 3164\WLASL Dataset\videos\{VIDEO_ID}.mp4"

# Your .npy directory
NPY_PATH = rf"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\processed_holistic_pose_hand_214_semantic\{VIDEO_ID}.npy"

# Output inside your data folder
OUTPUT_DIR = rf"D:\Monash University\Monash 2026 Sem 1\FIT 3164\FYP sign language translator\MDS05_FYP\sign_language_pipeline\data\preview_from_npy_{VIDEO_ID}"

os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_FRAMES = 30

# =========================
# LOAD DATA
# =========================
sequence = np.load(NPY_PATH)

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
# RESELECT SAME FRAMES
# (use same logic as preprocess)
# =========================
import mediapipe as mp
mp_holistic = mp.solutions.holistic

holistic = mp_holistic.Holistic(
    static_image_mode=False,
    model_complexity=1,
    smooth_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
            0, len(important_indices) - 1, num_frames
        ).astype(int)
        return np.array(important_indices)[selected_pos]

    selected = important_indices.copy()

    if len(idle_indices) > 0:
        selected += idle_indices[:2] + idle_indices[-2:]

    selected = sorted(list(set(selected)))

    if len(selected) < num_frames:
        selected = np.array(selected)
        selected_pos = np.linspace(0, len(selected) - 1, num_frames).astype(int)
        return selected[selected_pos]

    selected = np.array(selected)
    selected_pos = np.linspace(0, len(selected) - 1, num_frames).astype(int)
    return selected[selected_pos]


# =========================
# DRAW LANDMARKS FROM NPY
# =========================
def draw_from_npy(frame, keypoints):
    h, w, _ = frame.shape

    pose = keypoints[:88].reshape(22, 4)
    left_hand = keypoints[88:151].reshape(21, 3)
    right_hand = keypoints[151:214].reshape(21, 3)

    # =========================
    # HAND CONNECTIONS (MediaPipe standard)
    # =========================
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),
        (0,17)
    ]

    # =========================
    # POSE CONNECTIONS (subset upper body)
    # indices 11–32 → remapped to 0–21
    # =========================
    POSE_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,7),
        (0,4),(4,5),(5,6),(6,8),
        (9,10),(11,12),
        (11,13),(13,15),
        (12,14),(14,16)
    ]

    # =========================
    # DRAW POSE (blue lines + yellow points)
    # =========================
    for start, end in POSE_CONNECTIONS:
        x1 = int(pose[start][0] * w)
        y1 = int(pose[start][1] * h)
        x2 = int(pose[end][0] * w)
        y2 = int(pose[end][1] * h)

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (x1, y1), (x2, y2), (255, 191, 0), 2)  # blue-ish

    for lm in pose:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)  # yellow

    # =========================
    # DRAW LEFT HAND (#00FF00 → (0,255,0))
    # =========================
    for start, end in HAND_CONNECTIONS:
        x1 = int(left_hand[start][0] * w)
        y1 = int(left_hand[start][1] * h)
        x2 = int(left_hand[end][0] * w)
        y2 = int(left_hand[end][1] * h)

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    for lm in left_hand:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

    # =========================
    # DRAW RIGHT HAND (#FF0000 → (0,0,255))
    # =========================
    for start, end in HAND_CONNECTIONS:
        x1 = int(right_hand[start][0] * w)
        y1 = int(right_hand[start][1] * h)
        x2 = int(right_hand[end][0] * w)
        y2 = int(right_hand[end][1] * h)

        if x1 > 0 and y1 > 0 and x2 > 0 and y2 > 0:
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

    for lm in right_hand:
        x = int(lm[0] * w)
        y = int(lm[1] * h)
        if x > 0 and y > 0:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)

    return frame


# =========================
# MAIN
# =========================
frames = read_video_frames(VIDEO_PATH)

indices = select_useful_frame_indices(frames, NUM_FRAMES)

print("Selected indices:", indices)

for i, idx in enumerate(indices):
    frame = frames[int(idx)].copy()
    keypoints = sequence[i]

    # draw skeleton
    frame = draw_from_npy(frame, keypoints)

    # =========================
    # EXTRACT STATUS FROM NPY
    # =========================
    pose = keypoints[:88]
    left_hand = keypoints[88:151]
    right_hand = keypoints[151:214]

    has_pose = np.any(pose != 0)
    has_lh = np.any(left_hand != 0)
    has_rh = np.any(right_hand != 0)

    valid = has_pose or has_lh or has_rh

    # =========================
    # BUILD STATUS TEXT
    # =========================
    status_text = f"sample {i:02d} | original frame {idx} | valid={valid}"

    if has_lh:
        status_text += " | LH"
    if has_rh:
        status_text += " | RH"
    if has_pose:
        status_text += " | POSE"

    # =========================
    # DRAW TEXT
    # =========================
    cv2.putText(
        frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    save_path = os.path.join(OUTPUT_DIR, f"frame_{i:02d}.jpg")
    cv2.imwrite(save_path, frame)

print("Preview saved to:", OUTPUT_DIR)

holistic.close()