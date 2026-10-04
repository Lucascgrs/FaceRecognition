from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions, vision

MODELS_DIR = Path(__file__).parent / "models"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"

PoseLandmark = vision.PoseLandmark
ARMS = {
    "left": (PoseLandmark.LEFT_SHOULDER, PoseLandmark.LEFT_ELBOW, PoseLandmark.LEFT_WRIST),
    "right": (PoseLandmark.RIGHT_SHOULDER, PoseLandmark.RIGHT_ELBOW, PoseLandmark.RIGHT_WRIST),
}


def create_hand_landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        num_hands=2,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.HandLandmarker.create_from_options(options)


def create_pose_landmarker():
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        num_poses=1,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.PoseLandmarker.create_from_options(options)


def draw_hands(frame, hands_landmarks):
    for hand_landmarks in hands_landmarks:
        vision.drawing_utils.draw_landmarks(
            frame,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )


def draw_arms(frame, poses_landmarks):
    h, w = frame.shape[:2]
    for pose_landmarks in poses_landmarks:
        for shoulder, elbow, wrist in ARMS.values():
            pixels = [
                (int(pose_landmarks[i].x * w), int(pose_landmarks[i].y * h))
                for i in (shoulder, elbow, wrist)
            ]
            cv2.line(frame, pixels[0], pixels[1], (255, 200, 0), 2)
            cv2.line(frame, pixels[1], pixels[2], (255, 200, 0), 2)
            for x, y in pixels:
                cv2.circle(frame, (x, y), 6, (0, 140, 255), -1)


def make_mp_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)


def detect_hands(hand_landmarker, mp_image, timestamp_ms):
    return hand_landmarker.detect_for_video(mp_image, timestamp_ms)


def detect_pose(pose_landmarker, mp_image, timestamp_ms):
    return pose_landmarker.detect_for_video(mp_image, timestamp_ms)


def track_hands_and_arms(frame, hand_landmarker, pose_landmarker, timestamp_ms):
    mp_image = make_mp_image(frame)

    hand_result = detect_hands(hand_landmarker, mp_image, timestamp_ms)
    pose_result = detect_pose(pose_landmarker, mp_image, timestamp_ms)

    draw_hands(frame, hand_result.hand_landmarks)
    draw_arms(frame, pose_result.pose_landmarks)

    return hand_result
