import argparse
import time

import cv2

from face_db import create_detector, create_recognizer, load_db, track_faces
from face_log import FaceLogger
from gesture_db import GestureRecognizer
from hand_tracking import (
    create_hand_landmarker,
    create_pose_landmarker,
    detect_hands,
    make_mp_image,
    track_hands_and_arms,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Webcam avec overlays optionnels")
    parser.add_argument(
        "--face", action="store_true", help="Active la reconnaissance de visage"
    )
    parser.add_argument(
        "--hands", action="store_true", help="Active le suivi mains/doigts/bras"
    )
    parser.add_argument(
        "--geste", action="store_true", help="Active la reconnaissance de gestes appris"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detector = recognizer = db = face_logger = None
    if args.face:
        detector = create_detector()
        recognizer = create_recognizer()
        db = load_db()
        face_logger = FaceLogger(recognizer)
        if not db:
            print("Aucun visage enregistre. Lance d'abord: python enroll.py <nom>")

    need_hands = args.hands or args.geste
    hand_landmarker = create_hand_landmarker() if need_hands else None
    pose_landmarker = create_pose_landmarker() if args.hands else None

    gesture_recognizer = None
    if args.geste:
        gesture_recognizer = GestureRecognizer()
        if not gesture_recognizer.db:
            print("Aucun geste enregistre. Lance d'abord: python record_gesture.py")

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    start_time = time.time()

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            if args.face:
                track_faces(frame, detector, recognizer, db, logger=face_logger)

            if need_hands:
                timestamp_ms = int((time.time() - start_time) * 1000)
                if args.hands:
                    hand_result = track_hands_and_arms(
                        frame, hand_landmarker, pose_landmarker, timestamp_ms
                    )
                else:
                    mp_image = make_mp_image(frame)
                    hand_result = detect_hands(hand_landmarker, mp_image, timestamp_ms)

                if args.geste:
                    label = gesture_recognizer.update(hand_result.hand_landmarks)
                    if label:
                        cv2.putText(
                            frame, label, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2,
                        )

            cv2.imshow("Webcam", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if hand_landmarker:
            hand_landmarker.close()
        if pose_landmarker:
            pose_landmarker.close()


if __name__ == "__main__":
    main()
