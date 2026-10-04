import time

import cv2
import numpy as np

from gesture_db import MIN_SEGMENT_LENGTH, WRIST, extract_features, save_gesture
from hand_tracking import create_hand_landmarker, detect_hands, draw_hands, make_mp_image


def main():
    hand_landmarker = create_hand_landmarker()

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    start_time = time.time()
    recording = False
    sequence = []
    origin = None

    print("ESPACE: demarrer puis arreter l'enregistrement du mouvement. 'q': annuler.")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            mp_image = make_mp_image(frame)
            timestamp_ms = int((time.time() - start_time) * 1000)
            result = detect_hands(hand_landmarker, mp_image, timestamp_ms)
            draw_hands(frame, result.hand_landmarks)

            if recording and result.hand_landmarks:
                hand = result.hand_landmarks[0]
                if origin is None:
                    origin = np.array([hand[WRIST].x, hand[WRIST].y], dtype=np.float32)
                sequence.append(extract_features(hand, origin))

            if recording:
                status, color = f"ENREGISTREMENT ({len(sequence)} frames)", (0, 0, 255)
            else:
                status, color = "Pret (ESPACE pour enregistrer)", (0, 255, 0)
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            cv2.imshow("Enregistrement de geste", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Annule.")
                sequence = []
                break

            if key == ord(" "):
                if not recording:
                    recording = True
                    sequence = []
                    origin = None
                    print("Enregistrement demarre...")
                else:
                    break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()

    if len(sequence) < MIN_SEGMENT_LENGTH:
        print("Mouvement trop court, rien n'a ete enregistre.")
        return

    name = input("Nom du geste : ").strip()
    if not name:
        print("Nom vide, rien n'a ete enregistre.")
        return

    save_gesture(name, sequence)
    print(f"Geste '{name}' enregistre ({len(sequence)} frames).")


if __name__ == "__main__":
    main()
