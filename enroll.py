import sys

import cv2

from face_db import create_detector, create_recognizer, get_embedding, load_db, save_db


def main():
    if len(sys.argv) != 2:
        print("Usage: python enroll.py <nom>")
        sys.exit(1)
    name = sys.argv[1]

    detector = create_detector()
    recognizer = create_recognizer()
    db = load_db()

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not capture.isOpened():
        raise RuntimeError("Impossible d'ouvrir la webcam")

    print("Positionne ton visage face a la camera. Appuie sur ESPACE pour capturer, 'q' pour annuler.")

    try:
        while True:
            ret, frame = capture.read()
            if not ret:
                break

            detector.setInputSize((frame.shape[1], frame.shape[0]))
            _, faces = detector.detect(frame)

            display = frame.copy()
            if faces is not None:
                for face in faces:
                    x, y, w, h = face[:4].astype(int)
                    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow("Enrollment", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Annule.")
                break

            if key == ord(" "):
                if faces is None or len(faces) == 0:
                    print("Aucun visage detecte, reessaie.")
                    continue
                if len(faces) > 1:
                    print("Plusieurs visages detectes, assure-toi d'etre seul devant la camera.")
                    continue

                embedding = get_embedding(recognizer, frame, faces[0])
                db[name] = embedding
                save_db(db)
                print(f"Visage de '{name}' enregistre.")
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
