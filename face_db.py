import json
from pathlib import Path

import cv2
import numpy as np

MODELS_DIR = Path(__file__).parent / "models"
DETECTOR_PATH = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
RECOGNIZER_PATH = MODELS_DIR / "face_recognition_sface_2021dec.onnx"
DB_PATH = Path(__file__).parent / "faces_db.json"

# Seuil de similarité cosinus recommandé par le modèle SFace pour valider un match.
COSINE_MATCH_THRESHOLD = 0.363


def create_detector(input_size=(320, 320), score_threshold=0.9):
    return cv2.FaceDetectorYN_create(str(DETECTOR_PATH), "", input_size, score_threshold=score_threshold)


def create_recognizer():
    return cv2.FaceRecognizerSF_create(str(RECOGNIZER_PATH), "")


def detect_faces(detector, frame):
    detector.setInputSize((frame.shape[1], frame.shape[0]))
    _, faces = detector.detect(frame)
    return faces if faces is not None else []


def get_embedding(recognizer, frame, face):
    aligned = recognizer.alignCrop(frame, face)
    return recognizer.feature(aligned)


def load_db():
    if not DB_PATH.exists():
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.array(vec, dtype=np.float32) for name, vec in raw.items()}


def save_db(db):
    raw = {name: vec.flatten().tolist() for name, vec in db.items()}
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f)


def identify(recognizer, embedding, db):
    best_name, best_score = None, -1.0
    for name, known_embedding in db.items():
        score = recognizer.match(
            embedding, known_embedding, cv2.FaceRecognizerSF_FR_COSINE
        )
        if score > best_score:
            best_name, best_score = name, score

    if best_score >= COSINE_MATCH_THRESHOLD:
        return best_name, best_score
    return None, best_score


def track_faces(frame, detector, recognizer, db, logger=None):
    faces = detect_faces(detector, frame)
    for face in faces:
        x, y, w, h = face[:4].astype(int)
        embedding = get_embedding(recognizer, frame, face)
        name, score = identify(recognizer, embedding, db)

        if logger is not None:
            logger.update(frame, face, embedding, name, score)

        label = f"{name} ({score:.2f})" if name else f"Inconnu ({score:.2f})"
        color = (0, 255, 0) if name else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, max(y - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )
