import json
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from face_db import COSINE_MATCH_THRESHOLD

SIGHTINGS_DIR = Path(__file__).parent / "sightings"
IMAGES_DIR = SIGHTINGS_DIR / "images"
SIGHTINGS_DB_PATH = SIGHTINGS_DIR / "sightings_db.json"

# Meme seuil que l'identification : au-dela, deux embeddings sont consideres
# comme le meme visage (utilise pour le suivi frame-a-frame et le clustering).
TRACK_MATCH_THRESHOLD = COSINE_MATCH_THRESHOLD
TRACK_TIMEOUT = 1.0  # secondes sans detection avant de considerer la personne partie
CONTEXT_MARGIN = 1.5  # elargissement de la bbox (fraction de sa taille) pour le contexte autour du visage


def load_sightings():
    if not SIGHTINGS_DB_PATH.exists():
        return []
    with open(SIGHTINGS_DB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_sightings(records):
    """Ecrit records de facon atomique (fichier temporaire + remplacement) pour
    qu'un lecteur concurrent (webcam.py, le crawler, search_server...) ne tombe
    jamais sur un JSON tronque, et qu'une ecriture interrompue ne corrompe pas
    le fichier existant."""
    SIGHTINGS_DIR.mkdir(parents=True, exist_ok=True)
    fichier_temporaire = SIGHTINGS_DB_PATH.with_suffix(".tmp")
    with open(fichier_temporaire, "w", encoding="utf-8") as f:
        json.dump(records, f)
    fichier_temporaire.replace(SIGHTINGS_DB_PATH)


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-6
    return float(np.dot(a, b) / denom)


def cluster_records(records, threshold=COSINE_MATCH_THRESHOLD):
    """Regroupe des enregistrements de sightings par personne en comparant leurs
    vecteurs deux a deux (similarite cosinus, meme metrique que l'identification
    faciale). `threshold` est la marge d'erreur : au-dessus, deux enregistrements
    sont consideres comme la meme personne. Chaque cluster garde un embedding
    moyen qui sert de representant pour les comparaisons suivantes."""
    clusters = []  # [{"embedding": np.ndarray, "records": [...]}]

    for record in records:
        embedding = np.array(record["embedding"], dtype=np.float32)

        best_idx, best_sim = None, -1.0
        for i, cluster in enumerate(clusters):
            sim = cosine_similarity(embedding, cluster["embedding"])
            if sim > best_sim:
                best_idx, best_sim = i, sim

        if best_idx is not None and best_sim >= threshold:
            cluster = clusters[best_idx]
            cluster["records"].append(record)
            n = len(cluster["records"])
            cluster["embedding"] = (cluster["embedding"] * (n - 1) + embedding) / n
        else:
            clusters.append({"embedding": embedding, "records": [record]})

    return clusters


def summarize_clusters(clusters):
    """Retourne [(label, nombre_d_enregistrements), ...] par personne. Le label
    est le nom identifie le plus frequent dans le cluster, sinon 'Inconnu #k'."""
    summary = []
    for i, cluster in enumerate(clusters):
        names = [r["name"] for r in cluster["records"] if r["name"]]
        label = max(set(names), key=names.count) if names else f"Inconnu #{i + 1}"
        summary.append((label, len(cluster["records"])))
    return summary


def crop_with_context(frame, face):
    h_frame, w_frame = frame.shape[:2]
    x, y, w, h = face[:4].astype(int)
    mx, my = int(w * CONTEXT_MARGIN), int(h * CONTEXT_MARGIN)
    x0, y0 = max(x - mx, 0), max(y - my, 0)
    x1, y1 = min(x + w + mx, w_frame), min(y + h + my, h_frame)
    return frame[y0:y1, x0:x1]


def record_sighting(records, frame, face, embedding, name, score, source=None):
    """Ajoute un enregistrement (meme format que FaceLogger) a `records` et sauvegarde
    la photo associee. N'ecrit pas records sur disque : appeler save_sightings(records)
    apres, pour ne pas re-ecrire tout le fichier a chaque visage."""
    sighting_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
    image_rel = f"images/{sighting_id}.jpg"
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SIGHTINGS_DIR / image_rel), crop_with_context(frame, face))

    record = {
        "id": sighting_id,
        "timestamp": time.time(),
        "name": name,
        "score": float(score),
        "embedding": embedding.flatten().tolist(),
        "image": image_rel,
    }
    if source:
        record["source"] = source
    records.append(record)
    return record


class FaceLogger:
    """Enregistre chaque nouvelle apparition de visage (pas chaque frame) dans la
    base de sightings : vecteur + photo (visage + contexte alentour), les deux
    relies par un identifiant commun. Un visage qui reste visible en continu
    n'est enregistre qu'une fois ; il doit disparaitre (plus aucune detection
    pendant TRACK_TIMEOUT secondes) puis revenir pour etre re-enregistre."""

    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.records = load_sightings()
        self.active_tracks = []  # [{"embedding": ..., "last_seen": ...}]
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    def update(self, frame, face, embedding, name, score):
        now = time.time()
        self.active_tracks = [
            t for t in self.active_tracks if now - t["last_seen"] <= TRACK_TIMEOUT
        ]

        track_idx = self._match_track(embedding)
        if track_idx is not None:
            self.active_tracks[track_idx]["last_seen"] = now
            self.active_tracks[track_idx]["embedding"] = embedding
            return

        self.active_tracks.append({"embedding": embedding, "last_seen": now})
        self._record_sighting(frame, face, embedding, name, score)

    def _match_track(self, embedding):
        best_idx, best_score = None, -1.0
        for i, track in enumerate(self.active_tracks):
            score = self.recognizer.match(
                embedding, track["embedding"], cv2.FaceRecognizerSF_FR_COSINE
            )
            if score > best_score:
                best_idx, best_score = i, score

        if best_score >= TRACK_MATCH_THRESHOLD:
            return best_idx
        return None

    def _record_sighting(self, frame, face, embedding, name, score):
        record_sighting(self.records, frame, face, embedding, name, score)
        save_sightings(self.records)
