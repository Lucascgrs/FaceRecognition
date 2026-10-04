import json
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "gestures_db.json"

WRIST = 0
MIDDLE_MCP = 9

# Vitesse (normalisee par la taille de la main) au-dela de laquelle on considere
# que la main a commence / continue a bouger.
MOTION_START_THRESHOLD = 0.035
MOTION_END_THRESHOLD = 0.015
MOTION_END_FRAMES = 8  # frames consecutives de faible mouvement pour clore un geste
MIN_SEGMENT_LENGTH = 10

# Distance DTW ponderee max pour valider un match. Valeur de depart estimee sur
# donnees synthetiques : a recalibrer avec les distances affichees en console
# (GestureRecognizer les imprime a chaque segment detecte).
MATCH_THRESHOLD = 0.006
LABEL_DISPLAY_FRAMES = 45  # ~1.5s a 30 fps


def extract_features(hand_landmarks, origin):
    """Vecteur de features pour une frame : forme de la main (relative au
    poignet, normalisee par la taille de la main) + position du poignet
    relative a `origin` (normalisee aussi), pour capturer a la fois la forme
    de la main et son deplacement dans l'espace."""
    points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks], dtype=np.float32)
    wrist = points[WRIST]
    scale = np.linalg.norm(points[MIDDLE_MCP] - wrist) or 1e-6

    shape = (points - wrist) / scale
    trajectory = (wrist[:2] - origin) / scale

    return np.concatenate([shape.flatten(), trajectory])


def to_motion(sequence, smoothing=5):
    """Convertit une sequence de positions absolues en sequence de deplacements
    (delta frame-a-frame). Un point qui ne bouge pas contribue ~0, quelle que
    soit sa position/configuration absolue : seul le mouvement compte, pas la
    pose statique des points qui ne participent pas au geste. Un lissage leger
    precede la derivee car le delta amplifie le tremblement de la detection."""
    seq = np.asarray(sequence, dtype=np.float32)
    if smoothing > 1 and len(seq) >= smoothing:
        kernel = np.ones(smoothing, dtype=np.float32) / smoothing
        seq = np.stack(
            [np.convolve(seq[:, d], kernel, mode="valid") for d in range(seq.shape[1])],
            axis=1,
        )
    return np.diff(seq, axis=0)


def template_weights(template):
    """Poids par dimension = a quel point cette dimension a bouge pendant le
    geste enregistre. Les points qui ne participent pas au geste (quasi
    immobiles dans le template) pesent presque rien dans la comparaison :
    leur bruit lors d'une reproduction ne doit pas faire echouer le match."""
    energy = np.mean(template ** 2, axis=0)
    return energy / (energy.sum() + 1e-8)


def dtw_distance(seq_a, seq_b, weights=None):
    a = np.asarray(seq_a, dtype=np.float32)
    b = np.asarray(seq_b, dtype=np.float32)
    n, m = len(a), len(b)

    diff = a[:, None, :] - b[None, :, :]
    sq_diff = diff ** 2
    if weights is not None:
        sq_diff = sq_diff * weights
    cost = np.sqrt(sq_diff.sum(axis=2))

    dtw = np.full((n + 1, m + 1), np.inf, dtype=np.float32)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        prev, row, c = dtw[i - 1], dtw[i], cost[i - 1]
        for j in range(1, m + 1):
            row[j] = c[j - 1] + min(row[j - 1], prev[j], prev[j - 1])

    return dtw[n, m] / (n + m)


def load_db():
    if not DB_PATH.exists():
        return {}
    with open(DB_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {name: np.array(seq, dtype=np.float32) for name, seq in raw.items()}


def save_gesture(name, sequence):
    db = load_db()
    db[name] = to_motion(sequence)
    raw = {n: seq.tolist() for n, seq in db.items()}
    with open(DB_PATH, "w", encoding="utf-8") as f:
        json.dump(raw, f)


def identify(sequence, db):
    if len(sequence) < MIN_SEGMENT_LENGTH or not db:
        return None, None

    motion = to_motion(sequence)
    best_name, best_dist = None, float("inf")
    for name, template in db.items():
        dist = dtw_distance(motion, template, weights=template_weights(template))
        if dist < best_dist:
            best_name, best_dist = name, dist

    if best_dist <= MATCH_THRESHOLD:
        return best_name, best_dist
    return None, best_dist


class GestureRecognizer:
    """Segmente automatiquement les mouvements de la main (debut/fin bases sur
    la vitesse du poignet) et les compare a la bibliotheque de gestes enregistres."""

    def __init__(self):
        self.db = load_db()
        self.sequence = []
        self.origin = None
        self.recording = False
        self.idle_count = 0
        self.last_wrist = None
        self.label = None
        self.label_ttl = 0

    def update(self, hands_landmarks):
        if self.label_ttl > 0:
            self.label_ttl -= 1
        else:
            self.label = None

        if not hands_landmarks:
            self._reset()
            return self.label

        hand = hands_landmarks[0]
        wrist = np.array([hand[WRIST].x, hand[WRIST].y], dtype=np.float32)
        mcp = np.array([hand[MIDDLE_MCP].x, hand[MIDDLE_MCP].y], dtype=np.float32)
        scale = float(np.linalg.norm(mcp - wrist)) or 1e-6

        speed = 0.0
        if self.last_wrist is not None:
            speed = float(np.linalg.norm(wrist - self.last_wrist)) / scale
        self.last_wrist = wrist

        if not self.recording and speed >= MOTION_START_THRESHOLD:
            self.recording = True
            self.origin = wrist.copy()
            self.sequence = []
            self.idle_count = 0

        if self.recording:
            self.sequence.append(extract_features(hand, self.origin))
            self.idle_count = self.idle_count + 1 if speed < MOTION_END_THRESHOLD else 0

            if self.idle_count >= MOTION_END_FRAMES:
                self._finish_segment()

        return self.label

    def _finish_segment(self):
        # Les dernieres MOTION_END_FRAMES frames n'ont servi qu'a confirmer la
        # fin du geste (main immobile) : elles ne font pas partie du mouvement
        # et fausseraient la comparaison si on les gardait.
        segment = self.sequence[:-MOTION_END_FRAMES]
        if len(segment) >= MIN_SEGMENT_LENGTH:
            name, dist = identify(segment, self.db)
            if dist is not None:
                print(f"[geste] segment de {len(segment)} frames -> "
                      f"{name or 'aucun match'} (distance={dist:.4f}, seuil={MATCH_THRESHOLD})")
            if name:
                self.label = name
                self.label_ttl = LABEL_DISPLAY_FRAMES
        self.recording = False
        self.sequence = []
        self.origin = None
        self.idle_count = 0

    def _reset(self):
        self.recording = False
        self.sequence = []
        self.origin = None
        self.idle_count = 0
        self.last_wrist = None
