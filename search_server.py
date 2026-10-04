import io
import threading
import webbrowser

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request, send_from_directory
from PIL import Image, ImageOps

from face_db import COSINE_MATCH_THRESHOLD, create_detector, create_recognizer, detect_faces, get_embedding
from face_log import SIGHTINGS_DIR, load_sightings

HOTE = "127.0.0.1"
PORT = 5050

app = Flask(__name__)

SEUIL_DETECTION = 0.6  # plus permissif que le defaut (0.9) : les photos uploadees sont
                        # souvent moins nettes/bien cadrees qu'un flux webcam, mieux vaut
                        # quelques faux positifs qu'un visage manque completement.

detecteur = create_detector(score_threshold=SEUIL_DETECTION)
reconnaisseur = create_recognizer()


def charger_image(donnees):
    """Decode les octets uploades en image BGR (format OpenCV), en appliquant la
    rotation EXIF si presente : cv2.imdecode l'ignore, donc une photo prise en
    portrait sur un telephone peut arriver pivotee et etre manquee par le
    detecteur alors qu'elle s'affiche droite dans le navigateur."""
    try:
        image_pil = Image.open(io.BytesIO(donnees))
        image_pil = ImageOps.exif_transpose(image_pil).convert("RGB")
    except Exception:
        return None
    return cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)


def plus_grand_visage(image):
    visages = detect_faces(detecteur, image)
    if len(visages) == 0:
        return None, 0
    return max(visages, key=lambda v: v[2] * v[3]), len(visages)


@app.route("/")
def index():
    return render_template("recherche_visage.html", seuil_defaut=COSINE_MATCH_THRESHOLD)


@app.route("/api/rechercher", methods=["POST"])
def rechercher():
    fichier = request.files.get("photo")
    if fichier is None or fichier.filename == "":
        return jsonify(erreur="Aucune photo envoyee."), 400

    donnees = fichier.read()
    image = charger_image(donnees)
    if image is None:
        return jsonify(erreur="Fichier image illisible."), 400

    visage, nb_visages = plus_grand_visage(image)
    if visage is None:
        return jsonify(erreur="Aucun visage detecte dans cette photo."), 400

    embedding = get_embedding(reconnaisseur, image, visage)

    resultats = []
    for enregistrement in load_sightings():
        embedding_connu = np.array(enregistrement["embedding"], dtype=np.float32)
        score = reconnaisseur.match(embedding, embedding_connu, cv2.FaceRecognizerSF_FR_COSINE)
        resultats.append({
            "id": enregistrement["id"],
            "score": float(score),
            "nom": enregistrement.get("name"),
            "timestamp": enregistrement["timestamp"],
            "source": enregistrement.get("source"),
            "image_url": f"/sightings-image/{enregistrement['image']}",
        })

    resultats.sort(key=lambda r: r["score"], reverse=True)

    return jsonify(nb_visages_photo=nb_visages, resultats=resultats)


@app.route("/sightings-image/<path:chemin>")
def image_sighting(chemin):
    return send_from_directory(SIGHTINGS_DIR, chemin)


if __name__ == "__main__":
    url = f"http://{HOTE}:{PORT}/"
    threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    print(f"Ouverture de {url}")
    app.run(host=HOTE, port=PORT, debug=False, threaded=False)
