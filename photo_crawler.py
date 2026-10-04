import argparse
import hashlib
import re
import threading
import time
import urllib.robotparser
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import cv2
import numpy as np
import requests
from bs4 import BeautifulSoup

from face_db import create_detector, create_recognizer, detect_faces, get_embedding, identify, load_db
from face_log import load_sightings, record_sighting, save_sightings

USER_AGENT = "PhotoCrawlerBot/1.0 (+usage personnel de test)"
EXTENSIONS_IMAGE = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".bmp", ".svg", ".avif"}
CSS_URL_RE = re.compile(r"url\(\s*['\"]?([^'\")]+)['\"]?\s*\)")


class PhotoCrawler:
    """Crawle chaque URL de depart dans son propre thread (en parallele), pour
    aller plus vite qu'un parcours sequentiel. Tout etat partage entre threads
    (dedup, compteurs, cache robots.txt, sightings) est protege par self._lock.
    Les modeles de detection de visage ne sont eux jamais partages entre threads
    (cv2 FaceDetectorYN/FaceRecognizerSF ne sont pas thread-safe) : chaque thread
    cree ses propres instances au premier besoin, via threading.local."""

    def __init__(self, urls_depart, domaines_autorises, dossier_sortie,
                 max_pages=200, delai=0.5, respecter_robots=True, timeout=10,
                 filtrer_visages=False, enregistrer_sightings=False,
                 arret_event=None, on_log=print, on_progress=None):
        self.urls_depart = list(urls_depart)
        self.domaines_autorises = list(domaines_autorises)
        self.dossier_sortie = Path(dossier_sortie)
        self.max_pages = max_pages
        self.delai = delai
        self.timeout = timeout
        self.respecter_robots = respecter_robots
        self.filtrer_visages = filtrer_visages
        self.enregistrer_sightings = enregistrer_sightings
        self.arret_event = arret_event
        self.on_log = on_log
        self.on_progress = on_progress or (lambda stats: None)

        self.dossier_sortie.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

        self._lock = threading.Lock()
        self._local = threading.local()

        self.images_vues = set()
        # Empreintes (prefixe de hash) des fichiers deja presents dans le dossier de sortie,
        # pour ne pas re-telecharger une image issue d'un run precedent ou d'une autre URL de depart.
        self.empreintes_existantes = {
            p.name.split("_", 1)[0] for p in self.dossier_sortie.glob("*_*")
        }
        self.nb_visages_enregistres = 0
        self.robots_par_domaine = {}

        self.total_stats = {"pages_visitees": 0, "images_telechargees": 0, "erreurs": 0}
        self.stats_par_url = {
            i: {"url": url, "page_courante": "", "statut": "en attente"}
            for i, url in enumerate(self.urls_depart, start=1)
        }

        self.db_connue = None
        if self.filtrer_visages or self.enregistrer_sightings:
            self._log("Chargement de la base des visages connus...")
            self.db_connue = load_db()

        self.sightings = load_sightings() if self.enregistrer_sightings else None

    def _log(self, message):
        self.on_log(message)

    def _notifier(self):
        with self._lock:
            snapshot = {
                "total": dict(self.total_stats),
                "par_url": {i: dict(s) for i, s in self.stats_par_url.items()},
            }
        self.on_progress(snapshot)

    def _arret_demande(self):
        return self.arret_event is not None and self.arret_event.is_set()

    def _modeles_visage(self):
        """Instances de detection/reconnaissance propres au thread courant (voir
        docstring de la classe : ces objets cv2 ne peuvent pas etre partages)."""
        if not hasattr(self._local, "detecteur"):
            self._local.detecteur = create_detector()
            self._local.reconnaisseur = create_recognizer()
        return self._local.detecteur, self._local.reconnaisseur

    def _robots_pour(self, url):
        if not self.respecter_robots:
            return None

        parsed = urlparse(url)
        with self._lock:
            parser = self.robots_par_domaine.get(parsed.netloc)
        if parser is not None:
            return parser

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = urllib.robotparser.RobotFileParser()
        try:
            resp = self.session.get(robots_url, timeout=self.timeout)
            parser.parse(resp.text.splitlines() if resp.status_code == 200 else [])
        except requests.RequestException:
            parser.parse([])

        with self._lock:
            parser = self.robots_par_domaine.setdefault(parsed.netloc, parser)
        return parser

    def _url_autorisee(self, url):
        if not any(domaine in url for domaine in self.domaines_autorises):
            return False
        robots = self._robots_pour(url)
        if robots is not None and not robots.can_fetch(USER_AGENT, url):
            return False
        return True

    def _extraire_liens_et_images(self, html, url_page):
        soup = BeautifulSoup(html, "html.parser")

        liens = set()
        for balise_a in soup.find_all("a", href=True):
            lien_absolu = urljoin(url_page, balise_a["href"].strip())
            liens.add(lien_absolu.split("#")[0])

        images = set()
        for balise_img in soup.find_all("img"):
            for attribut in ("src", "data-src", "data-lazy-src"):
                if balise_img.get(attribut):
                    images.add(urljoin(url_page, balise_img[attribut].strip()))
            if balise_img.get("srcset"):
                for candidat in balise_img["srcset"].split(","):
                    src = candidat.strip().split(" ")[0]
                    if src:
                        images.add(urljoin(url_page, src))

        for balise_source in soup.find_all("source"):
            if balise_source.get("srcset"):
                for candidat in balise_source["srcset"].split(","):
                    src = candidat.strip().split(" ")[0]
                    if src:
                        images.add(urljoin(url_page, src))

        for correspondance in CSS_URL_RE.findall(html):
            correspondance = correspondance.strip()
            if "\\/" in correspondance:
                continue
            if any(correspondance.lower().endswith(ext) for ext in EXTENSIONS_IMAGE):
                images.add(urljoin(url_page, correspondance))

        return liens, images

    def _nom_fichier_local(self, url_image, empreinte, content_type=None):
        chemin = urlparse(url_image).path
        nom = Path(chemin).name

        extension = Path(nom).suffix.lower()
        if extension not in EXTENSIONS_IMAGE:
            if content_type and "/" in content_type:
                extension = "." + content_type.split("/")[-1].split(";")[0]
                extension = extension.replace("jpeg", "jpg")
            else:
                extension = ".jpg"
            nom = Path(nom).stem or "image"
            nom = f"{nom}{extension}"

        if not nom or nom == extension:
            nom = f"image{extension}"

        return f"{empreinte}_{nom}"

    def _traiter_visages(self, image, url_image):
        """Detecte les visages de `image`. Si enregistrer_sightings est actif,
        enregistre chacun (empreinte + photo recadree) dans la base sightings,
        qu'il corresponde ou non a un visage deja connu."""
        detecteur, reconnaisseur = self._modeles_visage()
        visages = detect_faces(detecteur, image)

        if self.enregistrer_sightings and len(visages) > 0:
            a_enregistrer = []
            for visage in visages:
                embedding = get_embedding(reconnaisseur, image, visage)
                nom, score = identify(reconnaisseur, embedding, self.db_connue)
                a_enregistrer.append((visage, embedding, nom, score))

            with self._lock:
                for visage, embedding, nom, score in a_enregistrer:
                    record_sighting(self.sightings, image, visage, embedding, nom, score, source=url_image)
                save_sightings(self.sightings)
                self.nb_visages_enregistres += len(a_enregistrer)

        return visages

    def _telecharger_image(self, url_image):
        with self._lock:
            if url_image in self.images_vues:
                return
            self.images_vues.add(url_image)

        if not url_image.lower().startswith(("http://", "https://")):
            return

        empreinte = hashlib.md5(url_image.encode("utf-8")).hexdigest()[:10]
        with self._lock:
            if empreinte in self.empreintes_existantes:
                return

        besoin_detection = self.filtrer_visages or self.enregistrer_sightings
        chemin_local = None
        try:
            reponse = self.session.get(url_image, timeout=self.timeout, stream=True)
            reponse.raise_for_status()

            content_type = reponse.headers.get("Content-Type", "")
            if "image" not in content_type and not any(
                url_image.lower().split("?")[0].endswith(ext) for ext in EXTENSIONS_IMAGE
            ):
                return

            if besoin_detection:
                contenu = reponse.content
                if not contenu:
                    return
                image = cv2.imdecode(np.frombuffer(contenu, dtype=np.uint8), cv2.IMREAD_COLOR)
                visages = self._traiter_visages(image, url_image) if image is not None else []

                if self.filtrer_visages and len(visages) == 0:
                    return

                nom_fichier = self._nom_fichier_local(url_image, empreinte, content_type)
                chemin_local = self.dossier_sortie / nom_fichier
                with open(chemin_local, "wb") as f:
                    f.write(contenu)
            else:
                nom_fichier = self._nom_fichier_local(url_image, empreinte, content_type)
                chemin_local = self.dossier_sortie / nom_fichier
                with open(chemin_local, "wb") as f:
                    for bloc in reponse.iter_content(8192):
                        f.write(bloc)
        except (requests.RequestException, OSError, cv2.error):
            with self._lock:
                self.total_stats["erreurs"] += 1
            self._notifier()
            if chemin_local is not None and chemin_local.exists():
                chemin_local.unlink()
            return

        with self._lock:
            self.empreintes_existantes.add(empreinte)
            self.total_stats["images_telechargees"] += 1
        self._notifier()

    def _crawl_une_url(self, index, url_depart):
        file_attente = deque([url_depart])
        pages_decouvertes = {url_depart}
        pages_traitees = 0

        while file_attente and pages_traitees < self.max_pages:
            if self._arret_demande():
                return "arretee"

            url_page = file_attente.popleft()

            if not self._url_autorisee(url_page):
                continue

            pages_traitees += 1
            with self._lock:
                self.total_stats["pages_visitees"] += 1
                self.stats_par_url[index]["page_courante"] = url_page
            self._notifier()

            try:
                reponse = self.session.get(url_page, timeout=self.timeout)
                reponse.raise_for_status()
            except requests.RequestException:
                with self._lock:
                    self.total_stats["erreurs"] += 1
                self._notifier()
                continue

            content_type = reponse.headers.get("Content-Type", "")
            if "text/html" not in content_type:
                continue

            liens, images = self._extraire_liens_et_images(reponse.text, url_page)

            for url_image in images:
                if self._arret_demande():
                    return "arretee"
                self._telecharger_image(url_image)

            for lien in liens:
                if lien not in pages_decouvertes and self._url_autorisee(lien):
                    pages_decouvertes.add(lien)
                    file_attente.append(lien)

            time.sleep(self.delai)

        return "toutes les pages explorees" if not file_attente else "max de pages atteint"

    def _crawl_worker(self, index, url_depart):
        with self._lock:
            self.stats_par_url[index]["statut"] = "en cours"
        self._notifier()
        self._log(f"--- [{index}/{len(self.urls_depart)}] Demarrage sur {url_depart} ---")

        raison = self._crawl_une_url(index, url_depart)

        with self._lock:
            self.stats_par_url[index]["statut"] = raison
            self.stats_par_url[index]["page_courante"] = ""
        self._notifier()
        self._log(f"--- [{index}/{len(self.urls_depart)}] Termine : {raison} ---")

    def crawl(self):
        threads = [
            threading.Thread(target=self._crawl_worker, args=(index, url_depart), daemon=True)
            for index, url_depart in enumerate(self.urls_depart, start=1)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        message = (f"\nTermine : {self.total_stats['pages_visitees']} pages visitees, "
                   f"{self.total_stats['images_telechargees']} images telechargees, "
                   f"{self.total_stats['erreurs']} erreurs, dans {self.dossier_sortie}")
        if self.enregistrer_sightings:
            message += f", {self.nb_visages_enregistres} visages enregistres dans sightings"
        self._log(message)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Crawle un ou plusieurs sites (en parallele) et telecharge leurs photos.")
    parser.add_argument("--url", required=True, nargs="+",
                         help="Une ou plusieurs URL de depart, ex: --url https://a.com https://b.com")
    parser.add_argument("--domaine", required=True, nargs="+",
                         help="Une ou plusieurs sous-chaines que l'URL doit contenir pour etre suivie")
    parser.add_argument("--sortie", default="./telechargements", help="Dossier de sortie des images")
    parser.add_argument("--max-pages", type=int, default=200, help="Nombre maximum de pages a visiter par URL de depart")
    parser.add_argument("--delai", type=float, default=0.5, help="Delai en secondes entre chaque requete de page, par thread")
    parser.add_argument("--ignorer-robots", action="store_true", help="Ne pas respecter robots.txt")
    parser.add_argument("--filtrer-visages", action="store_true",
                         help="Ne garder que les photos ou au moins un visage est detecte")
    parser.add_argument("--sightings", action="store_true",
                         help="Enregistrer les visages detectes (empreinte + photo) dans la base sightings")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    crawler = PhotoCrawler(
        urls_depart=args.url,
        domaines_autorises=args.domaine,
        dossier_sortie=args.sortie,
        max_pages=args.max_pages,
        delai=args.delai,
        respecter_robots=not args.ignorer_robots,
        filtrer_visages=args.filtrer_visages,
        enregistrer_sightings=args.sightings,
    )
    crawler.crawl()
