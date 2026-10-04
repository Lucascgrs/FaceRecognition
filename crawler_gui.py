import threading
from tkinter import filedialog

import customtkinter as ctk

from photo_crawler import PhotoCrawler

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class CrawlerApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Photo Crawler")
        self.geometry("780x720")
        self.minsize(640, 560)

        self.arret_event = None
        self.thread_crawl = None

        self._construire_interface()

    def _construire_interface(self):
        cadre_form = ctk.CTkFrame(self)
        cadre_form.pack(fill="x", padx=16, pady=16)
        cadre_form.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(cadre_form, text="URLs de depart (une par ligne)").grid(
            row=0, column=0, sticky="nw", padx=8, pady=6)
        self.textbox_urls = ctk.CTkTextbox(cadre_form, height=70)
        self.textbox_urls.insert("1.0", "https://datasulting.com")
        self.textbox_urls.grid(row=0, column=1, sticky="ew", padx=8, pady=6)

        ctk.CTkLabel(cadre_form, text="Domaines autorises (separes par une virgule)").grid(
            row=1, column=0, sticky="w", padx=8, pady=6)
        self.entry_domaines = ctk.CTkEntry(cadre_form)
        self.entry_domaines.insert(0, "datasulting.com")
        self.entry_domaines.grid(row=1, column=1, sticky="ew", padx=8, pady=6)

        ctk.CTkLabel(cadre_form, text="Dossier de sortie").grid(
            row=2, column=0, sticky="w", padx=8, pady=6)
        cadre_sortie = ctk.CTkFrame(cadre_form, fg_color="transparent")
        cadre_sortie.grid(row=2, column=1, sticky="ew", padx=8, pady=6)
        cadre_sortie.grid_columnconfigure(0, weight=1)
        self.entry_sortie = ctk.CTkEntry(cadre_sortie)
        self.entry_sortie.insert(0, "./telechargements")
        self.entry_sortie.grid(row=0, column=0, sticky="ew")
        ctk.CTkButton(cadre_sortie, text="Parcourir", width=90,
                      command=self._choisir_dossier).grid(row=0, column=1, padx=(8, 0))

        self.entry_max_pages = self._ligne(cadre_form, 3, "Max pages (par URL de depart)", "200")
        self.entry_delai = self._ligne(cadre_form, 4, "Delai entre pages (s)", "0.5")

        self.var_ignorer_robots = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cadre_form, text="Ignorer robots.txt",
                         variable=self.var_ignorer_robots).grid(
            row=5, column=0, columnspan=2, sticky="w", padx=8, pady=(10, 2))

        self.var_filtrer_visages = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cadre_form, text="Ne garder que les photos avec un visage detecte",
                         variable=self.var_filtrer_visages).grid(
            row=6, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        self.var_sightings = ctk.BooleanVar(value=False)
        ctk.CTkCheckBox(cadre_form, text="Enregistrer les visages detectes dans la base sightings",
                         variable=self.var_sightings).grid(
            row=7, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        cadre_boutons = ctk.CTkFrame(self, fg_color="transparent")
        cadre_boutons.pack(fill="x", padx=16)
        self.bouton_lancer = ctk.CTkButton(cadre_boutons, text="Lancer", command=self._lancer)
        self.bouton_lancer.pack(side="left")
        self.bouton_arreter = ctk.CTkButton(cadre_boutons, text="Arreter", state="disabled",
                                             fg_color="#8b2f2f", hover_color="#6e2424",
                                             command=self._arreter)
        self.bouton_arreter.pack(side="left", padx=8)

        cadre_stats = ctk.CTkFrame(self)
        cadre_stats.pack(fill="x", padx=16, pady=(12, 0))

        cadre_compteurs = ctk.CTkFrame(cadre_stats, fg_color="transparent")
        cadre_compteurs.pack(fill="x", padx=12, pady=(10, 6))
        self.label_pages = ctk.CTkLabel(cadre_compteurs, text="Pages visitees : 0")
        self.label_pages.pack(side="left", padx=(0, 20))
        self.label_images = ctk.CTkLabel(cadre_compteurs, text="Images telechargees : 0")
        self.label_images.pack(side="left", padx=(0, 20))
        self.label_erreurs = ctk.CTkLabel(cadre_compteurs, text="Erreurs : 0")
        self.label_erreurs.pack(side="left")

        self.cadre_urls_statut = ctk.CTkScrollableFrame(cadre_stats, height=120)
        self.cadre_urls_statut.pack(fill="x", padx=12, pady=(0, 10))
        self.labels_par_url = {}

        self.label_statut = ctk.CTkLabel(self, text="Pret")
        self.label_statut.pack(anchor="w", padx=16, pady=(8, 0))

        self.zone_log = ctk.CTkTextbox(self, state="disabled", font=("Consolas", 12))
        self.zone_log.pack(fill="both", expand=True, padx=16, pady=16)

    def _ligne(self, parent, ligne, libelle, valeur_defaut):
        ctk.CTkLabel(parent, text=libelle).grid(row=ligne, column=0, sticky="w", padx=8, pady=6)
        entry = ctk.CTkEntry(parent)
        entry.insert(0, valeur_defaut)
        entry.grid(row=ligne, column=1, sticky="ew", padx=8, pady=6)
        return entry

    def _choisir_dossier(self):
        dossier = filedialog.askdirectory()
        if dossier:
            self.entry_sortie.delete(0, "end")
            self.entry_sortie.insert(0, dossier)

    def _ajouter_log(self, message):
        self.zone_log.configure(state="normal")
        self.zone_log.insert("end", message + "\n")
        self.zone_log.see("end")
        self.zone_log.configure(state="disabled")

    def _log_depuis_thread(self, message):
        self.after(0, self._ajouter_log, message)

    def _construire_lignes_urls(self, urls):
        for widget in self.cadre_urls_statut.winfo_children():
            widget.destroy()
        self.labels_par_url = {}
        for i, url in enumerate(urls, start=1):
            label = ctk.CTkLabel(self.cadre_urls_statut, text=f"[{i}] {url} — en attente", anchor="w")
            label.pack(fill="x", padx=4, pady=2)
            self.labels_par_url[i] = label

    def _maj_stats(self, snapshot):
        total = snapshot["total"]
        self.label_pages.configure(text=f"Pages visitees : {total['pages_visitees']}")
        self.label_images.configure(text=f"Images telechargees : {total['images_telechargees']}")
        self.label_erreurs.configure(text=f"Erreurs : {total['erreurs']}")

        for i, etat in snapshot["par_url"].items():
            label = self.labels_par_url.get(i)
            if label is None:
                continue
            if etat["statut"] == "en cours":
                texte = f"[{i}] {etat['url']} — {etat['page_courante'] or '...'}"
            else:
                texte = f"[{i}] {etat['url']} — {etat['statut']}"
            label.configure(text=texte)

    def _progress_depuis_thread(self, snapshot):
        self.after(0, self._maj_stats, snapshot)

    def _lancer(self):
        urls = [u.strip() for u in self.textbox_urls.get("1.0", "end").splitlines() if u.strip()]
        domaines = [d.strip() for d in self.entry_domaines.get().split(",") if d.strip()]
        sortie = self.entry_sortie.get().strip()

        if not urls or not domaines or not sortie:
            self._ajouter_log("[erreur] Au moins une URL de depart, un domaine et un dossier de sortie sont obligatoires.")
            return

        try:
            max_pages = int(self.entry_max_pages.get())
            delai = float(self.entry_delai.get())
        except ValueError:
            self._ajouter_log("[erreur] Max pages doit etre un entier et Delai un nombre.")
            return

        self.arret_event = threading.Event()
        crawler = PhotoCrawler(
            urls_depart=urls,
            domaines_autorises=domaines,
            dossier_sortie=sortie,
            max_pages=max_pages,
            delai=delai,
            respecter_robots=not self.var_ignorer_robots.get(),
            filtrer_visages=self.var_filtrer_visages.get(),
            enregistrer_sightings=self.var_sightings.get(),
            arret_event=self.arret_event,
            on_log=self._log_depuis_thread,
            on_progress=self._progress_depuis_thread,
        )

        self._construire_lignes_urls(urls)

        self.bouton_lancer.configure(state="disabled")
        self.bouton_arreter.configure(state="normal")
        self.label_statut.configure(text="En cours...")
        self._ajouter_log(f"--- Demarrage du crawl sur {len(urls)} URL(s) de depart (en parallele) ---")

        self.thread_crawl = threading.Thread(target=self._executer, args=(crawler,), daemon=True)
        self.thread_crawl.start()

    def _executer(self, crawler):
        try:
            crawler.crawl()
        except Exception as erreur:
            self._log_depuis_thread(f"[erreur fatale] {erreur}")
        finally:
            self.after(0, self._crawl_termine)

    def _crawl_termine(self):
        self.bouton_lancer.configure(state="normal")
        self.bouton_arreter.configure(state="disabled")
        self.label_statut.configure(text="Termine")

    def _arreter(self):
        if self.arret_event is not None:
            self.arret_event.set()
            self.bouton_arreter.configure(state="disabled")
            self.label_statut.configure(text="Arret en cours...")


if __name__ == "__main__":
    app = CrawlerApp()
    app.mainloop()
