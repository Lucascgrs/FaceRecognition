from face_log import cluster_records, load_sightings, summarize_clusters


def main():
    records = load_sightings()
    if not records:
        print("Aucun enregistrement de visage (lance d'abord: python webcam.py --face).")
        return

    clusters = cluster_records(records)
    print(f"{len(records)} enregistrement(s) au total, {len(clusters)} personne(s) distincte(s) :\n")
    for label, count in sorted(summarize_clusters(clusters), key=lambda x: -x[1]):
        print(f"  {label}: {count} enregistrement(s)")


if __name__ == "__main__":
    main()
