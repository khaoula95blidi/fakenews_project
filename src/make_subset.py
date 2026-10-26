"""Construit le sous-ensemble Fakeddit (texte + image) a partir des TSV d'origine.

Entrees : data/raw/multimodal_train.tsv, multimodal_validate.tsv, multimodal_test_public.tsv
Sorties : data/train.tsv, data/val.tsv, data/test.tsv  (splits fixes)
          data/raw/images/<id>.jpg                      (images 224x224)

Usage (depuis la racine du depot) : python src/make_subset.py
Les images deja telechargees sont conservees : on peut relancer sans risque.
"""
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import pandas as pd
import requests
from PIL import Image

from config import DATA_DIR, IMG_DIR, IMG_SIZE, RAW_DIR, SEED

warnings.filterwarnings("ignore")

TEST = False   # True = petit test rapide : ecrit data/tmp_*.tsv sans toucher aux vrais splits

SPLITS = {     # fichier d'origine -> (nom de sortie, exemples par classe)
    "multimodal_train.tsv":       ("train", 4000),
    "multimodal_validate.tsv":    ("val",   750),
    "multimodal_test_public.tsv": ("test",  750),
}
if TEST:
    SPLITS = {k: (v[0], 100) for k, v in SPLITS.items()}

HEADERS = {"User-Agent": "Mozilla/5.0"}   # Imgur refuse les requetes sans User-Agent
WORKERS = 6                               # peu de threads : Imgur limite le debit
TIMEOUT = 20


def fetch(row):
    """Telecharge une image et la redimensionne. Retourne l'id, ou None si elle est indisponible."""
    path = IMG_DIR / f"{row['id']}.jpg"
    if path.exists():                      # deja telechargee : on la garde
        return row["id"]
    for attempt in range(3):
        try:
            r = requests.get(row["image_url"], timeout=TIMEOUT, headers=HEADERS)
        except requests.RequestException:
            time.sleep(1 + attempt)        # probleme reseau : on reessaie
            continue
        if r.status_code in (429, 500, 502, 503, 504):
            time.sleep(2 * (attempt + 1))  # limite de debit ou erreur serveur
            continue
        if r.status_code != 200 or "imgur.com/removed" in r.url:
            return None                    # lien mort ou image supprimee
        try:
            Image.open(BytesIO(r.content)).convert("RGB").resize((IMG_SIZE, IMG_SIZE)).save(path, quality=85)
            return row["id"]
        except Exception:
            return None                    # le fichier n'est pas une image
    return None


def main():
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    for fname, (out, n) in SPLITS.items():
        df = pd.read_csv(RAW_DIR / fname, sep="\t").dropna(subset=["clean_title", "image_url"])
        df = df.groupby("2_way_label").sample(n, random_state=SEED)
        rows = [r for _, r in df.iterrows()]
        with ThreadPoolExecutor(WORKERS) as ex:
            ids = set(i for i in ex.map(fetch, rows) if i)
        df = df[df["id"].isin(ids)]

        out_path = DATA_DIR / (("tmp_" if TEST else "") + out + ".tsv")
        df.to_csv(out_path, sep="\t", index=False)
        print(out, len(df), "exemples valides |",
              "2-way :", df["2_way_label"].value_counts().to_dict(), "|",
              "6-way :", df["6_way_label"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
