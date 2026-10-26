"""Retire les posts dont l'image est une image de remplacement (lien mort remplace par une image fixe).

Methode : on calcule l'empreinte MD5 de chaque image utilisee. Une image strictement identique
partagee par au moins MIN_COPIES posts est consideree comme une image de remplacement
(les vraies images se repetent rarement plus de quelques fois).

Entrees : data/train.tsv, data/val.tsv, data/test.tsv  et  data/raw/images/<id>.jpg
Sorties : data/train.tsv, data/val.tsv, data/test.tsv   (nettoyes, sur place)
          data/removed_placeholders.tsv                  (liste des posts retires)
          data/raw/before_cleaning/*.tsv                 (copie des splits avant nettoyage)

Usage (depuis la racine du depot) : python src/clean_placeholders.py
Le script peut etre relance : s'il n'y a plus d'image de remplacement, il ne change rien.
"""
import hashlib
import shutil
from collections import Counter

import pandas as pd

from config import DATA_DIR, IMG_DIR, RAW_DIR

MIN_COPIES = 20
SPLITS = ["train", "val", "test"]


def md5(path):
    return hashlib.md5(path.read_bytes()).hexdigest()


def main():
    dfs = {s: pd.read_csv(DATA_DIR / f"{s}.tsv", sep="\t") for s in SPLITS}
    ids = pd.concat(dfs.values())["id"].astype(str).unique()
    hashes = {i: md5(IMG_DIR / f"{i}.jpg") for i in ids}

    counts = Counter(hashes.values())
    print("images les plus repetees :", [c for _, c in counts.most_common(5)])
    bad = {h for h, c in counts.items() if c >= MIN_COPIES}
    if not bad:
        print(f"Aucune image repetee {MIN_COPIES} fois ou plus : rien a retirer.")
        return

    backup = RAW_DIR / "before_cleaning"
    backup.mkdir(parents=True, exist_ok=True)
    removed = []
    for s, df in dfs.items():
        mask = df["id"].astype(str).map(hashes).isin(bad)
        removed.append(df[mask].assign(split=s))
        if not (backup / f"{s}.tsv").exists():
            shutil.copy(DATA_DIR / f"{s}.tsv", backup / f"{s}.tsv")
        df[~mask].to_csv(DATA_DIR / f"{s}.tsv", sep="\t", index=False)
        print(f"{s}: {int(mask.sum())} retires, {int((~mask).sum())} restants | "
              f"labels restants : {df[~mask]['2_way_label'].value_counts().to_dict()}")

    pd.concat(removed).to_csv(DATA_DIR / "removed_placeholders.tsv", sep="\t", index=False)
    print("liste des posts retires : data/removed_placeholders.tsv")


if __name__ == "__main__":
    main()
