"""E1 : baseline texte seul (TF-IDF + regression logistique).

Entrees : data/train.tsv, data/val.tsv, data/test.tsv
Sorties : results/e1_val_proba.npy, results/e1_test_proba.npy   (pour la fusion tardive E5)
          results/e1_metrics.json                                (parametres + metriques de l'experience)

Usage (depuis la racine du depot) : python src/e1_tfidf.py
"""
import datetime
import json

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)

from config import DATA_DIR, RESULTS_DIR, SEED

RESULTS_DIR.mkdir(exist_ok=True)

TFIDF = dict(ngram_range=(1, 2), min_df=2, max_features=50000, sublinear_tf=True)
C_GRID = [0.1, 0.3, 1, 3, 10]

tr, va, te = [pd.read_csv(DATA_DIR / f"{s}.tsv", sep="\t") for s in ["train", "val", "test"]]

# controle de fuite : titres identiques entre train et test
leak = set(tr["clean_title"]) & set(te["clean_title"])
print("titres communs train/test :", len(leak))

vec = TfidfVectorizer(**TFIDF)
Xtr = vec.fit_transform(tr["clean_title"])
Xva, Xte = vec.transform(va["clean_title"]), vec.transform(te["clean_title"])


def make_model(C):
    return LogisticRegression(C=C, max_iter=1000, class_weight="balanced", random_state=SEED)


# choix de C sur la validation uniquement
scores = [(C, f1_score(va["2_way_label"], make_model(C).fit(Xtr, tr["2_way_label"]).predict(Xva), average="macro"))
          for C in C_GRID]
best = max(scores, key=lambda x: x[1])
print("meilleur C (val) :", best)

clf = make_model(best[0]).fit(Xtr, tr["2_way_label"])
y, p = te["2_way_label"], clf.predict_proba(Xte)[:, 1]
pred = (p > 0.5).astype(int)
metrics = dict(
    acc=accuracy_score(y, pred),
    prec=precision_score(y, pred, average="macro"),
    rec=recall_score(y, pred, average="macro"),
    macro_f1=f1_score(y, pred, average="macro"),
    auc=roc_auc_score(y, p),
)
print("TEST  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
print(classification_report(y, pred))

# accuracy par type de faux (6-way) et titres de test deja vus dans le train
te["ok"] = (pred == y.values)
per_type = {int(k): round(float(v), 3) for k, v in te.groupby("6_way_label")["ok"].mean().items()}
print("accuracy par type (6-way) :", per_type)
n_shared = int(te["clean_title"].isin(set(tr["clean_title"])).sum())
print("lignes de test dont le titre est aussi dans le train :", n_shared)

# probabilites sauvegardees pour la fusion tardive (E5)
np.save(RESULTS_DIR / "e1_val_proba.npy", clf.predict_proba(Xva)[:, 1])
np.save(RESULTS_DIR / "e1_test_proba.npy", p)

# suivi d'experience : parametres et resultats dans un fichier
log = dict(
    experiment="E1 TF-IDF + LogisticRegression",
    date=datetime.datetime.now().isoformat(timespec="seconds"),
    seed=SEED,
    tfidf={k: (list(v) if isinstance(v, tuple) else v) for k, v in TFIDF.items()},
    best_C=best[0],
    val_macro_f1=round(float(best[1]), 4),
    test={k: round(float(v), 4) for k, v in metrics.items()},
    test_accuracy_by_6way=per_type,
    n_train=len(tr), n_val=len(va), n_test=len(te),
    test_titles_shared_with_train=n_shared,
)
with open(RESULTS_DIR / "e1_metrics.json", "w", encoding="utf-8") as f:
    json.dump(log, f, indent=2, ensure_ascii=False)
print("metriques ecrites dans results/e1_metrics.json")
