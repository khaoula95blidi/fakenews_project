"""Suivi des expériences avec MLflow : projet Fakeddit, détection précoce (titre + image).

Un seul fichier, à mettre à la racine du projet. Il sert à trois choses :

    python suivi_mlflow.py                      enregistre la référence « tout vrai » et E1, E2, E3
                                                (chiffres des diapos ; ne touche pas à un run déjà présent)
    python suivi_mlflow.py --json resultat.json enregistre un résultat calculé ailleurs (Kaggle)
    python suivi_mlflow.py --ui                 ouvre l'interface MLflow dans le navigateur

Dans un notebook (E4, E5, robustesse...) :

    from suivi_mlflow import evaluer, log_run
    m = evaluer(y_test, y_pred, y_proba, types=types_test)
    log_run("E4", "fusion_precoce", params={"entree": "titre + image"}, metrics=m)

Relancer une expérience avec le même nom (étape + nom) REMPLACE son run précédent : pas de doublon.

Tout est stocké à côté de ce fichier (mlflow.db et mlartifacts/), quel que soit le dossier
d'où l'on lance Python : le notebook, le script et l'interface voient donc les mêmes runs.
"""
import argparse
import json
import os
import subprocess
import sys
import unicodedata
from pathlib import Path

# Un seul thread de calcul pour numpy (OpenBLAS), à fixer AVANT d'importer numpy. Sinon chaque
# processus réserve un tampon mémoire par cœur et, avec l'interface MLflow ouverte en même temps,
# la machine manque de mémoire : « OpenBLAS error: Memory allocation still failed ».
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

RACINE = Path(__file__).resolve().parent
URI_BASE = f"sqlite:///{(RACINE / 'mlflow.db').as_posix()}"
DOSSIER_FICHIERS = RACINE / "mlartifacts"
NOM_EXPERIENCE = "fakeddit-detection-precoce"

# Paramètres identiques pour toutes les expériences (on peut les remplacer dans log_run).
PARAMS_COMMUNS = {
    "jeu": "Fakeddit",
    "tache": "2 classes (vrai / faux)",
    "seed": 42,
    "n_train": 7432,
    "n_val": 1403,
    "n_test": 1396,
}

mlflow.set_tracking_uri(URI_BASE)


def _preparer():
    """Crée l'expérience MLflow du projet si besoin, puis la sélectionne."""
    if mlflow.get_experiment_by_name(NOM_EXPERIENCE) is None:
        mlflow.create_experiment(NOM_EXPERIENCE, artifact_location=DOSSIER_FICHIERS.as_uri())
    mlflow.set_experiment(NOM_EXPERIENCE)


def _slug(texte):
    """« Image manipulée » -> « image_manipulee » (noms de métriques sans accent ni espace)."""
    sans_accent = unicodedata.normalize("NFKD", str(texte)).encode("ascii", "ignore").decode()
    return "_".join(sans_accent.lower().split())


def evaluer(y_vrai, y_pred, y_proba=None, types=None):
    """Calcule les mesures du projet : exactitude, F1 macro, AUC et exactitude par type.

    y_proba : probabilité de la classe 1, celle que renvoie clf.predict_proba(X)[:, 1] (pour l'AUC).
    types : le type de chaque publication (liste de même longueur que y_vrai), pour l'exactitude
    par type de contenu.
    """
    y_vrai, y_pred = np.asarray(y_vrai), np.asarray(y_pred)
    res = {
        "accuracy": accuracy_score(y_vrai, y_pred),
        "f1_macro": f1_score(y_vrai, y_pred, average="macro"),
    }
    if y_proba is not None:
        res["auc"] = roc_auc_score(y_vrai, y_proba)
    if types is not None:
        types = np.asarray(types)
        for t in np.unique(types):
            masque = types == t
            res[f"acc_type_{_slug(t)}"] = accuracy_score(y_vrai[masque], y_pred[masque])
            res[f"n_type_{_slug(t)}"] = int(masque.sum())
    return {k: float(v) for k, v in res.items()}


def _ids_existants(nom_run):
    """Identifiants des runs actifs qui portent déjà ce nom."""
    trouves = mlflow.search_runs(
        experiment_names=[NOM_EXPERIENCE], filter_string=f"tags.mlflow.runName = '{nom_run}'"
    )
    return trouves["run_id"].tolist() if len(trouves) else []


def _existe(nom_run):
    return len(_ids_existants(nom_run)) > 0


def log_run(etape, nom, params=None, metrics=None, fichiers=(), modele=None, tags=None, si_existe="remplacer"):
    """Enregistre une expérience : paramètres, mesures, fichiers (courbes, tableaux) et modèle.

    etape : « E1 », « E4 », « RQ3 »... ; nom : description courte, par exemple « fusion_precoce ».
    modele : un objet scikit-learn (par exemple la régression logistique), facultatif.
    si_existe : que faire si un run porte déjà ce nom : « remplacer » (par défaut, l'ancien est
    supprimé) ou « ajouter » (on garde les deux).
    """
    _preparer()
    nom_run = f"{etape}_{nom}"
    if si_existe == "remplacer":
        for ancien in _ids_existants(nom_run):
            mlflow.delete_run(ancien)
    with mlflow.start_run(run_name=nom_run) as run:
        mlflow.set_tags({"etape": etape, **(tags or {})})
        mlflow.log_params({**PARAMS_COMMUNS, **(params or {})})
        mlflow.log_metrics({k: float(v) for k, v in (metrics or {}).items() if v is not None})
        for f in fichiers:
            mlflow.log_artifact(str(f))
        if modele is not None:
            mlflow.sklearn.log_model(modele, name="modele")
    print(f"Enregistré : {nom_run} (run {run.info.run_id[:8]})")


# Résultats de E1 à E3 déjà obtenus (jeu de test, 1 396 publications, graine 42).
AVERTISSEMENT_TYPES = (
    "Types de la tache a 6 classes : correspondance des etiquettes en cours de verification ; "
    "fausse_connexion : 51 publications seulement ; imposteur : 21, non retenu."
)
RESULTATS_DEJA_OBTENUS = [
    {
        "etape": "Base", "nom": "tout_vrai",
        "params": {"entree": "aucune", "representation": "aucune", "classifieur": "repond toujours vrai"},
        "metrics": {"accuracy": 0.503, "f1_macro": 0.335, "auc": 0.500},
    },
    {
        "etape": "E1", "nom": "tfidf_lr",
        "params": {"entree": "titre", "representation": "TF-IDF", "classifieur": "LogisticRegression"},
        "metrics": {
            "accuracy": 0.751, "f1_macro": 0.750, "auc": 0.822,
            "acc_type_image_manipulee": 0.833, "acc_type_vrai": 0.791,
            "acc_type_fausse_connexion": 0.667, "acc_type_trompeur": 0.618, "acc_type_satire": 0.540,
        },
    },
    {
        "etape": "E2", "nom": "distilbert_gele_lr",
        "params": {"entree": "titre", "representation": "DistilBERT (gele)", "classifieur": "LogisticRegression"},
        "metrics": {
            "accuracy": 0.801, "f1_macro": 0.801, "auc": 0.8615,
            "acc_type_image_manipulee": 0.942, "acc_type_vrai": 0.803,
            "acc_type_fausse_connexion": 0.706, "acc_type_trompeur": 0.675, "acc_type_satire": 0.603,
        },
    },
    {
        "etape": "E3", "nom": "resnet18_gele_lr",
        "params": {"entree": "image", "representation": "ResNet-18 (gele)", "classifieur": "LogisticRegression"},
        "metrics": {
            "accuracy": 0.699, "f1_macro": 0.699, "auc": 0.7547,
            "acc_type_image_manipulee": 0.778, "acc_type_vrai": 0.736,
            "acc_type_fausse_connexion": 0.824, "acc_type_trompeur": 0.486, "acc_type_satire": 0.587,
        },
    },
]


def enregistrer_resultats_deja_obtenus():
    """Enregistre la référence et E1 à E3 ; ne recrée pas un run qui existe déjà."""
    _preparer()
    for r in RESULTATS_DEJA_OBTENUS:
        nom_run = f"{r['etape']}_{r['nom']}"
        if _existe(nom_run):
            print(f"Déjà présent, ignoré : {nom_run}")
            continue
        tags = {"source": "retro-enregistre depuis les resultats obtenus"}
        if r["etape"] != "Base":
            tags["avertissement_types"] = AVERTISSEMENT_TYPES
        log_run(r["etape"], r["nom"], r["params"], r["metrics"], tags=tags)


def enregistrer_json(chemin):
    """Enregistre un résultat écrit en JSON (par exemple calculé sur Kaggle).

    Format : {"etape": "E4", "nom": "fusion_precoce", "params": {...}, "metrics": {...}}
    """
    r = json.loads(Path(chemin).read_text(encoding="utf-8"))
    log_run(r["etape"], r["nom"], r.get("params"), r.get("metrics"), tags=r.get("tags"))


def ouvrir_interface(port=5000):
    print(f"Interface MLflow : http://127.0.0.1:{port}  (Ctrl+C pour arrêter)")
    # Un seul processus serveur (4 par défaut) : largement suffisant en local et bien plus léger.
    subprocess.run(
        [sys.executable, "-m", "mlflow", "ui", "--backend-store-uri", URI_BASE,
         "--port", str(port), "--workers", "1"]
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--json", help="fichier JSON d'un résultat à enregistrer")
    parser.add_argument("--ui", action="store_true", help="ouvre l'interface MLflow")
    parser.add_argument("--port", type=int, default=5000, help="port de l'interface (5000 par défaut)")
    args = parser.parse_args()
    if args.ui:
        ouvrir_interface(args.port)
    elif args.json:
        enregistrer_json(args.json)
    else:
        enregistrer_resultats_deja_obtenus()