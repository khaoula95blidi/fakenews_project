"""Configuration commune : chemins (relatifs a la racine du depot) et graine aleatoire."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"               # splits fixes : train.tsv, val.tsv, test.tsv
RAW_DIR = DATA_DIR / "raw"             # donnees lourdes (non versionnees) : TSV d'origine et images
IMG_DIR = RAW_DIR / "images"           # images 224x224, une par post : <id>.jpg
RESULTS_DIR = ROOT / "results"         # probabilites, metriques

SEED = 42          # graine unique pour tout le projet
IMG_SIZE = 224     # taille des images apres redimensionnement
