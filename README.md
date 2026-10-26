# Détection multimodale précoce des fake news (Fakeddit)

*Early Multimodal Fake News Detection Using Text and Images* : projet académique de recherche.

## Objectif

Comparer trois familles de modèles (texte seul, image seule, fusion texte + image) pour savoir **si, quand et comment l'image apporte un gain** dans la détection de fake news.

« Précoce » désigne ici la détection à partir du **contenu initial d'une publication (titre + image)**, sans métadonnées de popularité ni commentaires, qui n'existent qu'après la publication.

## Données

[Fakeddit](https://arxiv.org/abs/1911.03854) (Nakamura et al., 2020) : posts Reddit avec titre, image et labels (2, 3 ou 6 classes). Dépôt officiel : <https://github.com/entitize/Fakeddit>.

- Tâche principale : classification binaire (`2_way_label` : 1 = vrai, 0 = faux).
- Sous-ensemble équilibré construit par `src/make_subset.py` (graine 42) : 10 711 posts avec image (7 792 train, 1 469 val, 1 450 test) avant nettoyage.
- `src/clean_placeholders.py` retire ensuite les posts dont l'image est une image de remplacement (lien mort, 480 posts). Les effectifs finaux sont dans `results/e1_metrics.json`.
- Les splits fixes `data/train.tsv`, `data/val.tsv`, `data/test.tsv` sont versionnés. Ne pas les régénérer.

## Structure

```
.
├── README.md
├── requirements.txt
├── data/
│   ├── train.tsv  val.tsv  test.tsv        # splits fixes (versionnés)
│   └── raw/                                # NON versionné
│       ├── multimodal_*.tsv                # fichiers d'origine Fakeddit
│       └── images/                         # images 224x224, <id>.jpg
├── src/
│   ├── config.py                           # chemins et graine (SEED = 42)
│   ├── make_subset.py                      # construit les splits et télécharge les images
│   ├── clean_placeholders.py               # retire les images de remplacement (liens morts)
│   └── e1_tfidf.py                         # E1 : TF-IDF + régression logistique
├── notebooks/
│   └── eda_fakeddit.ipynb                  # analyse exploratoire (Kaggle)
└── results/                                # probabilités et métriques de chaque expérience
```

## Reproduire

```bash
pip install -r requirements.txt

# 1. Télécharger les TSV multimodaux de Fakeddit (voir le dépôt officiel)
#    et les placer dans data/raw/
# 2. (Optionnel) reconstruire les splits et les images, puis les nettoyer
python src/make_subset.py
python src/clean_placeholders.py
# 3. Lancer la baseline E1
python src/e1_tfidf.py
```

Chaque expérience écrit ses paramètres et ses résultats dans `results/` (par exemple `e1_metrics.json`).

## Résultats (jeu de test)

| Modèle | Entrées | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| Majoritaire (tout « vrai ») | : | ≈ 0,51 | ≈ 0,34 | 0,500 |
| E1 TF-IDF + LR | Texte | voir `results/e1_metrics.json` | | |
| E2 DistilBERT | Texte | à venir | | |
| E3 ResNet-18 | Image | à venir | | |
| E4 Early fusion | Texte + image | à venir | | |
| E5 Late fusion | Texte + image | à venir | | |

## Limites connues

- Les labels Fakeddit sont dérivés du subreddit d'origine (supervision distante) : un modèle peut apprendre le style d'une communauté plutôt que la désinformation.
- Les titres de la classe « image manipulée » sont très courts et génériques, ce qui facilite sa détection par le texte sans rapport avec l'image.
- Le sous-ensemble ne représente qu'une petite partie du dataset complet.
- Environ 4,5 % des images téléchargées étaient une image de remplacement identique (lien mort) : ces posts sont retirés par `clean_placeholders.py` (liste dans `data/removed_placeholders.tsv`).
