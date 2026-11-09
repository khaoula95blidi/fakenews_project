# Détection multimodale précoce des fake news (Fakeddit)

*Early Multimodal Fake News Detection Using Text and Images* : projet académique de recherche.

## Objectif

Comparer trois familles de modèles (texte seul, image seule, fusion texte + image) pour savoir **si, quand et comment l'image apporte un gain** dans la détection de fake news.

« Précoce » désigne ici la détection à partir du **contenu initial d'une publication (titre + image)**, sans métadonnées de popularité ni commentaires, qui n'existent qu'après la publication.

## Avancement

| Étape | Contenu | État |
|---|---|---|
| E1 | Texte : TF-IDF + régression logistique | fait |
| E2 | Texte : DistilBERT gelé + régression logistique | fait |
| E3 | Image : ResNet-18 gelé (et comparaison avec EfficientNet-B0, ViT-B/16, CLIP ViT-B/32) | fait |
| E4b | Fusion précoce : DistilBERT + CLIP concaténés | notebook prêt, résultats à reporter |
| E5b | Fusion tardive : moyenne pondérée des probabilités de E2 et E3d | notebook prêt, résultats à reporter |
| RQ3 | Robustesse : texte masqué (30 %, 50 %), image floutée | à faire |
| RQ4 | Généralisation à des sous-reddits absents de l'entraînement | à faire |
| MLOps | Suivi MLflow : fait. Application Streamlit et image Docker : à faire | en cours |

## Données

[Fakeddit](https://arxiv.org/abs/1911.03854) (Nakamura et al., 2020) : posts Reddit avec titre, image et labels (2, 3 ou 6 classes). Dépôt officiel : <https://github.com/entitize/Fakeddit>.

- Tâche principale : classification binaire (`2_way_label` : 1 = vrai, 0 = faux).
- Sous-ensemble équilibré construit par `src/make_subset.py` (graine 42) : 10 711 posts avec image (7 792 train, 1 469 val, 1 450 test) avant nettoyage.
- `src/clean_placeholders.py` retire ensuite les 480 posts dont l'image est une image de remplacement (lien mort, 4,5 %). **Effectifs finaux : 10 231 posts, soit 7 432 train, 1 403 val, 1 396 test.** Tous les résultats ci-dessous sont calculés sur cette version nettoyée.
- Les splits fixes `data/train.tsv`, `data/val.tsv`, `data/test.tsv` sont versionnés. Ne pas les régénérer.

## Structure

```
.
├── README.md
├── requirements.txt
├── suivi_mlflow.py                         # suivi des expériences avec MLflow
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
├── notebooks/                              # exécutés sur Kaggle
│   ├── eda_fakeddit.ipynb                  # analyse exploratoire
│   ├── 02_embeddings_e2_e3.ipynb           # E2 (DistilBERT), E3 (ResNet-18) et autres réseaux d'images
│   ├── 05b_e4b_fusion_precoce_clip.ipynb   # E4b : fusion précoce
│   └── 06b_e5b_fusion_tardive_clip.ipynb   # E5b : fusion tardive
├── results/                                # métriques (.json) et probabilités (.npy) de chaque expérience
│   ├── e1_metrics.json  e2_e3_metrics.json  e3_autres_reseaux_metrics.json
│   ├── e4b_metrics.json  e5b_metrics.json  eda_resume.json
│   └── mlflow/                             # un résultat par expérience, à charger dans MLflow
└── figures/                                # figures de l'analyse exploratoire
```

Non versionnés (voir `.gitignore`) : `data/raw/`, `emb/` (représentations calculées sur Kaggle), `mlflow.db`, `mlartifacts/`, `.venv/`.

## Protocole commun à E1–E5

- Mêmes splits, graine 42, un seul test lu une fois par modèle.
- Texte et image sont représentés par des réseaux **gelés** (aucun réglage fin) : DistilBERT pour le titre ; ResNet-18, EfficientNet-B0, ViT-B/16 ou CLIP ViT-B/32 pour l'image. Les représentations sont calculées une fois, puis une régression logistique est entraînée dessus (`StandardScaler` puis `LogisticRegression`, `class_weight="balanced"`).
- Le coefficient de régularisation C est choisi sur la **validation** uniquement. Grille d'origine : 0,0003 à 1 (E2, E3 officiel). Grille étendue : 0,00001 à 1 (autres réseaux d'images, E4b).
- Mesures : exactitude, Macro F1 (mesure principale) et AUC. Marge d'erreur à 95 % de l'exactitude sur 1 396 publications : environ ± 2,4 points.
- Gain de la fusion = Macro F1 de la fusion − Macro F1 du meilleur modèle seul (choisi sur la validation), en points, avec un intervalle à 95 % par bootstrap sur le test (1 000 tirages). Si l'intervalle contient 0, le gain n'est pas établi. Il n'y a qu'une seule graine : l'intervalle reflète l'échantillonnage du test, pas la variabilité de l'entraînement.

## Reproduire

### 1. Environnement

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1          # Windows (PowerShell) ; Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Données et E1 (en local)

```bash
# Télécharger les TSV multimodaux de Fakeddit (voir le dépôt officiel) et les placer dans data/raw/
# (Optionnel) reconstruire les splits et les images, puis les nettoyer
python src/make_subset.py
# Lancer la baseline E1
python src/e1_tfidf.py
```

### 3. E2 à E5b (sur Kaggle)

1. Créer un dataset Kaggle avec la version **nettoyée** : `train.tsv`, `val.tsv`, `test.tsv`, `images.zip`, `e1_val_proba.npy`, `e1_test_proba.npy`. Les notebooks `eda_fakeddit` et `02` s'arrêtent si les effectifs ne sont pas 7 432 / 1 403 / 1 396.
2. Lancer dans l'ordre, avec « Save & Run All » :
   - `eda_fakeddit.ipynb` : analyse exploratoire.
   - `02_embeddings_e2_e3.ipynb` : E2, E3 et les trois autres réseaux d'images (GPU conseillé, Internet activé pour télécharger les poids).
   - `05b_e4b_fusion_precoce_clip.ipynb` et `06b_e5b_fusion_tardive_clip.ipynb` : ajouter en Input le dataset **et** la sortie du notebook 02 (version avec les réseaux d'images). Processeur suffisant.
3. Copier de l'onglet Output vers le projet les dossiers `results/` et `figures/` (fusionner, ne pas remplacer). Ne pas copier `emb/`.

### 4. Suivi des expériences avec MLflow

```bash
python suivi_mlflow.py                  # enregistre la référence « tout vrai » et E1 à E3
Get-ChildItem results\mlflow\*.json | ForEach-Object { python suivi_mlflow.py --json $_.FullName }   # PowerShell
python suivi_mlflow.py --ui             # interface sur http://127.0.0.1:5000
```

Relancer une expérience avec le même nom remplace son run : pas de doublon. La base MLflow se reconstruit toujours à partir de `results/mlflow/*.json`.

## Résultats (jeu de test, 1 396 publications)

Arrondis à 3 décimales ; les valeurs exactes sont dans `results/*.json`.

| Modèle | Entrées | Accuracy | Macro F1 | AUC |
|---|---|---|---|---|
| Majoritaire (tout « vrai ») | : | 0,503 | 0,335 | 0,500 |
| E1 TF-IDF + LR | Texte | 0,751 | 0,750 | 0,822 |
| E2 DistilBERT gelé + LR | Texte | 0,801 | 0,801 | 0,862 |
| E3 ResNet-18 gelé + LR | Image | 0,699 | 0,699 | 0,755 |
| E3d CLIP ViT-B/32 gelé + LR | Image | 0,774 | 0,774 | 0,859 |
| E4b Fusion précoce (DistilBERT + CLIP) | Texte + image | à compléter | à compléter | à compléter |
| E5b Fusion tardive (E2 + E3d) | Texte + image | à compléter | à compléter | à compléter |

Gain de la fusion par rapport au meilleur modèle seul (Macro F1, avec intervalle à 95 %) : E4b : à compléter ; E5b : à compléter. Les valeurs sont dans `results/e4b_metrics.json` et `results/e5b_metrics.json`.

### E3 : quatre réseaux d'images (même protocole)

| Réseau | Macro F1 validation | Accuracy test | Macro F1 test | AUC test | Marge accuracy à 95 % |
|---|---|---|---|---|---|
| ResNet-18 (E3) | 0,682 | 0,699 | 0,699 | 0,755 | ± 2,4 |
| EfficientNet-B0 (E3b) | 0,682 | 0,672 | 0,672 | 0,739 | ± 2,5 |
| ViT-B/16 (E3c) | 0,683 | 0,670 | 0,670 | 0,743 | ± 2,5 |
| CLIP ViT-B/32 (E3d) | 0,785 | 0,774 | 0,774 | 0,859 | ± 2,2 |

- Les trois réseaux entraînés sur ImageNet sont à égalité sur la validation (0,682 à 0,683). Sur le test, ResNet-18 est un peu devant, avec un écart proche de la marge d'erreur.
- CLIP est nettement meilleur : + 10 points de Macro F1 sur la validation et environ + 7,5 points sur le test par rapport à ResNet-18. Seul, il approche le texte (0,774 contre 0,801). C'est donc CLIP qui sert d'image dans la fusion (E4b, E5b), choisi sur la validation de E3 avant de regarder la fusion.
- Un nouveau calcul de ResNet-18 dans le notebook 02 (`E3ctrl`) redonne exactement les chiffres de E3 : le protocole est reproductible.

## Limites connues

- Les labels Fakeddit sont dérivés du subreddit d'origine (supervision distante) : un modèle peut apprendre le style d'une communauté plutôt que la désinformation. Le subreddit seul suffit à prédire la classe (exactitude de 1,0 sur le test) ; RQ4 teste ce risque.
- Les titres de la classe « image manipulée » sont très courts et génériques, ce qui facilite sa détection par le texte sans rapport avec l'image.
- La correspondance entre les 6 classes de type et leurs noms est en cours de vérification (« trompeur » et « fausse connexion » pourraient être inversés). Les exactitudes par type sont à lire avec cette réserve ; elles portent sur de petits effectifs (par exemple 51 publications pour « fausse connexion », 21 pour « imposteur »).
- Le sous-ensemble ne représente qu'une petite partie du dataset complet.
- Environ 4,5 % des images téléchargées étaient une image de remplacement identique (lien mort) : ces posts sont retirés par `clean_placeholders.py` (liste dans `data/removed_placeholders.tsv`).
- Une seule graine : les écarts de moins de 2 à 3 points ne sont pas établis.
- CLIP est entraîné sur des centaines de millions d'images du web : on ne peut pas exclure qu'il ait vu des images proches de celles de Fakeddit.
- Les réseaux sont gelés : aucun réglage fin n'a été testé.

## Références

- Nakamura, Levy, Wang (2020). *Fakeddit: A New Multimodal Benchmark Dataset for Fine-grained Fake News Detection.* LREC.
- Sanh et al. (2019). *DistilBERT, a distilled version of BERT.*
- He et al. (2016). *Deep Residual Learning for Image Recognition.* CVPR.
- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision (CLIP).* ICML.
