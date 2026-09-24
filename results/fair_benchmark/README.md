# Résultats finaux — comparaison équitable CNN / MLP

Les deux modèles utilisent les mêmes images, le même découpage stratifié (graine 42), les mêmes ensembles train/validation/test et des images RGB de 64 × 64 pixels.

| Ensemble | Images | Rôle |
|---|---:|---|
| Train | 17 440 | Entraîner les modèles |
| Validation | 3 735 | Choisir les paramètres |
| Test | 3 741 | Évaluation finale, une seule fois |

## Expérimentations CNN

| Configuration | Learning rate | Batch size | Accuracy validation | Loss validation |
|---|---:|---:|---:|---:|
| Baseline retenue | 0,001 | 32 | 65,35 % | 0,9699 |
| Batch 64 | 0,001 | 64 | 62,41 % | 1,0320 |
| Learning rate réduit | 0,0003 | 32 | 62,03 % | 1,0435 |

## Évaluation finale

| Modèle | Accuracy test | Precision macro | Recall macro | F1 macro |
|---|---:|---:|---:|---:|
| MLP | **66,08 %** | **65,35 %** | 61,60 % | 62,79 % |
| CNN | 64,80 % | 65,07 % | **62,46 %** | **63,30 %** |

Le MLP obtient l'accuracy globale la plus élevée. Le CNN a le meilleur recall macro et F1 macro, donc une performance légèrement plus équilibrée entre classes.

## Fichiers à insérer dans le rapport

- `cnn/baseline_curves.png` : courbes d'apprentissage du CNN.
- `cnn/test_confusion_matrix.png` : matrice de confusion finale du CNN.
- `cnn/comparison.csv` : chiffres des expériences CNN.
- `comparison_cnn_mlp.csv` : chiffres de la comparaison finale.
