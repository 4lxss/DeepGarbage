# DeepGarbage

Dataset setup for deep learning experiments with Hugging Face datasets.

This project uses the `garythung/trashnet` dataset from Hugging Face, with six garbage classes: cardboard, glass, metal, paper, plastic, and trash. To recreate the normalized training files, install the requirements and run `python scripts/download_normalize_dataset.py` from the project virtual environment.

## Install

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ubuntu/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For gated or private datasets, authenticate first:

```bash
hf auth login
```

## Download and normalize a dataset

By default, the script downloads `garythung/trashnet`:

```bash
python scripts/download_normalize_dataset.py
```

For a quick test run:

```bash
python scripts/download_normalize_dataset.py --max-rows-per-split 20
```

Outputs are written to `data/processed/`:

- Image datasets: resized RGB images, one CSV manifest per split, and channel mean/std in `dataset_info.json`.
- Text/tabular datasets: one normalized Parquet file per split and train-split normalization stats in `dataset_info.json`.

The script follows the Hugging Face `datasets.load_dataset(...)` flow. If your dataset has a non-standard label column, pass it explicitly:

```bash
python scripts/download_normalize_dataset.py --dataset username/my_dataset --label-column category
```

## Train the MLP baseline

`mlp_model.py` trains a simple MLP classifier as a first baseline model, using `torchvision.datasets.ImageFolder` to load images from a folder with one subfolder per class.

The processed dataset in `data/processed/` is flat (images plus a CSV manifest), so it needs to be reorganized into a per-class folder layout before running the baseline:

```bash
python3 -c "
import csv, json, shutil, os

info = json.load(open('data/processed/dataset_info.json'))
label_map = info['label_mapping']

for name in label_map.values():
    os.makedirs(f'data/imagefolder/{name}', exist_ok=True)

with open('data/processed/train.csv') as f:
    for row in csv.DictReader(f):
        src = f\"data/processed/{row['path']}\"
        cls = label_map[row['label']]
        dst = f\"data/imagefolder/{cls}/{os.path.basename(row['path'])}\"
        shutil.copy(src, dst)
"
```

Then set `DATA_DIR` in `mlp_model.py` to `"./data/imagefolder"` and run:

```bash
python mlp_model.py
```

This prints the detected classes, the train/val image counts, the total parameter count, and per-epoch train/val loss and validation accuracy. The goal is only to confirm that the training loss decreases and the whole pipeline runs end to end, not to obtain a final model.

## Entrainement, optimisation et evaluation du MLP

Cette section presente uniquement la partie consacree a l'entrainement, a l'optimisation et a l'evaluation du MLP. Elle explique ce qui a ete ajoute autour du modele afin de transformer une premiere boucle d'entrainement en protocole experimental exploitable pour le compte rendu et la presentation orale.

### 1. Liste exhaustive des fonctionnalites ajoutees

#### 1.1 Preparation reproductible des jeux de donnees

- Separation des donnees en trois ensembles : 70 % pour l'entrainement, 15 % pour la validation et 15 % pour le test.
- Separation realisee classe par classe, et non uniquement sur l'ensemble global.
- Utilisation d'une graine aleatoire fixe (`42`) afin de retrouver les memes images dans les trois ensembles lors d'une nouvelle execution.
- Utilisation de `Subset` pour construire les ensembles a partir du dataset charge.
- Conservation d'un vrai jeu de test independant, utilise seulement apres le choix des hyperparametres.

#### 1.2 Mise en place de la boucle d'entrainement

- Execution de la propagation avant sur chaque batch.
- Calcul de la fonction de perte `CrossEntropyLoss`.
- Remise a zero des gradients avec `optimizer.zero_grad()`.
- Calcul des gradients par retropropagation avec `loss.backward()`.
- Mise a jour des poids avec `optimizer.step()`.
- Passage explicite du modele en mode entrainement avec `model.train()`.
- Passage explicite du modele en mode evaluation avec `model.eval()`.
- Desactivation des gradients pendant la validation et le test avec `torch.no_grad()`.

#### 1.3 Choix et comparaison de l'optimisation

- Utilisation de `CrossEntropyLoss`, adaptee a la classification multiclasse avec les logits produits par le MLP.
- Utilisation de l'optimiseur SGD avec momentum.
- Valeurs de reference : learning rate `0.01`, momentum `0.9`, weight decay `0.0` et batch size `32`.
- Comparaison de plusieurs learning rates : `0.001`, `0.01` et `0.05`.
- Comparaison de plusieurs tailles de batch : `16`, `32` et `64`.
- Test d'une regularisation L2 avec `weight_decay=1e-4`.

#### 1.4 Suivi des performances

A chaque epoque, le script conserve quatre valeurs :

- la loss d'entrainement ;
- la loss de validation ;
- l'accuracy d'entrainement ;
- l'accuracy de validation.

Ces valeurs sont conservees dans un historique pour pouvoir analyser l'evolution du modele, et pas seulement afficher la performance de la derniere epoque.

#### 1.5 Selection du meilleur modele

- Conservation d'une copie des poids du modele lorsque la `val_loss` s'ameliore.
- Selection du meilleur modele selon la loss de validation, et non selon l'accuracy d'entrainement.
- Recherche de la meilleure configuration parmi la baseline et les experiences d'hyperparametres.
- Utilisation du jeu de validation pour choisir le modele.
- Conservation du jeu de test pour la mesure finale uniquement.

#### 1.6 Courbes d'apprentissage

Deux graphiques sont prepares :

- une courbe de loss comparant entrainement et validation ;
- une courbe d'accuracy comparant entrainement et validation.

Ces courbes permettent de voir si le modele apprend, s'il stagne, s'il sous-apprend ou s'il commence a memoriser les donnees d'entrainement.

#### 1.7 Detection de l'overfitting

Le script compare l'accuracy d'entrainement et l'accuracy de validation a la meilleure epoque. Un ecart important signifie que le modele est beaucoup plus performant sur les donnees vues pendant l'entrainement que sur des donnees non vues.

L'analyse repose aussi sur les courbes :

- si la loss d'entrainement continue de diminuer mais que la loss de validation augmente, le modele risque de surapprendre ;
- si l'accuracy d'entrainement augmente alors que l'accuracy de validation stagne ou diminue, le modele memorise probablement les donnees d'entrainement ;
- si les deux performances restent faibles, le modele peut etre en sous-apprentissage ou le learning rate peut etre mal choisi.

#### 1.8 Evaluation finale

Apres la selection du modele sur la validation, le modele est evalue sur le jeu de test. Les metriques calculees sont :

- accuracy ;
- precision macro et weighted ;
- recall macro et weighted ;
- F1-score macro et weighted ;
- precision, recall et F1-score pour chaque classe ;
- matrice de confusion.

La moyenne `macro` donne le meme poids a chaque categorie. Elle est utile ici car les classes ne contiennent pas forcement le meme nombre d'images. La moyenne `weighted` tient compte du nombre d'exemples de chaque classe.

### 2. Explication technique de l'entrainement

Le modele traite les images par lots, ou batches. Pour chaque image, la boucle d'entrainement execute d'abord une propagation avant. Le MLP produit un vecteur de logits, c'est-a-dire une valeur par classe. Ces logits ne sont pas transformes manuellement avec `softmax`, car `CrossEntropyLoss` effectue en interne la combinaison necessaire entre la stabilisation des logits et la perte de classification.

La loss mesure l'ecart entre la classe reelle et les logits predits. Plus elle est faible, plus les predictions sont coherentes avec les labels. Ensuite, `loss.backward()` applique la retropropagation et calcule le gradient de la loss par rapport a chaque poids du reseau. L'optimiseur SGD modifie alors les poids dans la direction qui diminue la loss. Le momentum aide a stabiliser et accelerer ces mises a jour.

La validation est executee apres chaque epoque sans modifier les poids. Elle sert a mesurer la generalisation du modele sur des images qui n'ont pas servi a l'apprentissage. Le test joue un role different : il ne sert pas a prendre de decision pendant l'optimisation et ne doit etre consulte qu'apres le choix final.

### 3. Pourquoi le decoupage est fait par classe

Un decoupage global aleatoire pourrait produire des proportions tres differentes entre les classes dans les ensembles train, validation et test. Avec le decoupage implemente ici, chaque categorie est melangee independamment, puis repartie selon les memes proportions.

Pour une classe contenant `N` images, le calcul est :

```text
nombre_train = int(0.70 * N)
nombre_validation = int(0.15 * N)
nombre_test = N - nombre_train - nombre_validation
```

Le reste est attribue au test afin que toutes les images de la classe soient utilisees exactement une fois. A cause de l'arrondi entier, une classe peut avoir un ecart d'une image par rapport a 70 %, 15 % ou 15 %, ce qui est normal et necessaire pour obtenir des nombres entiers.

### 4. Comment les experiences sont comparees

La baseline utilise les parametres de reference. Les autres experiences ne changent pas l'architecture du MLP : elles changent uniquement un parametre d'entrainement a la fois autant que possible. Cette methode rend la comparaison interpretable.

La comparaison porte principalement sur la meilleure `val_accuracy` obtenue et sur la `val_loss` correspondante. Le temps d'entrainement, l'ecart train-validation et la stabilite des courbes peuvent aussi etre commentes. Une accuracy d'entrainement elevee ne suffit pas pour choisir un modele, car elle peut indiquer que le modele memorise le jeu d'entrainement.

### 5. Interpretation des metriques

Pour une classe donnee, on utilise les definitions suivantes :

```text
precision = vrais positifs / (vrais positifs + faux positifs)
recall    = vrais positifs / (vrais positifs + faux negatifs)
F1-score  = 2 * precision * recall / (precision + recall)
```

La precision indique, parmi les images predites comme appartenant a une classe, combien sont correctes. Le recall indique, parmi les images qui appartiennent reellement a une classe, combien ont ete retrouvees. Le F1-score combine les deux et devient faible lorsqu'une des deux valeurs est faible.

La matrice de confusion indique les classes reelles en lignes et les classes predites en colonnes. Les valeurs sur la diagonale correspondent aux bonnes predictions. Les valeurs hors diagonale indiquent les confusions entre categories. Elle permet donc d'aller au-dela d'une accuracy globale et d'identifier les classes difficiles a reconnaitre.

### 6. Texte de presentation orale

> Ma partie concerne l'entrainement, l'optimisation et l'evaluation du MLP. L'objectif n'etait pas de redefinir l'architecture, mais de mettre en place un protocole experimental permettant de mesurer la qualite de l'apprentissage et de choisir les meilleurs parametres.
>
> J'ai commence par separer le dataset en trois parties : 70 % pour l'entrainement, 15 % pour la validation et 15 % pour le test. La separation est realisee independamment pour chaque categorie de dechets. Cela permet de conserver une representation comparable des classes dans les trois ensembles. Une graine aleatoire fixe a ete utilisee afin que la separation soit reproductible.
>
> La boucle d'entrainement fonctionne par batches. Pour chaque batch, le modele produit des logits pendant la propagation avant. La fonction `CrossEntropyLoss` mesure l'erreur de classification. La retropropagation calcule ensuite les gradients avec `loss.backward()`, puis l'optimiseur SGD met a jour les poids avec `optimizer.step()`. Pendant la validation et le test, les poids ne sont pas modifies et les gradients sont desactives.
>
> J'ai conserve, pour chaque epoque, la loss et l'accuracy sur l'entrainement ainsi que sur la validation. Cette conservation permet de tracer les courbes d'apprentissage et de comparer l'evolution des performances. Les courbes servent notamment a detecter l'overfitting. Par exemple, si la performance d'entrainement continue de progresser alors que la validation stagne ou se degrade, cela signifie que le modele generalise moins bien.
>
> J'ai ensuite compare plusieurs hyperparametres. La baseline utilise un learning rate de 0.01, un momentum de 0.9, un batch size de 32 et aucune regularisation L2. J'ai teste des learning rates de 0.001 et 0.05, des batch sizes de 16 et 64, ainsi qu'un weight decay de 1e-4. L'architecture reste fixe afin que les differences observees proviennent des choix d'entrainement et non d'un changement simultane du reseau.
>
> Pour chaque experience, je conserve les poids correspondant a la meilleure loss de validation. Je selectionne ensuite la meilleure configuration avec les performances de validation. Le jeu de test n'intervient pas dans cette selection : il est utilise uniquement a la fin pour obtenir une evaluation independante.
>
> Enfin, j'evalue le modele retenu avec l'accuracy, la precision, le recall et le F1-score, a la fois globalement et pour chaque classe. J'utilise les moyennes macro et weighted pour tenir compte du desequilibre possible entre les categories. Je genere aussi une matrice de confusion afin d'identifier les classes qui sont le plus souvent confondues.
>
> Cette partie permet donc de passer d'un modele qui fonctionne techniquement a une evaluation experimentale complete : on observe l'apprentissage, on compare les reglages, on recherche l'overfitting, on selectionne un modele sur validation et on mesure finalement ses performances sur un jeu de test independant.

### 7. Points a verifier avant la presentation

- Executer le script jusqu'a la fin et conserver les valeurs obtenues dans le tableau des experiences.
- Reporter les valeurs reelles d'accuracy, precision, recall et F1-score dans le compte rendu.
- Commenter les courbes avec les observations effectivement obtenues, plutot que de presenter un diagnostic hypothetique.
- Lire la matrice de confusion et citer au moins une ou deux confusions importantes entre classes.
- Preciser le nombre d'epoques effectivement utilise. La configuration actuelle du script est `NUM_EPOCHS = 3`; elle peut etre augmentee pour l'experimentation finale si le temps de calcul le permet.
- Preciser que le MLP utilise des images aplaties et ne conserve pas explicitement les relations spatiales entre pixels, ce qui constitue une limite importante par rapport a un CNN.

## Compte rendu chiffre des experiences MLP

Cette section reprend les resultats de l'execution complete de `mlp_model.py`. Les valeurs ci-dessous sont les resultats effectivement observes et peuvent etre presentes aux collegues et au professeur.

### 1. Configuration experimentale

Le dataset contient `24 916` images reparties en six classes : `cardboard`, `glass`, `metal`, `paper`, `plastic` et `trash`.

Le decoupage realise classe par classe donne :

| Ensemble | Nombre d'images | Pourcentage theorique |
|---|---:|---:|
| Entrainement | 17 440 | 70 % |
| Validation | 3 735 | 15 % |
| Test | 3 741 | 15 % |
| Total | 24 916 | 100 % |

Les petites differences eventuelles avec les pourcentages exacts viennent de l'arrondi a l'entier pour chaque classe. Le test n'est pas utilise pendant la comparaison des hyperparametres.

Les images sont redimensionnees en `64 x 64` et contiennent trois canaux RGB. L'entree du MLP contient donc :

```text
64 x 64 x 3 = 12 288 valeurs
```

L'architecture contient `6 358 406` parametres entrainables. Ce total se calcule ainsi :

```text
Couche 1 : 12 288 x 512 + 512 = 6 291 968
Couche 2 : 512 x 128 + 128 = 65 664
Sortie   : 128 x 6 + 6 = 774
Total    : 6 358 406
```

La fonction de perte est `CrossEntropyLoss`. L'optimiseur est SGD avec momentum `0.9`. La baseline utilise un learning rate de `0.01`, un batch size de `32`, un weight decay de `0` et `30` epoques.

### 2. Resultats de la baseline

La baseline atteint :

```text
Meilleure validation accuracy : 53.44 %
Meilleure validation loss      : 1.2485
Train accuracy a la fin       : 55.33 %
Validation accuracy a la fin  : 50.31 %
Train loss a la fin           : 1.1718
Validation loss a la fin      : 1.3590
```

La loss d'entrainement diminue de `1.6396` a `1.1718` et l'accuracy d'entrainement augmente de `32.11 %` a `55.33 %`. Le modele apprend donc.

Cependant, la validation devient moins bonne apres son meilleur niveau. A la fin, l'ecart train-validation vaut :

```text
55.33 - 50.31 = 5.02 points
```

Cela indique un debut d'overfitting : le modele continue a progresser sur les images d'entrainement, mais generalise moins bien sur les images de validation.

### 3. Comparaison des hyperparametres

Les six configurations gardent exactement la meme architecture. Seuls les parametres d'optimisation changent.

| Configuration | Meilleure val. accuracy | Meilleure val. loss | Epoque de la meilleure loss | Interpretation |
|---|---:|---:|---:|---|
| Learning rate `0.001` | **65.38 %** | **0.9699** | 30 | Meilleur resultat |
| Batch size `64` | 58.55 % | 1.1334 | 27 | Deuxieme meilleur resultat |
| Baseline, learning rate `0.01` | 53.44 % | 1.2485 | 26 | Reference |
| Weight decay `1e-4` | 50.04 % | 1.3114 | 22 | Regularisation penalisee dans ce cas |
| Batch size `16` | 37.59 % | 1.5562 | 27 | Apprentissage instable |
| Learning rate `0.05` | 24.10 % | 1.7174 | 13 | Learning rate trop eleve |

#### Learning rate `0.001`

C'est la meilleure configuration. Elle termine avec `70.08 %` d'accuracy d'entrainement, `65.38 %` d'accuracy de validation et une validation loss de `0.9699`.

L'ecart entre entrainement et validation vaut :

```text
70.08 - 65.38 = 4.70 points
```

Cet ecart reste raisonnable. Le learning rate plus faible permet des mises a jour plus progressives et une meilleure convergence.

#### Learning rate `0.05`

Le modele reste autour de `22 %` a `24 %` d'accuracy et sa loss reste proche de `1.72`. Avec six classes, une prediction aleatoire correspond a environ :

```text
100 / 6 = 16.67 %
```

Le learning rate `0.05` est donc trop eleve : les mises a jour sont trop importantes et le modele ne converge pas correctement.

#### Batch size `16`

La meilleure validation accuracy n'est que de `37.59 %`. Les performances varient fortement d'une epoque a l'autre. Les petits batches produisent ici des mises a jour plus bruitees et ne permettent pas une bonne generalisation.

#### Batch size `64`

Le batch size `64` atteint `58.55 %`, soit `5.11 points` de plus que la baseline :

```text
58.55 - 53.44 = 5.11 points
```

Il est meilleur que le batch size `16`, mais reste moins performant que le learning rate `0.001`.

#### Weight decay `1e-4`

Le weight decay ajoute une regularisation L2. Dans cette experience, il reduit la validation accuracy a `50.04 %`, contre `53.44 %` pour la baseline. Cette valeur de regularisation est donc trop penalisee ou mal adaptee au learning rate `0.01`.

### 4. Choix du modele final

Le modele retenu est celui avec un learning rate de `0.001` et un batch size de `32`.

La comparaison est faite sur le jeu de validation, jamais sur le test. Le test est conserve pour mesurer la performance finale sur des images qui n'ont pas servi a choisir les hyperparametres.

L'accuracy de validation du meilleur modele est `65.38 %`. Le modele presente un ecart train-validation de `4.70 points`, ce qui montre une generalisation correcte mais pas parfaite.

### 5. Evaluation finale sur le test

Le modele final est ensuite evalue sur les `3 741` images du test :

| Metrique | Resultat |
|---|---:|
| Accuracy | **66.08 %** |
| Precision macro | 65.35 % |
| Recall macro | 61.60 % |
| F1-score macro | 62.79 % |
| Precision weighted | 66.25 % |
| Recall weighted | 66.08 % |
| F1-score weighted | 65.80 % |

La difference entre la validation et le test est :

```text
66.08 - 65.38 = 0.70 point
```

Cette difference est faible et indique que le modele se comporte de maniere similaire sur les deux ensembles. Il n'y a donc pas de degradation anormale sur le test.

### 6. Resultats par classe

| Classe | Precision | Recall | F1-score | Support test |
|---|---:|---:|---:|---:|
| Cardboard | 0.79 | 0.71 | 0.75 | 696 |
| Glass | 0.63 | 0.74 | 0.68 | 857 |
| Metal | 0.60 | 0.55 | 0.57 | 480 |
| Paper | 0.66 | 0.67 | 0.67 | 691 |
| Plastic | 0.63 | 0.68 | 0.66 | 792 |
| Trash | 0.60 | 0.35 | 0.44 | 225 |

`Cardboard` est la classe la mieux reconnue, avec un F1-score de `0.75`. Le modele fait peu de faux positifs sur cette classe et retrouve une grande partie des vrais cartons.

`Trash` est la classe la plus difficile. Son recall est de `0.35`, ce qui signifie que le modele ne retrouve qu'environ 35 % des vrais exemples de cette classe. Environ 65 % des objets `trash` sont donc classes dans une autre categorie.

Le F1-score macro est de `62.79 %`, alors que le F1-score weighted est de `65.80 %`. Cette difference montre que les performances sont meilleures sur les classes plus representees et que les classes minoritaires influencent negativement la moyenne macro.

### 7. Conclusion a presenter a l'oral

> Nous avons compare plusieurs reglages d'entrainement en conservant la meme architecture de MLP. Le dataset contient 24 916 images reparties en six classes. Nous avons utilise une separation stratifiee de 70 % pour l'entrainement, 15 % pour la validation et 15 % pour le test.
>
> La baseline, avec un learning rate de 0.01 et un batch size de 32, atteint 53.44 % de validation accuracy. Le meilleur resultat est obtenu avec un learning rate de 0.001 : la validation accuracy atteint 65.38 % et la validation loss descend a 0.9699. Le test sur 3 741 images donne ensuite une accuracy de 66.08 %, ce qui est tres proche de la validation et indique une generalisation coherente.
>
> Le learning rate de 0.05 est trop eleve : l'accuracy reste autour de 24 % et la loss reste proche de 1.72. Le batch size de 64 donne 58.55 %, tandis que le batch size de 16 donne seulement 37.59 %. Le weight decay de 1e-4 n'ameliore pas la baseline dans cette configuration.
>
> Les performances varient selon les classes. Cardboard est la classe la mieux reconnue avec un F1-score de 0.75. Trash est la plus difficile avec un recall de 0.35 et un F1-score de 0.44. Cette difference est liee au desequilibre des classes et a la diversite visuelle des objets de la categorie trash.
>
> Le MLP fournit donc une baseline fonctionnelle avec environ 66 % d'accuracy sur le test. Cependant, il aplatit les images et ne conserve pas explicitement leurs relations spatiales. Le CNN devrait donc normalement etre plus adapte a cette tache de reconnaissance d'images.

### 8. Remarque sur le tableau des epoques

Les logs montrent les epoques correspondant aux meilleures validation losses : baseline `26`, learning rate `0.001` `30`, learning rate `0.05` `13`, batch size `16` `27`, batch size `64` `27` et weight decay `22`.

Si le tableau affiche `30` pour toutes les configurations ou concatene `epochs` et `best_epoch`, il s'agit d'un probleme d'affichage du tableau, pas d'un resultat d'entrainement. Il faut corriger cet affichage avant de l'utiliser dans le rapport. Le critere d'entrainement doit aussi etre explique clairement : les poids sont conserves selon la meilleure validation loss, tandis que les configurations sont comparees selon leur meilleure validation accuracy.
