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