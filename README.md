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
