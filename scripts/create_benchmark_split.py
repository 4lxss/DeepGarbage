"""Crée un split commun, reproductible et inédit pour comparer MLP et CNN."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "sorted"
OUTPUT_DIR = ROOT / "data" / "benchmark"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
SEED = 42


def main() -> None:
    # Reproduction exacte de `stratified_split` dans mlp_model.py :
    # même ordre de fichiers (ImageFolder) et même générateur PyTorch.
    generator = torch.Generator().manual_seed(SEED)
    rows = {"train": [], "validation": [], "test": []}

    for label, class_name in enumerate(CLASS_NAMES):
        files = sorted(path for path in (SOURCE_DIR / class_name).iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        order = torch.randperm(len(files), generator=generator).tolist()
        files = [files[index] for index in order]
        train_end = int(0.70 * len(files))
        validation_end = train_end + int(0.15 * len(files))
        for split, split_files in (
            ("train", files[:train_end]),
            ("validation", files[train_end:validation_end]),
            ("test", files[validation_end:]),
        ):
            rows[split].extend({"path": str(path.relative_to(ROOT)), "label": label} for path in split_files)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split, split_rows in rows.items():
        frame = pd.DataFrame(split_rows)
        frame.to_csv(OUTPUT_DIR / f"{split}.csv", index=False)
        print(f"{split:<10}: {len(frame)} images")


if __name__ == "__main__":
    main()
