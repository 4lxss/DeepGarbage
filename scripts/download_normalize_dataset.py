"""Download and normalize a Hugging Face dataset for deep learning.

Examples:
    python scripts/download_normalize_dataset.py
    python scripts/download_normalize_dataset.py --dataset username/my_dataset --label-column label
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any
from zipfile import ZipFile

try:
    import numpy as np
    import pandas as pd
    import kagglehub
    from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Value, concatenate_datasets, load_dataset
    from datasets.exceptions import DatasetGenerationError
    from huggingface_hub import hf_hub_download
    from PIL import Image as PILImage
except ModuleNotFoundError as error:
    missing = error.name or "a required package"
    raise SystemExit(
        f"Missing dependency: {missing}\n"
        "Run with the project virtual environment:\n"
        r"  .\.venv\Scripts\python.exe scripts\download_normalize_dataset.py"
        "\nOr activate it first:\n"
        r"  .\.venv\Scripts\Activate.ps1"
        "\n"
        r"  python scripts\download_normalize_dataset.py"
    ) from error


TRASHNET_DATASET_ID = "garythung/trashnet"
KAGGLE_DATASET_ID = "sumn2u/garbage-classification-v2"
TRASHNET_CLASSES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
KAGGLE_CLASS_ALIASES = {
    "brown-glass": "glass",
    "green-glass": "glass",
    "white-glass": "glass",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face dataset and create normalized training artifacts."
    )
    parser.add_argument(
        "--dataset",
        default="garythung/trashnet",
        help="Hugging Face dataset id, e.g. 'garythung/trashnet', 'beans', or 'org/name'.",
    )
    parser.add_argument("--config", default=None, help="Optional dataset configuration name.")
    parser.add_argument("--output-dir", default="data/processed", help="Directory for normalized artifacts.")
    parser.add_argument("--cache-dir", default="data/raw/hf", help="Hugging Face cache directory.")
    parser.add_argument("--label-column", default=None, help="Label column name. Auto-detected when omitted.")
    parser.add_argument("--image-size", type=int, default=224, help="Square image size for image datasets.")
    parser.add_argument("--max-rows-per-split", type=int, default=None, help="Optional limit for quick experiments.")
    parser.add_argument(
        "--include-kaggle",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=f"Merge filtered images from Kaggle dataset '{KAGGLE_DATASET_ID}'.",
    )
    return parser.parse_args()


def dataset_to_dict(dataset: Dataset | DatasetDict) -> DatasetDict:
    if isinstance(dataset, DatasetDict):
        return dataset
    return DatasetDict({"train": dataset})


def limit_rows(dataset: DatasetDict, max_rows: int | None) -> DatasetDict:
    if max_rows is None:
        return dataset
    return DatasetDict(
        {
            split: part.select(range(min(max_rows, len(part))))
            for split, part in dataset.items()
        }
    )


def path_class_name(path: str) -> str | None:
    parts = Path(path).parts
    for part in reversed(parts[:-1]):
        if part in TRASHNET_CLASSES:
            return part
    return None


def normalized_part(value: str) -> str:
    return value.lower().strip().replace("_", "-").replace(" ", "-")


def kaggle_class_name(path: Path) -> str | None:
    for part in reversed(path.parts[:-1]):
        normalized = normalized_part(part)
        if normalized in KAGGLE_CLASS_ALIASES:
            return KAGGLE_CLASS_ALIASES[normalized]
        if normalized in TRASHNET_CLASSES:
            return normalized
    return None


def image_features() -> Features:
    return Features(
        {
            "image": Image(),
            "label": ClassLabel(names=TRASHNET_CLASSES),
            "source": Value("string"),
        }
    )


def load_trashnet_from_zip(cache_dir: str) -> DatasetDict:
    zip_path = hf_hub_download(
        repo_id=TRASHNET_DATASET_ID,
        filename="dataset-resized.zip",
        repo_type="dataset",
        cache_dir=cache_dir,
    )

    records = {"image": [], "label": [], "source": []}
    with ZipFile(zip_path) as archive:
        for member in archive.namelist():
            suffix = Path(member).suffix.lower()
            class_name = path_class_name(member)
            if not class_name or suffix not in IMAGE_EXTENSIONS:
                continue
            records["image"].append(f"zip://{member}::{zip_path}")
            records["label"].append(TRASHNET_CLASSES.index(class_name))
            records["source"].append("huggingface_trashnet")

    if not records["image"]:
        raise RuntimeError(f"No TrashNet images were found in {zip_path}")

    return DatasetDict({"train": Dataset.from_dict(records, features=image_features())})


def load_kaggle_filtered_dataset() -> DatasetDict:
    root = Path(kagglehub.dataset_download(KAGGLE_DATASET_ID))
    records = {"image": [], "label": [], "source": []}

    for image_path in root.rglob("*"):
        if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        class_name = kaggle_class_name(image_path)
        if not class_name:
            continue

        records["image"].append(str(image_path))
        records["label"].append(TRASHNET_CLASSES.index(class_name))
        records["source"].append("kaggle_garbage_classification_v2")

    if not records["image"]:
        raise RuntimeError(f"No matching Kaggle images were found under {root}")

    return DatasetDict({"train": Dataset.from_dict(records, features=image_features())})


def merge_image_datasets(base: DatasetDict, extra: DatasetDict) -> DatasetDict:
    merged: dict[str, Dataset] = {}
    for split, base_part in base.items():
        if split in extra:
            merged[split] = concatenate_datasets([base_part, extra[split]])
        else:
            merged[split] = base_part

    for split, extra_part in extra.items():
        if split not in merged:
            merged[split] = extra_part

    return DatasetDict(merged)


def load_hf_dataset(dataset_id: str, config: str | None, cache_dir: str) -> Dataset | DatasetDict:
    if dataset_id == TRASHNET_DATASET_ID and not config:
        return load_trashnet_from_zip(cache_dir)

    try:
        return load_dataset(
            dataset_id,
            config,
            cache_dir=cache_dir,
        )
    except DatasetGenerationError:
        if dataset_id != TRASHNET_DATASET_ID or config:
            raise
        return load_trashnet_from_zip(cache_dir)


def detect_image_column(part: Dataset) -> str | None:
    for name, feature in part.features.items():
        if isinstance(feature, Image):
            return name
    if len(part) == 0:
        return None
    first = part[0]
    for name, value in first.items():
        if isinstance(value, PILImage.Image):
            return name
    return None


def detect_label_column(part: Dataset, requested: str | None) -> str | None:
    if requested:
        if requested not in part.column_names:
            raise ValueError(f"Label column '{requested}' was not found. Available: {part.column_names}")
        return requested

    for name, feature in part.features.items():
        if isinstance(feature, ClassLabel):
            return name

    candidates = ("label", "labels", "target", "class", "category")
    for name in candidates:
        if name in part.column_names:
            return name
    return None


def class_names(part: Dataset, label_column: str | None) -> list[str] | None:
    if not label_column:
        return None
    feature = part.features.get(label_column)
    if isinstance(feature, ClassLabel):
        return list(feature.names)
    return None


def label_mapping(class_names_value: list[str] | None) -> dict[str, str] | None:
    if not class_names_value:
        return None
    return {str(index): name for index, name in enumerate(class_names_value)}


def normalize_image(value: Any, image_size: int) -> PILImage.Image:
    if isinstance(value, dict) and "path" in value and value["path"]:
        image = PILImage.open(value["path"])
    elif isinstance(value, PILImage.Image):
        image = value
    else:
        image = PILImage.fromarray(np.asarray(value))
    image = image.convert("RGB")
    return image.resize((image_size, image_size), PILImage.Resampling.BILINEAR)


def export_image_dataset(
    dataset: DatasetDict,
    output_dir: Path,
    image_column: str,
    label_column: str | None,
    image_size: int,
) -> dict[str, Any]:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    channel_sum = np.zeros(3, dtype=np.float64)
    channel_sq_sum = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    manifests: dict[str, int] = {}
    skipped: dict[str, int] = {}
    source_counts: dict[str, dict[str, int]] = {}

    for split, part in dataset.items():
        split_dir = images_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, Any]] = []
        skipped[split] = 0
        split_sources: Counter[str] = Counter()

        for idx in range(len(part)):
            try:
                example = part[idx]
            except Exception as error:
                skipped[split] += 1
                print(f"Skipping unreadable example {split}/{idx}: {error}")
                continue

            try:
                image = normalize_image(example[image_column], image_size)
                rel_path = Path("images") / split / f"{idx:08d}.jpg"
                image.save(output_dir / rel_path, quality=95)

                array = np.asarray(image, dtype=np.float32) / 255.0
                channel_sum += array.sum(axis=(0, 1))
                channel_sq_sum += np.square(array).sum(axis=(0, 1))
                pixel_count += array.shape[0] * array.shape[1]

                row = {"path": rel_path.as_posix()}
                if label_column:
                    row["label"] = example[label_column]
                if "source" in example:
                    row["source"] = example["source"]
                    split_sources.update([example["source"]])
                rows.append(row)
            except Exception as error:
                skipped[split] += 1
                print(f"Skipping image {split}/{idx}: {error}")

        manifest_path = output_dir / f"{split}.csv"
        pd.DataFrame(rows).to_csv(manifest_path, index=False)
        manifests[split] = len(rows)
        source_counts[split] = dict(sorted(split_sources.items()))

    mean = channel_sum / max(pixel_count, 1)
    std = np.sqrt((channel_sq_sum / max(pixel_count, 1)) - np.square(mean))

    return {
        "kind": "image",
        "image_column": image_column,
        "label_column": label_column,
        "image_size": image_size,
        "splits": manifests,
        "skipped": skipped,
        "source_counts": source_counts,
        "normalization": {
            "mean": mean.round(6).tolist(),
            "std": std.round(6).tolist(),
        },
    }


def text_columns(df: pd.DataFrame) -> list[str]:
    return [name for name in df.columns if pd.api.types.is_object_dtype(df[name])]


def numeric_columns(df: pd.DataFrame, exclude: set[str]) -> list[str]:
    return [
        name
        for name in df.columns
        if name not in exclude and pd.api.types.is_numeric_dtype(df[name])
    ]


def export_tabular_dataset(
    dataset: DatasetDict,
    output_dir: Path,
    label_column: str | None,
) -> dict[str, Any]:
    frames = {split: part.to_pandas() for split, part in dataset.items()}
    train_frame = frames.get("train", next(iter(frames.values())))
    excluded = {label_column} if label_column else set()
    numeric = numeric_columns(train_frame, excluded)
    text = text_columns(train_frame)

    stats: dict[str, dict[str, float]] = {}
    for column in numeric:
        mean = float(train_frame[column].mean())
        std = float(train_frame[column].std(ddof=0))
        stats[column] = {"mean": mean, "std": std if std > 0 else 1.0}

    rows_per_split: dict[str, int] = {}
    for split, frame in frames.items():
        normalized = frame.copy()
        for column in text:
            normalized[column] = normalized[column].fillna("").astype(str).str.strip()
        for column, column_stats in stats.items():
            normalized[column] = (normalized[column] - column_stats["mean"]) / column_stats["std"]

        normalized.to_parquet(output_dir / f"{split}.parquet", index=False)
        rows_per_split[split] = len(normalized)

    return {
        "kind": "tabular_or_text",
        "label_column": label_column,
        "numeric_columns": numeric,
        "text_columns": text,
        "splits": rows_per_split,
        "normalization": stats,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_hf_dataset(args.dataset, args.config, args.cache_dir)
    if args.include_kaggle:
        kaggle_dataset = load_kaggle_filtered_dataset()
        dataset = merge_image_datasets(dataset_to_dict(dataset), kaggle_dataset)

    dataset_dict = limit_rows(dataset_to_dict(dataset), args.max_rows_per_split)
    first_split = next(iter(dataset_dict.values()))
    label_column = detect_label_column(first_split, args.label_column)
    image_column = detect_image_column(first_split)

    if image_column:
        metadata = export_image_dataset(
            dataset_dict,
            output_dir,
            image_column=image_column,
            label_column=label_column,
            image_size=args.image_size,
        )
    else:
        metadata = export_tabular_dataset(dataset_dict, output_dir, label_column)

    metadata["dataset"] = args.dataset
    metadata["config"] = args.config
    detected_class_names = class_names(first_split, label_column)
    metadata["class_names"] = detected_class_names
    metadata["label_mapping"] = label_mapping(detected_class_names)

    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")

    if metadata["label_mapping"]:
        with (output_dir / "labels.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata["label_mapping"], handle, indent=2)
            handle.write("\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
