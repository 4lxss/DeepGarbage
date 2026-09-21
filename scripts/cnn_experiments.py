

from __future__ import annotations

import argparse
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deepgarbage_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models

from cnn_evaluation import (
    EVALUATION_CSV,
    OUTPUT_DIR,
    SEED,
    TRAIN_CSV,
    make_dataset,
    split_validation_and_test,
)


EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
BATCH_SIZE = 32
NUM_CLASSES = 6


@dataclass(frozen=True)
class Configuration:
    name: str
    filters: tuple[int, int, int] = (32, 64, 128)
    learning_rate: float = 1e-3
    l2_strength: float = 1e-4


CONFIGURATIONS = {
    "learning_rate_0_0003": Configuration("learning_rate_0_0003", learning_rate=3e-4),
    "more_filters_64_128_256": Configuration("more_filters_64_128_256", filters=(64, 128, 256)),
    "stronger_l2_0_0005": Configuration("stronger_l2_0_0005", l2_strength=5e-4),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Lancer une expérience CNN contrôlée.")
    parser.add_argument("--experiment", choices=CONFIGURATIONS, default="learning_rate_0_0003",
                        help="Paramètre unique à modifier par rapport à la baseline.")
    parser.add_argument("--epochs", type=int, default=25,
                        help="25 pour une comparaison exacte avec Aless; réduire seulement pour un essai rapide.")
    return parser.parse_args()


def build_cnn(config: Configuration) -> tf.keras.Model:
    """Architecture d'Aless, avec le seul paramètre testé rendu configurable."""
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ], name="data_augmentation")

    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)),
        augmentation,
        layers.Conv2D(config.filters[0], (3, 3), padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(config.filters[1], (3, 3), padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D((2, 2)),
        layers.Conv2D(config.filters[2], (3, 3), padding="same"),
        layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config.l2_strength)),
        layers.Dropout(0.4),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ], name=config.name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config.learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def plot_learning_curves(history: tf.keras.callbacks.History, configuration: Configuration) -> None:
    epochs = range(1, len(history.history["loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        axis.plot(epochs, history.history[metric], label="train")
        axis.plot(epochs, history.history[f"val_{metric}"], label="validation")
        axis.set(xlabel="Époque", ylabel=title, title=f"{title} — {configuration.name}")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(EXPERIMENTS_DIR / f"{configuration.name}_curves.png", dpi=180)
    plt.close(figure)


def run_experiment(
    configuration: Configuration,
    train_ds: tf.data.Dataset,
    train_eval_ds: tf.data.Dataset,
    validation_ds: tf.data.Dataset,
    epochs: int,
) -> dict[str, object]:
    tf.keras.utils.set_random_seed(SEED)
    model = build_cnn(configuration)
    history = model.fit(train_ds, validation_data=validation_ds, epochs=epochs, verbose=2)
    train_metrics = model.evaluate(train_eval_ds, verbose=0, return_dict=True)
    validation_metrics = model.evaluate(validation_ds, verbose=0, return_dict=True)
    plot_learning_curves(history, configuration)
    model.save(EXPERIMENTS_DIR / f"{configuration.name}.keras")
    return {
        "configuration": configuration.name,
        "filters": "-".join(map(str, configuration.filters)),
        "learning_rate": configuration.learning_rate,
        "l2_strength": configuration.l2_strength,
        "epochs": epochs,
        "train_loss": float(train_metrics["loss"]),
        "validation_loss": float(validation_metrics["loss"]),
        "train_accuracy": float(train_metrics["accuracy"]),
        "validation_accuracy": float(validation_metrics["accuracy"]),
    }


def baseline_row() -> dict[str, object]:
    baseline = pd.read_csv(OUTPUT_DIR / "baseline_metrics.csv").set_index("split")
    return {
        "configuration": "baseline_Aless",
        "filters": "32-64-128",
        "learning_rate": 1e-3,
        "l2_strength": 1e-4,
        "epochs": 25,
        "train_loss": baseline.loc["train", "loss"],
        "validation_loss": baseline.loc["validation", "loss"],
        "train_accuracy": baseline.loc["train", "accuracy"],
        "validation_accuracy": baseline.loc["validation", "accuracy"],
    }


def main() -> None:
    args = parse_args()
    if not (OUTPUT_DIR / "baseline_metrics.csv").exists():
        raise FileNotFoundError("Lance d'abord : ./.venv/bin/python scripts/cnn_evaluation.py")
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    train_frame = pd.read_csv(TRAIN_CSV)
    evaluation_frame = pd.read_csv(EVALUATION_CSV)
    validation_frame, _ = split_validation_and_test(evaluation_frame)
    train_ds = make_dataset(train_frame, BATCH_SIZE, shuffle=True)
    train_eval_ds = make_dataset(train_frame, BATCH_SIZE)
    validation_ds = make_dataset(validation_frame, BATCH_SIZE)

    configuration = CONFIGURATIONS[args.experiment]
    print(f"\n===== {configuration.name} ({args.epochs} époques) =====")
    result = run_experiment(configuration, train_ds, train_eval_ds, validation_ds, args.epochs)
    pd.DataFrame([result]).to_csv(EXPERIMENTS_DIR / f"{configuration.name}_metrics.csv", index=False)
    tf.keras.backend.clear_session()

    results = [baseline_row()]
    for metrics_file in EXPERIMENTS_DIR.glob("*_metrics.csv"):
        results.append(pd.read_csv(metrics_file).iloc[0].to_dict())
    table = pd.DataFrame(results).sort_values("validation_accuracy", ascending=False)
    table.to_csv(EXPERIMENTS_DIR / "comparison.csv", index=False)
    print("\nTableau comparatif (choix sur validation uniquement) :")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nFichiers créés dans : {EXPERIMENTS_DIR}")


if __name__ == "__main__":
    main()
