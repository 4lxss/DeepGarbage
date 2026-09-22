"""Benchmark CNN équitable : même protocole que le MLP déjà réalisé.

Le test est volontairement protégé : les variantes sont choisies uniquement
avec la validation, puis `--final-test` mesure une seule fois le gagnant.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deepgarbage_matplotlib"))

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from tensorflow.keras import layers, models


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "benchmark"
RESULTS_DIR = ROOT / "results" / "fair_benchmark" / "cnn"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (64, 64)
BATCH_SIZE = 32
EPOCHS = 30
SEED = 42
# Variantes CNN : une seule caractéristique est modifiée à chaque fois.
CONFIGURATIONS = {
    "baseline": {"learning_rate": 1e-3, "batch_size": 32, "l2": 1e-4},
    "low_learning_rate": {"learning_rate": 3e-4, "batch_size": 32, "l2": 1e-4},
    "batch_16": {"learning_rate": 1e-3, "batch_size": 16, "l2": 1e-4},
    "batch_64": {"learning_rate": 1e-3, "batch_size": 64, "l2": 1e-4},
    "stronger_l2": {"learning_rate": 1e-3, "batch_size": 32, "l2": 5e-4},
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configuration", choices=CONFIGURATIONS, default="baseline")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--final-test", action="store_true", help="Évalue le meilleur CNN sur le test, une fois le choix fait.")
    return parser.parse_args()


def dataset(split: str, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    frame = pd.read_csv(DATA_DIR / f"{split}.csv")
    paths = [str(ROOT / path) for path in frame.path]
    labels = frame.label.astype("int32").to_numpy()
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_image(tf.io.read_file(path), channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0, label

    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def build_cnn(config: dict[str, float]) -> tf.keras.Model:
    """Architecture d'Aless, adaptée seulement à l'entrée commune 64x64."""
    augmentation = tf.keras.Sequential([
        layers.RandomFlip("horizontal_and_vertical"),
        layers.RandomRotation(0.2),
        layers.RandomZoom(0.2),
    ])
    model = models.Sequential([
        layers.Input(shape=(*IMAGE_SIZE, 3)), augmentation,
        layers.Conv2D(32, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding="same"), layers.BatchNormalization(), layers.Activation("relu"), layers.MaxPooling2D(),
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation="relu", kernel_regularizer=tf.keras.regularizers.l2(config["l2"])),
        layers.Dropout(0.4), layers.Dense(len(CLASS_NAMES), activation="softmax"),
    ], name="aless_cnn_64x64")
    model.compile(optimizer=tf.keras.optimizers.Adam(config["learning_rate"]), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def plot_history(history: tf.keras.callbacks.History, name: str) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for axis, metric, title in zip(axes, ("loss", "accuracy"), ("Loss", "Accuracy")):
        axis.plot(history.history[metric], label="train")
        axis.plot(history.history[f"val_{metric}"], label="validation")
        axis.set(title=title, xlabel="Époque", ylabel=title)
        axis.legend(); axis.grid(alpha=.25)
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / f"{name}_curves.png", dpi=180)
    plt.close(figure)


def train(name: str, epochs: int) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tf.keras.utils.set_random_seed(SEED)
    config = CONFIGURATIONS[name]
    model = build_cnn(config)
    # Même règle que le MLP : 30 époques prévues et conservation des poids
    # correspondant à la plus faible loss de validation.
    callbacks = [tf.keras.callbacks.ModelCheckpoint(
        RESULTS_DIR / f"{name}.keras", monitor="val_loss", save_best_only=True, verbose=1
    )]
    history = model.fit(dataset("train", config["batch_size"], True), validation_data=dataset("validation", config["batch_size"]), epochs=epochs, callbacks=callbacks, verbose=2)
    model = tf.keras.models.load_model(RESULTS_DIR / f"{name}.keras")
    metrics = model.evaluate(dataset("validation", config["batch_size"]), return_dict=True, verbose=0)
    plot_history(history, name)
    row = {"configuration": name, **config, "epochs_ran": len(history.history["loss"]), "validation_loss": metrics["loss"], "validation_accuracy": metrics["accuracy"]}
    pd.DataFrame([row]).to_csv(RESULTS_DIR / f"{name}_metrics.csv", index=False)
    metrics_files = RESULTS_DIR.glob("*_metrics.csv")
    comparison = pd.concat((pd.read_csv(file) for file in metrics_files), ignore_index=True).sort_values("validation_accuracy", ascending=False)
    comparison.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    print("\nChoix sur validation (le test reste intact) :")
    print(comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def final_test() -> None:
    comparison_path = RESULTS_DIR / "comparison.csv"
    if not comparison_path.exists():
        raise FileNotFoundError("Entraîne au moins une configuration avant le test final.")
    winner = pd.read_csv(comparison_path).iloc[0]["configuration"]
    model = tf.keras.models.load_model(RESULTS_DIR / f"{winner}.keras")
    batch_size = int(pd.read_csv(comparison_path).iloc[0]["batch_size"])
    test_ds = dataset("test", batch_size)
    metrics = model.evaluate(test_ds, return_dict=True, verbose=2)
    labels, predictions = [], []
    for images, batch_labels in test_ds:
        labels.extend(batch_labels.numpy())
        predictions.extend(model.predict(images, verbose=0).argmax(axis=1))
    precision, recall, f1, support = precision_recall_fscore_support(labels, predictions, labels=range(6), zero_division=0)
    per_class = pd.DataFrame({"class": CLASS_NAMES, "precision": precision, "recall": recall, "f1": f1, "support": support})
    per_class.to_csv(RESULTS_DIR / "test_metrics_per_class.csv", index=False)
    matrix = confusion_matrix(labels, predictions, labels=range(6))
    pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(RESULTS_DIR / "test_confusion_matrix.csv")
    figure, axis = plt.subplots(figsize=(7, 6))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(6), yticks=range(6), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Classe prédite", ylabel="Classe réelle", title="Matrice de confusion — CNN")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row in range(6):
        for column in range(6):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.tight_layout()
    figure.savefig(RESULTS_DIR / "test_confusion_matrix.png", dpi=180)
    plt.close(figure)
    macro = precision_recall_fscore_support(labels, predictions, average="macro", zero_division=0)
    payload = {"model": "CNN (architecture Aless)", "configuration": winner, "test_loss": float(metrics["loss"]), "test_accuracy": float(metrics["accuracy"]), "precision_macro": float(macro[0]), "recall_macro": float(macro[1]), "f1_macro": float(macro[2])}
    (RESULTS_DIR / "final_test.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nTEST FINAL :", payload)


if __name__ == "__main__":
    args = arguments()
    final_test() if args.final_test else train(args.configuration, args.epochs)
