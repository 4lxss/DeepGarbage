"""Partie Lysa : évaluation du CNN de référence d'Aless.

Aless a entraîné ``cnn_model.keras`` sur ``data/processed/train.csv`` (70 %).
Les images de ``val.csv`` (30 %, jamais vues par le modèle) sont réparties ici
en validation et test de manière stratifiée : 15 % / 15 % du dataset complet.

Usage :
    ./.venv/bin/python scripts/cnn_evaluation.py
    ./.venv/bin/python scripts/cnn_evaluation.py --final-test

Sans ``--final-test``, seules les métriques train/validation sont calculées.
Le test est volontairement réservé à l'évaluation finale du modèle retenu.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "deepgarbage_matplotlib"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "data" / "processed" / "train.csv"
EVALUATION_CSV = ROOT / "data" / "processed" / "val.csv"
MODEL_PATH = ROOT / "cnn_model.keras"
OUTPUT_DIR = ROOT / "results" / "cnn_evaluation"
CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
IMAGE_SIZE = (224, 224)  # résolution utilisée par Aless
SEED = 42
# Historique transmis par Aless après l'entraînement de la baseline.
BASELINE_TRAIN_ACCURACY = [
    0.4337, 0.4963, 0.5270, 0.5489, 0.5659, 0.5804, 0.5872, 0.5984,
    0.6105, 0.6119, 0.6246, 0.6321, 0.6333, 0.6429, 0.6462, 0.6461,
    0.6519, 0.6558, 0.6645, 0.6662, 0.6719, 0.6746, 0.6779, 0.6813,
    0.6833,
]
BASELINE_TRAIN_LOSS = [
    1.4728, 1.3408, 1.2781, 1.2305, 1.1918, 1.1677, 1.1389, 1.1142,
    1.0947, 1.0804, 1.0602, 1.0362, 1.0298, 1.0099, 1.0016, 0.9947,
    0.9824, 0.9765, 0.9610, 0.9488, 0.9378, 0.9302, 0.9118, 0.9147,
    0.9089,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évaluer le CNN d'Aless.")
    parser.add_argument("--final-test", action="store_true",
                        help="Évalue aussi le test et crée la matrice de confusion finale.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH,
                        help="Modèle à évaluer; par défaut, la baseline d'Aless.")
    parser.add_argument("--plot-history-only", action="store_true",
                        help="Crée seulement le graphique baseline à partir du log d'Aless.")
    parser.add_argument("--test-only", action="store_true",
                        help="Réutilise les métriques train/validation déjà enregistrées et évalue seulement le test.")
    return parser.parse_args()


def split_validation_and_test(evaluation_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partage les 30 % non vus en validation et test, classe par classe."""
    test_frame = (
        evaluation_frame.groupby("label", group_keys=False)
        .sample(frac=0.5, random_state=SEED)
    )
    validation_frame = evaluation_frame.drop(test_frame.index)
    return validation_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def make_dataset(frame: pd.DataFrame, batch_size: int, shuffle: bool = False) -> tf.data.Dataset:
    """Même prétraitement que celui utilisé par Aless : RGB, 224x224, [0, 1]."""
    paths = [str((ROOT / relative_path).resolve()) for relative_path in frame["path"]]
    labels = frame["label"].astype("int32").to_numpy()
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
        image = tf.image.resize(image, IMAGE_SIZE)
        return tf.cast(image, tf.float32) / 255.0, label

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        dataset = dataset.shuffle(len(frame), seed=SEED, reshuffle_each_iteration=True)
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def make_keras_compatible_copy(source: Path) -> Path:
    """Adapte temporairement des métadonnées Keras, sans modifier le modèle d'Aless."""
    target = Path(tempfile.gettempdir()) / "deepgarbage_cnn_model_compatible.keras"
    if target.exists() and target.stat().st_mtime >= source.stat().st_mtime:
        return target

    def clean_config(value: object) -> None:
        if isinstance(value, dict):
            config = value.get("config")
            class_name = value.get("class_name")
            if isinstance(config, dict):
                if class_name in {"GlorotUniform", "GlorotNormal", "VarianceScaling"}:
                    config.pop("input_axes", None)
                    config.pop("output_axes", None)
                if class_name == "BatchNormalization":
                    for key in ("renorm", "renorm_clipping", "renorm_momentum"):
                        config.pop(key, None)
                config.pop("quantization_config", None)
            for child in value.values():
                clean_config(child)
        elif isinstance(value, list):
            for child in value:
                clean_config(child)

    with zipfile.ZipFile(source) as original, zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as rebuilt:
        for item in original.infolist():
            content = original.read(item.filename)
            if item.filename == "config.json":
                config = json.loads(content)
                clean_config(config)
                content = json.dumps(config, separators=(",", ":")).encode()
            rebuilt.writestr(item, content)
    return target


def load_model(model_path: Path) -> tf.keras.Model:
    model_path = model_path.resolve()
    # Seule la baseline a été enregistrée par une version plus récente de Keras.
    path_to_load = make_keras_compatible_copy(model_path) if model_path == MODEL_PATH.resolve() else model_path
    model = tf.keras.models.load_model(path_to_load, compile=False)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )
    return model


def evaluate(model: tf.keras.Model, dataset: tf.data.Dataset, split_name: str) -> dict[str, float]:
    metrics = model.evaluate(dataset, verbose=2, return_dict=True)
    return {"split": split_name, "loss": float(metrics["loss"]), "accuracy": float(metrics["accuracy"])}


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, pd.DataFrame]:
    matrix = tf.math.confusion_matrix(y_true, y_pred, num_classes=len(CLASS_NAMES)).numpy()
    correct = np.diag(matrix).astype(float)
    precision = np.divide(correct, matrix.sum(axis=0), out=np.zeros_like(correct), where=matrix.sum(axis=0) != 0)
    recall = np.divide(correct, matrix.sum(axis=1), out=np.zeros_like(correct), where=matrix.sum(axis=1) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(correct), where=(precision + recall) != 0)
    report = pd.DataFrame({"class": CLASS_NAMES, "precision": precision, "recall": recall, "f1": f1, "support": matrix.sum(axis=1)})
    return matrix, report


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Nombre d'images")
    axis.set(xticks=np.arange(6), yticks=np.arange(6), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Classe prédite", ylabel="Classe réelle", title="Matrice de confusion — test")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max() / 2
    for row in range(6):
        for column in range(6):
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center",
                      color="white" if matrix[row, column] > threshold else "black")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_baseline_history(validation_loss: float, validation_accuracy: float) -> None:
    """Trace le log train d'Aless et le seul point validation réellement disponible."""
    epochs = range(1, len(BASELINE_TRAIN_LOSS) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for axis, train_values, validation_value, title in (
        (axes[0], BASELINE_TRAIN_LOSS, validation_loss, "Loss"),
        (axes[1], BASELINE_TRAIN_ACCURACY, validation_accuracy, "Accuracy"),
    ):
        axis.plot(epochs, train_values, label="train")
        axis.scatter([25], [validation_value], color="crimson", zorder=3,
                     label="validation après entraînement")
        axis.set(xlabel="Époque", ylabel=title, title=f"Baseline — {title}")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    figure.savefig(OUTPUT_DIR / "baseline_learning_curves.png", dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.plot_history_only:
        metrics_file = OUTPUT_DIR / "baseline_metrics.csv"
        if not metrics_file.exists():
            raise FileNotFoundError("Lance d'abord l'évaluation baseline.")
        metrics = pd.read_csv(metrics_file).set_index("split")
        plot_baseline_history(metrics.loc["validation", "loss"], metrics.loc["validation", "accuracy"])
        print(f"Graphique créé : {OUTPUT_DIR / 'baseline_learning_curves.png'}")
        return
    if args.test_only and not args.final_test:
        raise ValueError("--test-only doit être utilisé avec --final-test.")
    train_frame = pd.read_csv(TRAIN_CSV)
    evaluation_frame = pd.read_csv(EVALUATION_CSV)
    validation_frame, test_frame = split_validation_and_test(evaluation_frame)
    print(f"Train       : {len(train_frame)} images (Aless)")
    print(f"Validation  : {len(validation_frame)} images")
    print(f"Test        : {len(test_frame)} images")

    validation_frame.to_csv(OUTPUT_DIR / "validation_split.csv", index=False)
    test_frame.to_csv(OUTPUT_DIR / "test_split.csv", index=False)
    model = load_model(args.model_path)
    if model.input_shape[1:3] != IMAGE_SIZE:
        raise ValueError(f"Taille inattendue du modèle : {model.input_shape}; attendu 224x224.")

    if args.test_only:
        previous_metrics = OUTPUT_DIR / "baseline_metrics.csv"
        if not previous_metrics.exists():
            raise FileNotFoundError("Les métriques baseline train/validation sont introuvables.")
        rows = pd.read_csv(previous_metrics).query("split != 'test'").to_dict("records")
    else:
        rows = [
            evaluate(model, make_dataset(train_frame, args.batch_size), "train"),
            evaluate(model, make_dataset(validation_frame, args.batch_size), "validation"),
        ]
    if args.final_test:
        test_dataset = make_dataset(test_frame, args.batch_size)
        rows.append(evaluate(model, test_dataset, "test"))
        predictions = model.predict(test_dataset, verbose=2).argmax(axis=1)
        matrix, report = classification_report(test_frame["label"].to_numpy(), predictions)
        report.to_csv(OUTPUT_DIR / "test_metrics_per_class.csv", index=False)
        plot_confusion_matrix(matrix, OUTPUT_DIR / "test_confusion_matrix.png")

    results = pd.DataFrame(rows)
    results_file = "baseline_metrics.csv" if args.model_path.resolve() == MODEL_PATH.resolve() else "final_model_metrics.csv"
    results.to_csv(OUTPUT_DIR / results_file, index=False)
    if args.model_path.resolve() == MODEL_PATH.resolve():
        validation_result = results.loc[results["split"] == "validation"].iloc[0]
        plot_baseline_history(validation_result["loss"], validation_result["accuracy"])
    print("\nRésultats :")
    print(results.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    if not args.final_test:
        print("\nLe test n'a pas été utilisé. Lance --final-test seulement après le choix final du modèle.")
    print(f"Résultats enregistrés dans : {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
