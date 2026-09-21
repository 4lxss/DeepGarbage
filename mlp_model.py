import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

DATA_DIR = "./data/sorted"
IMG_SIZE = 64
NUM_CHANNELS = 3
HIDDEN_SIZES = [512, 128]
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
BATCH_SIZE = 32
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
NUM_EPOCHS = 30

transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES = len(full_dataset.classes)

def stratified_split(dataset):
    """Split each class independently into train, validation, and test."""
    generator = torch.Generator().manual_seed(42)
    labels = torch.as_tensor(dataset.targets)
    train_indices, val_indices, test_indices = [], [], []

    for class_index in range(NUM_CLASSES):
        class_indices = torch.where(labels == class_index)[0]
        shuffled = class_indices[torch.randperm(len(class_indices), generator=generator)]
        train_size = int(len(shuffled) * TRAIN_SPLIT)
        val_size = int(len(shuffled) * VAL_SPLIT)
        train_indices.extend(shuffled[:train_size].tolist())
        val_indices.extend(shuffled[train_size:train_size + val_size].tolist())
        test_indices.extend(shuffled[train_size + val_size:].tolist())

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )


train_dataset, val_dataset, test_dataset = stratified_split(full_dataset)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

INPUT_DIM = IMG_SIZE * IMG_SIZE * NUM_CHANNELS


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_SIZES[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(HIDDEN_SIZES[0], HIDDEN_SIZES[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(HIDDEN_SIZES[1], NUM_CLASSES)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


model = MLP()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=LEARNING_RATE,
    momentum=MOMENTUM,
    weight_decay=WEIGHT_DECAY,
)


def run_epoch(current_model, loader, current_criterion, current_optimizer=None):
    training = current_optimizer is not None
    current_model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_items = 0
    all_labels, all_predictions = [], []

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, labels in loader:
            if training:
                current_optimizer.zero_grad()
            outputs = current_model(images)
            loss = current_criterion(outputs, labels)
            if training:
                loss.backward()
                current_optimizer.step()
            predictions = outputs.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            total_correct += (predictions == labels).sum().item()
            total_items += images.size(0)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predictions.numpy())

    return {
        "loss": total_loss / total_items,
        "accuracy": total_correct / total_items,
        "labels": all_labels,
        "predictions": all_predictions,
    }


def train_experiment(config):
    experiment_model = MLP()
    experiment_criterion = nn.CrossEntropyLoss()
    experiment_optimizer = torch.optim.SGD(
        experiment_model.parameters(),
        lr=config["learning_rate"],
        momentum=config["momentum"],
        weight_decay=config["weight_decay"],
    )
    experiment_train_loader = DataLoader(
        train_dataset, batch_size=config["batch_size"], shuffle=True
    )
    experiment_val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=False
    )
    history = {key: [] for key in (
        "train_loss", "val_loss", "train_accuracy", "val_accuracy"
    )}
    best_state = None
    best_val_loss = float("inf")

    for epoch in range(config["epochs"]):
        train_metrics = run_epoch(
            experiment_model, experiment_train_loader,
            experiment_criterion, experiment_optimizer
        )
        val_metrics = run_epoch(
            experiment_model, experiment_val_loader, experiment_criterion
        )
        for key, value in (
            ("train_loss", train_metrics["loss"]),
            ("val_loss", val_metrics["loss"]),
            ("train_accuracy", train_metrics["accuracy"]),
            ("val_accuracy", val_metrics["accuracy"]),
        ):
            history[key].append(value)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = copy.deepcopy(experiment_model.state_dict())
        print(
            f"{config['name']} - Epoch {epoch + 1}/{config['epochs']} - "
            f"Train loss: {train_metrics['loss']:.4f} - "
            f"Train accuracy: {100 * train_metrics['accuracy']:.2f}% - "
            f"Val loss: {val_metrics['loss']:.4f} - "
            f"Val accuracy: {100 * val_metrics['accuracy']:.2f}%"
        )

    experiment_model.load_state_dict(best_state)
    return {
        **config,
        "model": experiment_model,
        "history": history,
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "best_val_loss": min(history["val_loss"]),
        "best_val_accuracy": max(history["val_accuracy"]),
    }


def plot_history(result):
    history = result["history"]
    epochs = range(1, len(history["train_loss"]) + 1)
    figure, axes = plt.subplots(1, 2, figsize=(13, 4))
    axes[0].plot(epochs, history["train_loss"], label="train")
    axes[0].plot(epochs, history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[1].plot(epochs, history["train_accuracy"], label="train")
    axes[1].plot(epochs, history["val_accuracy"], label="validation")
    axes[1].set_title("Accuracy")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    figure.tight_layout()
    plt.show()


def evaluate_model(result):
    evaluation_loader = DataLoader(
        test_dataset, batch_size=result["batch_size"], shuffle=False
    )
    metrics = run_epoch(result["model"], evaluation_loader, nn.CrossEntropyLoss())
    labels = metrics["labels"]
    predictions = metrics["predictions"]
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    metrics_table = pd.DataFrame({
        "metric": [
            "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "precision_weighted", "recall_weighted", "f1_weighted",
        ],
        "value": [
            accuracy_score(labels, predictions), precision_macro, recall_macro,
            f1_macro, precision_weighted, recall_weighted, f1_weighted,
        ],
    })
    print("\nFinal evaluation on test:")
    print(metrics_table.to_string(index=False))
    print(classification_report(
        labels, predictions, target_names=full_dataset.classes, zero_division=0
    ))
    confusion = confusion_matrix(labels, predictions, labels=range(NUM_CLASSES))
    plt.imshow(confusion, interpolation="nearest", cmap="Blues")
    plt.colorbar()
    threshold = confusion.max() / 2.0
    for row in range(NUM_CLASSES):
        for column in range(NUM_CLASSES):
            color = "white" if confusion[row, column] > threshold else "black"
            plt.text(column, row, confusion[row, column], ha="center", va="center", color=color)
    plt.xticks(range(NUM_CLASSES), full_dataset.classes, rotation=45, ha="right")
    plt.yticks(range(NUM_CLASSES), full_dataset.classes)
    plt.xlabel("Predicted class")
    plt.ylabel("True class")
    plt.title("MLP confusion matrix")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print("Detected classes:", full_dataset.classes)
    print(f"Train images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Test images: {len(test_dataset)}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

    baseline_config = {
        "name": "baseline",
        "learning_rate": LEARNING_RATE,
        "momentum": MOMENTUM,
        "weight_decay": WEIGHT_DECAY,
        "batch_size": BATCH_SIZE,
        "epochs": NUM_EPOCHS,
    }
    configs = [
        baseline_config,
        {**baseline_config, "name": "lr=0.001", "learning_rate": 0.001},
        {**baseline_config, "name": "lr=0.05", "learning_rate": 0.05},
        {**baseline_config, "name": "batch=16", "batch_size": 16},
        {**baseline_config, "name": "batch=64", "batch_size": 64},
        {**baseline_config, "name": "weight_decay=1e-4", "weight_decay": 1e-4},
    ]
    results = [train_experiment(config) for config in configs]
    results_table = pd.DataFrame([
        {key: value for key, value in result.items()
         if key not in ("model", "history")}
        for result in results
    ]).sort_values("best_val_accuracy", ascending=False)
    print("\nExperiment comparison:")
    print(results_table.to_string(index=False))

    best_result = max(results, key=lambda result: result["best_val_accuracy"])
    print(f"\nSelected model: {best_result['name']}")
    plot_history(best_result)
    best_index = best_result["best_epoch"] - 1
    history = best_result["history"]
    accuracy_gap = history["train_accuracy"][best_index] - history["val_accuracy"][best_index]
    print(f"Train-validation accuracy gap: {100 * accuracy_gap:.2f} points")
    if accuracy_gap > 0.10:
        print("Diagnostic: possible overfitting.")
    evaluate_model(best_result)