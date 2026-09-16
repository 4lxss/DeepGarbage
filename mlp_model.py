import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import Compose, Resize, ToTensor

DATA_DIR = "./data"
IMG_SIZE = 64
NUM_CHANNELS = 3
HIDDEN_SIZES = [512, 128]
LEARNING_RATE = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 0.0
BATCH_SIZE = 32
VAL_SPLIT = 0.2
NUM_EPOCHS = 3

transform = Compose([Resize((IMG_SIZE, IMG_SIZE)), ToTensor()])
full_dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
NUM_CLASSES = len(full_dataset.classes)

val_size = int(len(full_dataset) * VAL_SPLIT)
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

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


if __name__ == "__main__":
    print("Detected classes:", full_dataset.classes)
    print(f"Train images: {len(train_dataset)}")
    print(f"Val images: {len(val_dataset)}")

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {num_params}")

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                predicted = outputs.argmax(dim=1)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_dataset)
        val_accuracy = 100.0 * correct / len(val_dataset)

        print(
            f"Epoch {epoch + 1}/{NUM_EPOCHS} - "
            f"Train loss: {train_loss:.4f} - "
            f"Val loss: {val_loss:.4f} - "
            f"Val accuracy: {val_accuracy:.2f}%"
        )