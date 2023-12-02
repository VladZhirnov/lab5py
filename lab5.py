import os
from typing import Tuple, Any
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
import random

class CustomImageDataset(Dataset):
    def __init__(self, path_to_annotation_file: str, transform: Any=None, target_transform: Any=None) -> None:
        self.path_to_annotation_file = path_to_annotation_file
        self.dataset_info = pd.read_csv(path_to_annotation_file, header=None, names=['path_to_image', 'label'])
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        return len(self.dataset_info)

    def __getitem__(self, index: int) -> Tuple[torch.tensor, int]:
        path_to_image = self.dataset_info.iloc[index, 0]
        image = cv2.cvtColor(cv2.imread(path_to_image), cv2.COLOR_BGR2RGB)
        label = self.dataset_info.iloc[index, 1]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)

        return image, label

# Пайплайн предобработки данных
custom_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

path_to_dataset = "dataset"

custom_dataset = CustomImageDataset(os.path.join(path_to_dataset, "annotation3.csv"), custom_transforms)

# Разделение данных
total_size = len(custom_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(custom_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Загрузчики данных
batch_size = 32  
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

class CNN(nn.Module):
    def __init__(self) -> None:
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(576, 10)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.layer1(x)
        output = self.layer2(output)
        output = self.layer3(output)
        output = torch.nn.Flatten()(output)
        output = self.relu(self.fc1(output))
        output = self.fc2(output)
        return torch.nn.Sigmoid()(output)

# Эксперименты с learning rate и batch size
learning_rates = [0.001, 0.01, 0.1]
batch_sizes = [16, 32, 64]

for lr in learning_rates:
    for bs in batch_sizes:
        print(f"\nTraining with learning rate: {lr}, batch size: {bs}")

        # Инициализация модели, функции потерь и оптимизатора
        model = CNN()
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=lr)

        # Перенос модели на устройство 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Цикл обучения
        num_epochs = 5
        train_losses = []  
        val_losses = []    
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels.float().view(-1, 1))
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            train_losses.append(running_loss / len(train_dataloader))

            print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_losses[-1]:.4f}")

            # Валидация
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, labels in val_dataloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels.float().view(-1, 1))
                    val_loss += loss.item()

            val_losses.append(val_loss / len(val_dataloader))
            print(f"Validation Loss: {val_losses[-1]:.4f}")

        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(range(1, num_epochs + 1), train_losses, label='Train')
        plt.plot(range(1, num_epochs + 1), val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.show()


# Оценка модели на тестовой выборке
model.eval()

test_loss = 0
test_accuracy = 0

with torch.no_grad():
    for data, label in test_dataloader:
        data = data.to(device)
        label = label.to(device)

        output = model(data)
        loss = criterion(output, label.float().unsqueeze(dim=1))

        predictions = (output >= 0.5).int()
        correct_predictions = (predictions == label.int().view(-1, 1)).sum().item()

        test_accuracy += correct_predictions / len(test_dataloader.dataset)
        test_loss += loss.item()

test_accuracy /= len(test_dataloader)
test_loss /= len(test_dataloader)

print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Сохранение обученной модели
torch.save(model.state_dict(), os.path.join("dataset", "weight.pt"))
print("Model saved to dataset")

# Загрузка сохраненной модели
loaded_model = CNN()
loaded_model.load_state_dict(torch.load(os.path.join("dataset", "weight.pt")))
loaded_model.to(device)
