import os
from typing import Tuple, Any
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision
import torch.nn as nn

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

# Загрузка данных
path_to_dataset = "dataset"
df = pd.read_csv(os.path.join(path_to_dataset, "annotation3.csv"), header=None, names=['path_to_image', 'label'])

# Пайплайн предобработки данных
custom_transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                    torchvision.transforms.Resize((224, 224)),
                                                    torchvision.transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

# Создание объекта датасета
custom_dataset = CustomImageDataset(os.path.join(path_to_dataset, "annotation3.csv"), custom_transforms)

# Разделение данных
total_size = len(custom_dataset)
train_size = int(0.8 * total_size)
val_size = int(0.1 * total_size)
test_size = total_size - train_size - val_size

train_data, val_data, test_data = random_split(custom_dataset, [train_size, val_size, test_size], generator=torch.Generator().manual_seed(42))

# Создание тензоров
train_labels_tensor = torch.tensor(train_data.dataset.dataset_info['label'].values)
val_labels_tensor = torch.tensor(val_data.dataset.dataset_info['label'].values)
test_labels_tensor = torch.tensor(test_data.dataset.dataset_info['label'].values)

# Проверка баланса классов
train_class_distribution = torch.bincount(train_labels_tensor)
val_class_distribution = torch.bincount(val_labels_tensor)
test_class_distribution = torch.bincount(test_labels_tensor)

print("Train Class Distribution:")
print(train_class_distribution)

print("\nValidation Class Distribution:")
print(val_class_distribution)

print("\nTest Class Distribution:")
print(test_class_distribution)

# Визуализация изображений
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("tiger" if custom_dataset[0][1] == 0 else "leopard")
plt.imshow(custom_dataset[0][0].permute(1, 2, 0).numpy()[:, :, ::-1])
plt.subplot(1, 2, 2)
plt.title("tiger" if custom_dataset[500][1] == 0 else "leopard")
plt.imshow(custom_dataset[500][0].permute(1, 2, 0).numpy()[:, :, ::-1])
plt.show()

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
    
