from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

from network_for_each_pattern import train_model, test_model, get_test_and_train_loaders

class Dataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes
    

class CNN_all(nn.Module):
    def __init__(self):
        super(CNN_all, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 8, 128)
        self.fc2 = nn.Linear(128, 7)  # 2 clases: con patrón de doble techo o sin él

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    


if __name__ == "__main__":
    # Definir hiperparámetros y cargar datos
    batch_size = 32
    learning_rate = 0.001
    num_epochs = 20

    train_loader, test_loader = get_test_and_train_loaders('database_red_conjunta', batch_size)
    # Inicializar el modelo, la función de pérdida y el optimizador
    model = CNN_all()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_model(model, train_loader, criterion, optimizer, num_epochs)
    test_model(model, test_loader)
    torch.save(model.state_dict(), 'all_patterns_model.pth')



