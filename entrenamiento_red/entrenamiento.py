from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import matplotlib.pyplot as plt

class DoubleTopDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data = ImageFolder(data_dir, transform=transform)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    
    @property
    def classes(self):
        return self.data.classes

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 32 * 8, 128)
        self.fc2 = nn.Linear(128, 2)  # 2 clases: con patrón de doble techo o sin él

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 8)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Definir hiperparámetros y cargar datos
batch_size = 32
learning_rate = 0.001
num_epochs = 15

transform = transforms.Compose([
    transforms.Resize((256, 64)),  # Ajustar al tamaño de las imágenes
    transforms.ToTensor()
])

dataset = ImageFolder(root='database/double_top', transform=transform)
train_size = int(0.8 * len(dataset))  # 80% de datos para entrenamiento
test_size = len(dataset) - train_size  # Resto para prueba

# Dividir el conjunto de datos en conjuntos de entrenamiento y prueba de manera aleatoria
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

# Inicializar el modelo, la función de pérdida y el optimizador
model = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.8f}'.format(epoch+1, num_epochs, i+1, total_step, loss.item()))


def test_model(model, test_loader):
    model.eval()  # Establecer el modelo en modo de evaluación
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            print(outputs)
            _, predicted = torch.max(outputs.data, 1) # https://pytorch.org/docs/stable/generated/torch.max.html
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))

# Llamar a la función de prueba después del entrenamiento
test_model(model, test_loader)

# guardar modelo
torch.save(model.state_dict(), 'model.pth')

'''
# Define transformations
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 8, 128)
        self.fc2 = nn.Linear(128, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 64 * 32 * 8)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x
    


def train(model, train_loader, criterion, optimizer, epochs=5):
    for epoch in range(epochs):
        print(f'epoch: {epoch}')
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.float().unsqueeze(1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model(inputs)
            print(outputs)
            predicted = torch.round(outputs)
            print(f'Clase real: {labels}, Clase predicha {predicted}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))

transform = transforms.Compose([
    transforms.Resize((256, 64)),  # Ajustar al tamaño de las imágenes
    transforms.ToTensor()
])

if __name__ == "__main__":
    dataset = DoubleTopDataset(data_dir='database/double_top', transform=transform)

    train_size = int(0.8 * len(dataset))  # 80% de datos para entrenamiento
    test_size = len(dataset) - train_size  # Resto para prueba

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba de manera aleatoria
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    model = CNN()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train(model, train_loader, criterion, optimizer, epochs=5)
    test(model, test_loader)
'''