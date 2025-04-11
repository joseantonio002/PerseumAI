from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import io
from PIL import Image
from matplotlib.figure import Figure


class PatternsDataset(Dataset):
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


transform = transforms.Compose([
    transforms.Resize((256, 64)),  # Ajustar al tamaño de las imágenes
    transforms.ToTensor()
])

def train_model(model, train_loader, criterion, optimizer, num_epochs):
    """
    Trains the given model using the specified training data, criterion, optimizer, and number of epochs
    
    Args:
        model (nn.Module): The model to be trained
        train_loader (DataLoader): DataLoader for the training data
        criterion: The loss function
        optimizer: The optimization algorithm
        num_epochs (int): Number of epochs for training
    """
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
    """
    Tests the given model using the specified test data loader
    
    Args:
        model (nn.Module): The model to be tested
        test_loader (DataLoader): DataLoader for the test data
    
    Returns:
        float: The accuracy of the model on the test set
    """
    model.eval()  # Establecer el modelo en modo de evaluación
    
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            #print(outputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    print('Accuracy on test set: {:.2f}%'.format(100 * accuracy))

def get_test_and_train_loaders(data_dir, batch_size):
    """
    Splits the dataset into training and test sets, and returns the corresponding data loaders
    
    Args:
        data_dir (str): Directory containing the data
        batch_size (int): Batch size for the data loaders
    
    Returns:
        DataLoader, DataLoader: Training and test data loaders
    """
    dataset = ImageFolder(root=data_dir, transform=transform)
    train_size = int(0.8 * len(dataset))  # 80% de datos para entrenamiento
    test_size = len(dataset) - train_size  # Resto para prueba

    # Dividir el conjunto de datos en conjuntos de entrenamiento y prueba de manera aleatoria
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)
    return train_loader, test_loader


pattern_index = {
    'double_top': 0,
    'double_bottom': 1,
    'ascending_triangle': 2,
    'descending_triangle': 3,
    'head_and_shoulders': 4,
    'inv_head_and_shoulders': 5
}

index_pattern = {
    0: 'double_top',
    1: 'double_bottom',
    2: 'ascending_triangle',
    3: 'descending_triangle',
    4: 'head_and_shoulders',
    5: 'inv_head_and_shoulders'
}

def classify(image_tensor, models):
    """
    Classifies the given image tensor using the provided models
    
    Args:
        image_tensor (Tensor): The image tensor to be classified
        models (dict): Dictionary of models for each pattern type
    
    Returns:
        int, float: The predicted pattern index and the confidence of the prediction
    """
    predicted_final = -1
    confidence_final = -1
    for pattern, model in models.items():
        model.eval() 
        with torch.no_grad():
            outputs = model(image_tensor)
            confidence, predicted = torch.max(outputs.data, 1)
            if predicted == 0:
                confidence_final = confidence.item()
                predicted_final = pattern_index[pattern]
                break
    return predicted_final, confidence_final

def pattern_to_image(pattern):
    """
    Converts a given pattern to an image, the pattern is NOT a 
    Pattern object, but a list of values
    
    Args:
        patron (list[int]): List of values representing the pattern
    
    Returns:
        Image: The resulting image of the pattern
    """
    dpi = 100
    figsize_x = 256 / dpi
    figsize_y = 64 / dpi
    fig = Figure(figsize=(figsize_x, figsize_y), dpi=dpi)
    ax = fig.add_subplot(111)
    ax.plot(pattern, color='black', linewidth=1)
    ax.axis('off')
    fig.patch.set_visible(False)
    fig.tight_layout(pad=0)
    # Save the figure to a BytesIO object
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    plt.close(fig)
    buf.close()
    return image







if __name__ == "__main__":
    print("This is a module, it should not be run as a script.")