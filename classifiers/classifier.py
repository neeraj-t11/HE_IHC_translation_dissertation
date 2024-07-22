import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import os

# Check if CUDA is available, else use CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Define the IHC Classifier based on ResNet
class IHCClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(IHCClassifier, self).__init__()
        self.resnet = models.resnet101(pretrained=True)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Custom Dataset to handle IHC image grades
class IHCDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory)]
        self.labels = [self.extract_label(img) for img in self.images]

    def extract_label(self, image_path):
        return int(image_path.split('_')[-1].replace('+', '').replace('.png', ''))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Define transformations
transformations = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Prepare data loaders
train_dataset = IHCDataset('./datasets/BCI/B/train', transform=transformations)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = IHCDataset('./datasets/BCI/B/test', transform=transformations)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Initialize the model and optimizer
model = IHCClassifier().to(device)  # This line ensures model is on GPU if available
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Training the classifier
def train(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    print("Training started...")
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)  # Move data to the same device as the model
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training finished. Loss: {total_loss / len(dataloader)}")

# Testing Function with Class-specific Accuracy
def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    class_correct = list(0. for i in range(4))
    class_total = list(0. for i in range(4))
    print("Testing started...")
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    print(f"Testing finished. Loss: {total_loss / len(dataloader)}")
    for i in range(4):
        print(f'Accuracy of {i}+ : {100 * class_correct[i] / class_total[i]}%')

# Example training and testing call
if __name__ == "__main__":
    # Check if CUDA is available, else use CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    train(model, train_loader, optimizer, criterion)
    test(model, test_loader, criterion, device)
