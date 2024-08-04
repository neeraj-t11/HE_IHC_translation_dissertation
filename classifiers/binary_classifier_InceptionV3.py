import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image
import os

import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, balanced_accuracy_score

def load_image(file_path):
    """ Load an image from disk. """
    return cv2.imread(file_path)

def calculate_contrast(image):
    """ Calculate the contrast of an image based on its standard deviation. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray.std()

def is_blurry(image, threshold=100):
    """ Check if an image is blurry using the Laplacian method. """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def has_artifacts(image, threshold=1.0):
    """ Detect significant artifacts in an image using edge detection. """
    edges = cv2.Canny(image, 100, 200)
    edge_area = np.mean(edges)
    return edge_area > threshold

def filter_image(file_path):
    """ Apply filters to an image and return None if it fails any quality checks. """
    image = load_image(file_path)
    if image is None or calculate_contrast(image) < 20 or is_blurry(image) or has_artifacts(image):
        return None
    return image

def filter_images(directory):
    """ Filter images in the directory and return list of paths that pass the filters. """
    filtered_paths = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if filter_image(file_path) is not None:
            filtered_paths.append(file_path)
    return filtered_paths

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class IHCClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super(IHCClassifier, self).__init__()
        # Assuming `inception_v4` is available as a function to load the model
        self.inception = models.inception_v3(pretrained=True)  # or your custom function to load Inception-v4
        num_ftrs = self.inception.fc.in_features
        self.inception.fc = nn.Linear(num_ftrs, num_classes)  # Replace the classifier
        # Handle the auxiliary net
        num_aux_ftrs = self.inception.AuxLogits.fc.in_features
        self.inception.AuxLogits.fc = nn.Linear(num_aux_ftrs, num_classes)

    def forward(self, x):
        # Inception V4 requires the model to return both outputs during training
        if self.training:
            outputs, aux_outputs = self.inception(x)
            return outputs, aux_outputs
        else:
            return self.inception(x)

class IHCDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory)]
        self.labels = [self.extract_label(img) for img in self.images]

    def extract_label(self, image_path):
        grade = image_path.split('_')[-1].replace('.png', '')
        return 1 if grade == '3+' else 0
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]
        image = Image.open(image_path).convert('RGB')
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

transformations = transforms.Compose([
    transforms.Resize((299, 299)),  # Adjust according to Inception input size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = IHCDataset('./datasets/BCI/B/train', transform=transformations)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = IHCDataset('./datasets/BCI/B/test', transform=transformations)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = IHCClassifier().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    print("Training started...")
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, aux_outputs = model(images)  # Adjust to handle two outputs
        loss1 = criterion(outputs, labels)
        loss2 = criterion(aux_outputs, labels)  # Auxiliary classifier loss
        loss = loss1 + 0.4 * loss2  # Combine losses
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    torch.save(model, "./saved_models/inception_v3")
    print(f"Training finished. Loss: {total_loss / len(dataloader)}")

def test(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    total_accuracy = accuracy_score(all_labels, all_preds) * 100
    conf_matrix = confusion_matrix(all_labels, all_preds)
    class_report = classification_report(all_labels, all_preds, output_dict=True)
    class_accuracy = {str(k): v['precision'] for k, v in class_report.items() if k.isdigit()}
    print(f"Testing finished. Loss: {total_loss / len(dataloader):.4f}, Total Accuracy: {total_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(conf_matrix)
    print("Class-specific Accuracy:")
    for cls, acc in class_accuracy.items():
        print(f"Class {cls} Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, device)
    test(model, test_loader, criterion, device)
