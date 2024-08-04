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
        self.densenet = models.densenet121(pretrained=True)  # Load the pretrained DenseNet
        # self.densenet = models.DenseNet121_Weights.DEFAULT  # Load the pretrained DenseNet
        num_ftrs = self.densenet.classifier.in_features
        self.densenet.classifier = nn.Linear(num_ftrs, num_classes)  # Modify the classifier

    def forward(self, x):
        return self.densenet(x)

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
    transforms.Resize((224, 224)),
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
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        
    torch.save(model, "./saved_models/densenet121_1")
    print(f"Training finished. Loss: {total_loss / len(dataloader)}")

def test(model, dataloader, criterion, device, model_name, excel_path='model_performance.xlsx'):
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
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    total_accuracy = accuracy_score(all_labels, all_preds) * 100
    balanced_accuracy = balanced_accuracy_score(all_labels, all_preds) * 100
    
    # Calculate accuracy for each class
    class_accuracy = cm.diagonal() / cm.sum(axis=1) * 100
    
    # Prepare data for Excel
    data = {
        'Model': [model_name],
        'Confusion Matrix': [cm.tolist()],
        'Class Accuracy': [class_accuracy.tolist()],
        'Total Accuracy': [total_accuracy],
        'Balanced Accuracy': [balanced_accuracy]
    }
    df = pd.DataFrame(data)
    
    # Write or update Excel file
    try:
        existing_df = pd.read_excel(excel_path)
        df = pd.concat([existing_df, df], ignore_index=True)
    except FileNotFoundError:
        pass
    
    df.to_excel(excel_path, index=False)
    
    print(f"Testing finished. Loss: {total_loss / len(dataloader):.4f}, Total Accuracy: {total_accuracy:.2f}%, Balanced Accuracy: {balanced_accuracy:.2f}%")
    print("Confusion Matrix:")
    print(cm)
    print("Class-specific Accuracy:")
    for idx, acc in enumerate(class_accuracy):
        print(f"Class {idx} Accuracy: {acc:.2f}%")


if __name__ == "__main__":
    train(model, train_loader, optimizer, criterion, device)
    test(model, test_loader, criterion, device, model_name='DenseNet121')

