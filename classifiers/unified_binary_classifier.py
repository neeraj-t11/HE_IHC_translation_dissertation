"""
General-purpose training and testing script for binary classification using various pre-trained models.

This script allows you to train a binary classifier using different pre-trained models available in PyTorch.
You can specify the model, number of epochs, batch size, learning rate, and dataset paths as command-line arguments.
After training, the model is saved, and the testing process is carried out to evaluate its performance.

It first initializes the dataset and model based on the given options. It then runs the training loop and 
saves the model to a specified directory. Finally, it evaluates the model on the test dataset, displaying 
the loss, accuracy, confusion matrix, and class-specific accuracy.

Example:
    Train and test a ResNet101 model with default dataset paths:
        python unified_classifier.py --model_name resnet101 --epochs 10 --batch_size 32 --learning_rate 0.001

    Train and test an EfficientNet-B7 model with custom dataset paths:
        python unified_classifier.py --model_name efficientnet_b7 --epochs 20 --batch_size 64 --learning_rate 0.0001 --train_data_path /path/to/train --test_data_path /path/to/test

    The script will automatically save the trained model in the './saved_models/' directory with the model name
    as the filename (e.g., resnet101.pth, efficientnet_b7.pth). Warnings during execution are suppressed to provide
    a cleaner output.

Use '--model_name <model_name>' to specify the model architecture you want to use from the following options:
    - resnet101
    - densenet121
    - efficientnet_b7
    - inception_v3
    - regnet_y_400mf
    - resnext50_32x4d
    - vit_b_16

You can also specify the paths for the training and test datasets using '--train_data_path' and '--test_data_path'.
The default paths are:
    - Training Data: './datasets/BCI/B/train'
    - Test Data: './datasets/BCI/B/test'

For more options, modify the script or consult the PyTorch documentation for available models.

Note:
    This script assumes you have prepared the dataset and paths beforehand. Replace the dummy data in the script 
    with your actual dataset paths or use the command-line arguments to specify them.
"""
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
import argparse
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

# Custom dataset class, image loading, contrast calculation etc.
class IHCDataset(Dataset):
    def __init__(self, directory, transform=None):
        # self.directory = filter_images(directory)
        self.directory = directory
        self.transform = transform
        self.images = [os.path.join(directory, img) for img in os.listdir(directory)]
        self.labels = [self.extract_label(img) for img in self.images]

    def extract_label(self, image_path):
        # Extracts labels from filenames and categorizes into binary classes
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

# Testing function
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

# Main function to select model, train and test
def main(model_name='resnet101', epochs=10, batch_size=32, learning_rate=0.001, train_data_path='./HE_IHC_translation_dissertation/datasets/BCI/B/train', test_data_path='./HE_IHC_translation_dissertation/datasets/BCI/B/test'):
    # Choose model based on input
    print(model_name)
    if model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        num_ftrs = model.classifier.in_features
        model.classifier = nn.Linear(num_ftrs, 2)
    elif model_name == 'efficientnet_b7':
        model = models.efficientnet_b7(pretrained=True)
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, 2)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'regnet_y_400mf':
        model = models.regnet_y_400mf(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'resnext50_32x4d':
        model = models.resnext50_32x4d(pretrained=True)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(pretrained=True)
        num_ftrs = model.heads.head.in_features
        model.heads.head = nn.Linear(num_ftrs, 2)
    else:
        raise ValueError("Unsupported model name provided.")

    # Training settings
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Transformations for the dataset
    transform  = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Dataset and DataLoader
    train_dataset = IHCDataset(train_data_path, transform=transform)
    test_dataset = IHCDataset(test_data_path, transform=transform)

    if len(train_dataset) == 0:
        raise ValueError(f"No images found in the training dataset path: {train_data_path}")
    if len(test_dataset) == 0:
        raise ValueError(f"No images found in the test dataset path: {test_data_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Training loop
    print("training started for ",model_name)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

    # Save the trained model
    torch.save(model, f"./saved_models/{model_name}.pth")
    print(f"Training finished. Model saved as './saved_models/{model_name}.pth'")

    # Testing the model
    test(model, test_loader, criterion, device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate a binary classifier using various models.")
    parser.add_argument("--model_name", type=str, default="resnet101", 
                        help="Name of the model to use (e.g., resnet101, densenet121, efficientnet_b7, etc.)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for optimizer.")
    parser.add_argument("--train_data_path", type=str, 
                        default="./datasets/BCI/B/train", 
                        help="Path to the training dataset.")
    parser.add_argument("--test_data_path", type=str, 
                        default="./datasets/BCI/B/test", 
                        help="Path to the test dataset.")
    
    args = parser.parse_args()
    
    main(model_name=args.model_name, epochs=args.epochs, batch_size=args.batch_size, 
         learning_rate=args.learning_rate, train_data_path=args.train_data_path, test_data_path=args.test_data_path)
