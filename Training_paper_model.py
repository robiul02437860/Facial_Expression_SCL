# import necessary libraries
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
from pytorch_metric_learning import losses
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import sys
from models import PaperNetwork
from datasets import CustomDatasetJF, CustomDatasetCK
from sklearn.metrics import classification_report, confusion_matrix
import wandb

# Prompt user to choose a dataset
print("Choose one of the following two datasets for training your model:")
print("1. JAFFE dataset")
print("2. CK+ dataset")
dataset_choice = int(input("Type 1 or 2: ")) # User input for dataset selection

# Initialize WandB for experiment tracking
wandb.login()

# Dataset-specific settings
if dataset_choice==1:
    print("chosen JAFFE dataset")
elif dataset_choice==2:
    print("chosen CK+ dataset")
else:
    print("Re-run and type 1 or 2")
    sys.exit()


# Hyperparameters settings
BATCH_SIZE = 8 if dataset_choice==1 else 8 # Batch size for training and validation, need to adjust for user choice
IMG_SIZE = 224 # Image resolution
NUM_CLASSES = 6 if dataset_choice==1 else 8 # Number of classes in the dataset
EPOCHS = 100 # Number of training epochs
LR= 0.001 # Learning rate
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available

# WandB configuration
wandb.init(
    project="CS6680_CV",
    config={
        "learning_rate": LR,
        "epoch": EPOCHS,
        "batch_size": BATCH_SIZE,
        "Model": "Paper Proposed",
        "dataset": "Jaffe" if dataset_choice==1 else "CK+"
    }
)

# used to resize the input image and convert it to tensor array
transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
])

# Load and preprocess dataset
if dataset_choice==1:
    # load CSV file for JAFFE dataset
    df = pd.read_csv("jaffee_label.csv")
    # Image folder
    image_dir = 'Jaffe/Jaffe_cropped/'
    # Split the dataset into training and testing dataset using dataframe
    train_df, val_df = train_test_split(df, test_size = 0.20, shuffle=True, random_state= 44, stratify = df['label_Encoded'])
    train_dataset = CustomDatasetJF(train_df, image_dir, transform=transform)
    test_dataset = CustomDatasetJF(val_df, image_dir, transform=transform)

else:
    # load CSV file for CK+ dataset
    df = pd.read_csv('cropped_images.csv')
    image_dir = 'all_ck_images/'
    # Split the dataset into training and testing dataset using dataframe
    train_df, val_df = train_test_split(df, test_size = 0.20, shuffle=True, random_state= 44, stratify = df['diagnosis'])
    train_dataset = CustomDatasetCK(train_df, image_dir, transform=transform)
    test_dataset = CustomDatasetCK(val_df, image_dir, transform=transform)


# Initialize model proposed by the paper
model = PaperNetwork(num_classes=NUM_CLASSES)
model.to(device=DEVICE)

# Define data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Define optimizer and loss function
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.CrossEntropyLoss() # Cross-entropy loss for classification

# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    """
    Trains the model and evaluates its performance on the validation set.

    Parameters:
    - model: The PyTorch model to train.
    - train_loader: DataLoader for the training set.
    - val_loader: DataLoader for the validation set.
    - criterion: Loss function.
    - optimizer: Optimizer for training.
    - num_epochs: Number of training epochs.

    Outputs:
    - Logs training/validation accuracy and loss to WandB.
    """
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        
        train_loss = running_loss / len(train_loader)
        train_acc = correct / total

        # Validation loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        
        val_loss /= len(val_loader)
        val_acc = correct / total

        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        wandb.log({"Trainig ccuracy": train_acc, "Validation Accuracy": val_acc, "Training Loss": train_loss, "Validation Loss": val_loss})

# Train the Model
train_model(model, train_dataloader, test_dataloader, criterion, optimizer, EPOCHS)

# Function to get predictions and true labels
def get_predictions_and_labels(dataloader, model, device):
    """
    Gets predictions and true labels for a dataset.

    Parameters:
    - dataloader: DataLoader for the test dataset.
    - model: Trained PyTorch model.
    - device: Device (CPU/GPU).

    Outputs:
    - all_preds: Predicted class labels.
    - all_labels: True class labels.
    """
    model.eval()  # Set model to evaluation mode
    all_preds = []
    all_labels = []
    
    with torch.no_grad():  # Disable gradient calculation for inference
        for inputs, labels in dataloader:

            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Forward pass
            preds = torch.argmax(outputs, dim=1)  # Get class with highest score
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return all_preds, all_labels


# Get predictions and labels for training and test sets
test_preds, test_labels = get_predictions_and_labels(test_dataloader, model, DEVICE)

# Print classification report
print("\nClassification Report for Test Data:")
print(classification_report(test_labels, test_preds))

# Compute the confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()