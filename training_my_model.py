# Import necessary libraries
import pandas as pd
import numpy as np
from PIL import Image
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
warnings.filterwarnings("ignore", category=FutureWarning)
import sys
from models import PaperNetwork, ConLModel, Siamese_Encoder, FinalModel
from datasets import CLCKDataset, CLJaffeDataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import wandb

# Prompt the user to choose a dataset for training the model
print("Choose one of the following two datasets for training your model:")
print("1. JAFFE dataset")
print("2. CK+ dataset")
dataset_choice = int(input("Type 1 or 2: ")) # User input for dataset selection

# Dataset-specific settings
if dataset_choice==1:
    print("chosen JAFFE dataset")
elif dataset_choice==2:
    print("chosen CK+ dataset")
else:
    print("Re-run and type 1 or 2")
    sys.exit()

# Initialize WandB for experiment tracking
wandb.login()

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
        "Model": "SuperCon",
        "Encoder": "Vgg16_bn", 
        "dataset": "Jaffe" if dataset_choice==1 else "CK+"
    }
)

# used to resize the input image and convert it to tensor array
transform = transforms.Compose([
            transforms.Resize(size=(224, 224), interpolation=Image.NEAREST),
            transforms.ToTensor()
])

# random augmentation sued for creating one augmented version of original image
aug = A.Compose([
    A.OneOf([ # from the following augmentation techniques, one is choosen randomly
        A.Rotate(limit=90, p=1, border_mode=cv2.BORDER_CONSTANT), # random rotate from 0-90 degree
        A.Rotate(limit=270, p=1, border_mode=cv2.BORDER_CONSTANT), # random rotate from 0-270 degree
        A.HorizontalFlip(p=1), # Horizontal flip
        A.VerticalFlip(p=1) # Vertical flip
    ], p=1)
])


# Load and preprocess dataset
if dataset_choice==1:
    # load CSV file for JAFFE dataset
    df = pd.read_csv("jaffee_label.csv")
    # Image folder
    image_dir = 'Jaffe/Jaffe_cropped/'
    # Split the dataset into training and testing dataset using dataframe
    train_df, val_df = train_test_split(df, test_size = 0.20, shuffle=True, random_state= 44, stratify = df['label_Encoded'])
    train_dataset = CLJaffeDataset(train_df, image_dir, transform=transform, aug=aug)
    test_dataset = CLJaffeDataset(val_df, image_dir, transform=transform, aug=aug)

else:
    # load CSV file for CK+ dataset
    df = pd.read_csv('cropped_images.csv')
    image_dir = 'all_ck_images/'
    # Split the dataset into training and testing dataset using dataframe
    train_df, val_df = train_test_split(df, test_size = 0.20, shuffle=True, random_state= 44, stratify = df['diagnosis'])
    train_dataset = CLCKDataset(train_df, image_dir, transform=transform, aug=aug)
    test_dataset = CLCKDataset(val_df, image_dir, transform=transform, aug=aug)

# Initialize the Supervised Contrastive model encoder
encoder = ConLModel()
model = Siamese_Encoder(encoder)
model.to(device=DEVICE)

# Define data loaders
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.SGD(model.parameters(), lr=LR, momentum=0.9)
criterion = losses.SupConLoss(temperature=0.1) # This is the contrastive loss function for 1st stage training


# Save the model with lowest supervised contrastive loss
training_loss = []
valid_loss_min = np.inf

for n in range(EPOCHS):
    train_loss = 0.0
    model.train()
    for batch in tqdm(train_dataloader, leave=False):
        x1, x2, y = batch
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        y = torch.from_numpy(np.asarray(y)).to(DEVICE)

        # Forward pass through the Siamese Encoder
        out1 = model(x1)
        out2 = model(x2)

        # Combine features and duplicate labels for contrastive loss
        features = torch.cat((out1, out2), dim=0).squeeze()
        y = torch.cat((y, y), dim=0)

        # Compute loss
        loss = criterion(features, y)
        train_loss += loss.item()

        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    train_loss /= len(train_dataloader.dataset)
    training_loss.append([n,train_loss])
    print("epoch number: ", n)
    print(f"Train Loss: {(100*train_loss):.2f}%")
    wandb.log({"SC loss per epoch":train_loss})
    
    # Save model with lowest validation loss
    if valid_loss_min>train_loss:
        torch.save(model, 'saved_model/JF_embedding_128.pth') if dataset_choice==1 else torch.save(model, 'saved_model/CK_embedding_128.pth')
        print("new added", train_loss)
        valid_loss_min = train_loss
    

# Load the saved encoder from the contrastive learning
if dataset_choice==1:
    encoder2 = torch.load('saved_model/JF_embedding_128.pth')
else:
    encoder2 = torch.load('saved_model/CK_embedding_128.pth')
print("Model (lowest SC loss) loaded for classification.................")


# Freeze the conder weights, only classifier layer will learn (The 2nd training stage)
model = FinalModel(encoder2, NUM_CLASSES).to(DEVICE)
for param in model.encoder.parameters():
    param.requires_grad = False


# hyperparameters for 2nd stage training (classification)
learning_rate = 0.001
epochs = 30
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), learning_rate, momentum=0.9)

Test = 0.0
training_acc = []
Test_Result = []
for n in range(epochs):
    # Training phase
    current_acc = 0
    train_loss = 0.0
    model.train()
    for batch in tqdm(train_dataloader, leave=False):
        x1, x2, y = batch
        x1 = x1.to(DEVICE)
        del x2 # drop the augmented image as in 2nd stage we need to learn classification with original tensor image

        y = torch.from_numpy(np.asarray(y)).to(DEVICE)
        pred = model(x1) # model prediction of class score
        loss = loss_fn(pred, y) # cross-entropy loss calculation

        # backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        current_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    current_acc = current_acc/ len(train_dataloader.dataset)
    train_loss /= len(train_dataloader.dataset)
    training_acc.append({'Accuracy': current_acc, 'Avg loss': train_loss})
    
    # Evaluation phase
    model.eval()
    size = len(test_dataloader.dataset)
    test_loss, correct = 0, 0
    # Store true labels and predictions
    true_labels = []
    predictions = []
    with torch.no_grad():
        for batch in tqdm(test_dataloader, leave=False):
            x1, x2, y = batch
            x1 = x1.to(DEVICE)
            del x2
            y = torch.from_numpy(np.asarray(y)).to(DEVICE)
            pred = model(x1)
            true_labels.extend(y.cpu().numpy())
            predictions.extend(pred.cpu().numpy())
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print("epoch number: ", n)
    print(f"Train Accuracy: {(100*current_acc):.2f}%")
    print(f"Test Result: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    Test_Result.append({'Accuracy': correct, 'Avg loss': test_loss})

    # Logging with wandb
    wandb.log({"Training Accuracy": current_acc, "Training Loss": train_loss, "Validation Accuracy": correct, "val loss": test_loss})


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
        for inputs, _, labels in dataloader:

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


from sklearn.metrics import confusion_matrix
# Compute the confusion matrix
cm = confusion_matrix(test_labels, test_preds)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(test_labels), yticklabels=np.unique(test_labels))
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# Visualize the embedding space
if dataset_choice==1:
    encoder2 = torch.load('saved_model/JF_embedding_128.pth')
else:
    encoder2 = torch.load('saved_model/CK_embedding_128.pth')
print("encoder loaded for Visualization..........")

features = []
labels = []
encoder2.eval()
with torch.no_grad():
    for batch in tqdm(train_dataloader, leave=False):
        x1, x2, y = batch
        x1 = x1.to(DEVICE)
        del x2
        y = torch.from_numpy(np.asarray(y)).to(DEVICE)
        pred = encoder2(x1)
        labels += list(y.cpu())
        features.append(pred.cpu().numpy())

X = np.concatenate(features, axis=0)
y = np.array(labels)

embedded_space = TSNE(n_components=2).fit_transform(X)

# Plot the results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedded_space[:, 0], embedded_space[:, 1], c=labels, cmap='viridis', s=10)
plt.colorbar(scatter, label='Class Labels')
plt.title("t-SNE Visualization of Test Data Features")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()
