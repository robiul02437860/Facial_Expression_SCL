# Import necessary libraries for handling datasets, transformations, and augmentations
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.datasets as datasets
from sklearn.model_selection import train_test_split
import torchvision.transforms as transforms
import torch.nn.functional as F

# Functionality: Its used for creating Pytorch dataset of JAFFE dataset for the use of Contrastive learning
# Input: DataFrame from the Jaffe CSV file, single Jaffee image directory, pytorch tranform to resize and tensor convertion, Albumentaion for random augmentation
# Output: Tensor image, tensor augmented image and its label
class CLJaffeDataset(Dataset):
    def __init__(self, df, image_dir, transform= None, aug = None):
        """
        Initialize the CLJaffeDataset.

        Parameters:
        - df (pd.DataFrame): DataFrame containing metadata (columns: 'PIC', 'label_Encoded').
        - image_dir (str): Directory where the all images are stored in a single folder.
        - transform (callable, optional): Transformations to apply to the images to resize and convert numpy array to tensor array
        - aug (callable, optional): Augmentations to apply to the images (e.g., flipping, rotations) to make another copy of the same image used for contrastive learning
        """        
        super(CLJaffeDataset, self).__init__()
        self.image_ids = list(df['PIC']) # List of image file names
        self.labels = list(df['label_Encoded']) # List of encoded labels
        self.image_dir = image_dir
        self.transform = transform
        self.aug = aug

    def __getitem__(self, idx):
        """
        Fetch a single item (image, its augmented image with random augmentation and label) from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (Tensor): Transformed original image.
        - image2 (Tensor): Augmented and transformed image (if augmentations are applied).
        - label (Tensor): Encoded label.
        """
        file_name = self.image_ids[idx]+".tiff" # Construct file name with .tiff extension as in image folder
        label = self.labels[idx] # Retrieve label
        image = Image.open(self.image_dir+file_name).convert('RGB') # read and convert image to RGB format
        aug_image = np.array(image) # Convert image to NumPy array and apply augmentations if provided
        if self.aug:
            aug_image = self.aug(image=aug_image) # Apply augmentations
            image2 = transforms.ToPILImage()(aug_image['image']) # Convert augmented image back to PIL format
        if self.transform:
            image2 = self.transform(image2)
            image = self.transform(image)
        label = torch.tensor(label) # Convert label to PyTorch tensor
        return image, image2, label # Return original image, its augmented image and label

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.image_ids)
    

# Functionality: Its used for creating Pytorch dataset of CK+ dataset for the use of Contrastive learning
# Input: DataFrame from the CK+ CSV file, single CK+ image directory, pytorch tranform to resize and tensor convertion, Albumentaion for random augmentation
# Output: Tensor image, tensor augmented image and its label
class CLCKDataset(Dataset):
    def __init__(self, df, image_dir, transform= None, aug = None):
        """
        Initialize the CLCKDataset.

        Parameters:
        - df (pd.DataFrame): DataFrame containing metadata (columns: 'id_code', 'diagnosis').
        - image_dir (str): Directory where the images are stored in a single folder.
        - transform (callable, optional): Transformations to apply to the images to resize and convert numpy array to tensor array
        - aug (callable, optional): Augmentations to apply to the images (e.g., flipping, rotations) to make another copy of the same image used for contrastive learning
        """
        super(CLCKDataset, self).__init__()
        self.image_ids = list(df['id_code']) # List of image file names
        self.labels = list(df['diagnosis']) # List of labels
        self.image_dir = image_dir
        self.transform = transform
        self.aug = aug

    def __getitem__(self, idx):
        """
        Fetch a single item (image and label) from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (Tensor): Transformed original image.
        - image2 (Tensor): Augmented and transformed image (if augmentations are applied).
        - label (Tensor): Label.
        """

        file_name = self.image_ids[idx]
        label = self.labels[idx]
        image = Image.open(self.image_dir+file_name).convert('RGB') # read and convert image to RGB format
        # Convert image to NumPy array and apply augmentations if provided
        aug_image = np.array(image) 
        if self.aug:
            aug_image = self.aug(image=aug_image) # Apply augmentations
            image2 = transforms.ToPILImage()(aug_image['image']) # Convert augmented image back to PIL format
        # Apply transformations if provided
        if self.transform:
            image2 = self.transform(image2)
            image = self.transform(image)
        label = torch.tensor(label) # Convert label to PyTorch tensor
        return image, image2, label # Return original image, its augmented image and label

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.image_ids)


# Functionality: Its used for creating Pytorch dataset of JAFFE dataset for the use of classification by the paper model
# Input: DataFrame from the Jaffe CSV file, single Jaffee image directory, pytorch tranform to resize and tensor convertion, Albumentaion for random augmentation
# Output: Tensor image and its label
class CustomDatasetJF(Dataset):
    def __init__(self, df, image_dir, transform= None):
        """
        Initialize the CustomDatasetJF.

        Parameters:
        - df (pd.DataFrame): DataFrame containing metadata (columns: 'PIC', 'label_Encoded').
        - image_dir (str): Directory where the images are stored in a single folder.
        - transform (callable, optional): Transformations to apply to the images to resize and convert numpy array to tensor array
        """
        super(CustomDatasetJF, self).__init__()
        self.image_ids = list(df['PIC']) # List of image file names
        self.labels = list(df['label_Encoded']) # List of encoded labels
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        """
        Fetch a single item (image and label) from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (Tensor): Transformed image.
        - label (Tensor): Encoded label.
        """
        file_name = self.image_ids[idx]+".tiff" # Construct file name with .tiff extension as in stored folder
        label = self.labels[idx]
        image = Image.open(self.image_dir+file_name).convert('RGB') # read and convert image to RGB format
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label) # Convert label to PyTorch tensor
        return image, label

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.image_ids)


# Functionality: Its used for creating Pytorch dataset of CK+ dataset for the use of classification by the paper model
# Input: DataFrame from the CK+ CSV file, single CK+ image directory, pttorch tranform to resize and tensor convertion, Albumentaion for random augmentation
# Output: Tensor image and its label
class CustomDatasetCK(Dataset):
    def __init__(self, df, image_dir, transform= None):
        """
        Initialize the CustomDatasetCK.

        Parameters:
        - df (pd.DataFrame): DataFrame containing metadata (columns: 'id_code', 'diagnosis').
        - image_dir (str): Directory where the images are stored in a single folder.
        - transform (callable, optional): Transformations to apply to the images to resize and convert numpy array to tensor array
        """
        super(CustomDatasetCK, self).__init__()
        self.image_ids = list(df['id_code']) # List of image file names
        self.labels = list(df['diagnosis']) # List of labels
        self.image_dir = image_dir
        self.transform = transform

    def __getitem__(self, idx):
        """
        Fetch a single item (image and label) from the dataset.

        Parameters:
        - idx (int): Index of the sample to retrieve.

        Returns:
        - image (Tensor): Transformed image.
        - label (Tensor): Label.
        """
        file_name = self.image_ids[idx]
        label = self.labels[idx]
        image = Image.open(self.image_dir+file_name).convert('RGB') # read and convert image to RGB format
        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(label) # Convert label to PyTorch tensor
        return image, label

    def __len__(self):
        """
        Get the number of samples in the dataset.

        Returns:
        - int: Total number of samples.
        """
        return len(self.image_ids)
