import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

# ------------------------------------------------------------
# ResidualBlock Class
# ------------------------------------------------------------
"""This is a part of the proposed model"""
# Functionality: This is the residual block proposed by the paper used to avoid the vanishing gradient problem during training
# Details: It has total four conv with one short connection and one skip connection
# Input: It takes the num of input channels, mid_channels and output_channels for creating ResidualBlock() object
# Outputs: It outputs with num of out_channels by executing forward method automatically when called its object by some input
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, mid_channels1, mid_channels2, out_channels):
        """
        Initializes the residual block with specified dimensions.
        Args:
            in_channels (int): Number of input channels.
            mid_channels1 (int): Number of channels for the first and second convolutions.
            mid_channels2 (int): Number of channels for the third convolution.
            out_channels (int): Number of output channels.
        """
        super(ResidualBlock, self).__init__()

        # # First 1x1 convolution for channel reduction
        self.conv1 = nn.Conv2d(in_channels, mid_channels1, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels1)

        # Second convolution (3x3)
        self.conv2 = nn.Conv2d(mid_channels1, mid_channels1, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(mid_channels1)

        # Third convolution (3x3)
        self.conv3 = nn.Conv2d(mid_channels1, mid_channels2, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(mid_channels2)

        # Fourth 1x1 convolution for channel expansion
        self.conv4 = nn.Conv2d(mid_channels2, out_channels, kernel_size=1, stride=1, padding=0)
        self.bn4 = nn.BatchNorm2d(out_channels)

        # Shortcut connection to match input and output dimensions
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.shortcut_bn = nn.BatchNorm2d(out_channels)

    # This is the forward mehtod
    # when an object of ResidualBlock is called with some input value, x, execution starts from forward() method
    # No need to call this method manually
    def forward(self, x):
        """
        Defines the forward pass of the residual block.
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, in_channels, height, width].
        Returns:
            torch.Tensor: Output tensor after applying residual connections.
        """
        # Shortcut connection (input passed through a 1x1 convolution)
        shortcut = self.shortcut_bn(self.shortcut(x))

        # First convolution followed by BatchNorm and ReLU
        x1 = F.relu(self.bn1(self.conv1(x)))

        # Second convolution followed by BatchNorm and ReLU
        x2 = F.relu(self.bn2(self.conv2(x1)))

        # Short connection: Add output of first conv to the input of third conv
        x3_input = x1 + x2

        # Third convolution followed by BatchNorm and ReLU
        x3 = F.relu(self.bn3(self.conv3(x3_input)))

        # Fourth convolution followed by BatchNorm
        x4 = self.bn4(self.conv4(x3))

        # Skip connection: Add shortcut to the output
        x_out = F.relu(x4 + shortcut)
        return x_out

# ------------------------------------------------------------
# PaperNetwork Class
# ------------------------------------------------------------
"""This model is the paper model proposed by the authors"""
# Functionality: This model is the proposed paper model which is used for classification of the input images
# Details: It has the architecture designed by the authors where have convolution, batch normalization and above written residual block
# Input: It takes the number of classes for calssfication as input for creating PaperNetwork() object
# Output: it outputs the classification score of each class for each input image by executing forward method automatically when called its object by input images 
# e.g i/p image of shape(1, 3, 224, 224) where 1 in batch size, 3 is input channels, 224, 224 image resolution
# it will output of shape (1, num_classes)
class PaperNetwork(nn.Module):
    def __init__(self, num_classes):
        """
        Initializes the PaperNetwork with the given number of classes.
        Args:
            num_classes (int): Number of output classes for classification.
        """
        super(PaperNetwork, self).__init__()

        # Conv1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=5, stride=2, padding=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv2
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Res1
        self.res1 = ResidualBlock(in_channels=64, mid_channels1=64, mid_channels2=128, out_channels=256)

        # Conv3
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv4
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        # Res2
        self.res2 = ResidualBlock(in_channels=128, mid_channels1=64, mid_channels2=128, out_channels=256)

        # Conv5
        self.conv5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Conv6
        self.conv6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.bn6 = nn.BatchNorm2d(512)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(100352, 1024)
        self.fc2 = nn.Linear(1024, 512)

        self.output = nn.Linear(512, num_classes)

    # This is the forward mehtod
    # when an object of PaperNetwork is called with some input image, x, execution starts from forward() method
    # No need to call this method manually
    def forward(self, x):
        """
        Defines the forward pass of the PaperNetwork.
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, 3, height, width].
        Returns:
            torch.Tensor: Classification scores with shape [batch_size, num_classes].
        """
        # Conv1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool1(x)

        # Conv2
        x = F.relu(self.bn2(self.conv2(x)))

        # Res1
        x = self.res1(x)

        # Conv3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.maxpool2(x)

        # Conv4
        x = F.relu(self.bn4(self.conv4(x)))

        # Res2
        x = self.res2(x)

        # Conv5
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.maxpool3(x)

        # Conv6
        x = F.relu(self.bn6(self.conv6(x)))

        # Flatten
        x = x.view(x.size(0), -1)
        # Fully Connected Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return self.output(x)



# ------------------------------------------------------------
# ConLModel Class
# ------------------------------------------------------------
"""1st stage training encoder model learned with contrastive learning"""
# Functionality: This model is my proposed Contrastive learning model which learns to differentiate among the classes 
# with features in higher 128-d dimensional space with contrastive loss
# Input: It takes no input for creating ConLModel() object
# Output: it convert given input to higher dimension and outputs 128-d for each input image by executing forward method automatically when called its object by input images 
# e.g i/p image of shape(1, 3, 224, 224) where 1 in batch size, 3 is input channels, 224, 224 image resolution
# it will output of shape (1, 128)
# As the base encoder I have used VGG16_bn pretrained model and modified it to output 128-d
# 1st Stage Training happens with this model
class ConLModel(nn.Module): #contrastive learning model
    def __init__(self):
        """
        Initializes the contrastive learning model by modifying VGG16_bn.
        """
        super(ConLModel, self).__init__()
        # # Pretrained model VGG16 with batch normalization
        model = torchvision.models.vgg16_bn(False, True)

        # Drop the last few layers and convert it  to 128d dimension of feature embedding
        model.classifier[3] = nn.Linear(4096, 128)
        model.classifier[4] = nn.Identity()
        model.classifier[5] = nn.Identity()
        model.classifier[6] = nn.Identity()
        self.features = model.features
        self.classifier = model.classifier
    
    # This is the forward mehtod
    # when an object of ConLModel is called with some input image, x, execution starts from forward() method
    # No need to call this method manually
    def forward(self, x):
        """
        Defines the forward pass for ConLModel.
        Args:
            x (torch.Tensor): Input tensor with shape [batch_size, 3, height, width].
        Returns:
            torch.Tensor: 128-dimensional embeddings for each input.
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
# ------------------------------------------------------------
# Siamese_Encoder Class
# ------------------------------------------------------------
"""For simplicity and make it more readable I have made another class Siamese_Encoder"""
"""This is just a replica of the ConLModel"""
# The above define ConLModel is passed as encoder to learn the embedding space with super contrastive learning
# Functionality: Same as the above ConLModel
# input: Takes the object of ConLModel as encoder
# Output: Outputs 128-d vector
# e.g i/p image of shape(1, 3, 224, 224) where 1 in batch size, 3 is input channels, 224, 224 image resolution
# it will output of shape (1, 128)
# As the base encoder I have used VGG16_bn pretrained model and modified it to output 128-d
# 1st Stage Training happens with this model
class Siamese_Encoder(nn.Module):
    def __init__(self, encoder):
        super(Siamese_Encoder, self).__init__()
        self.encoder = encoder  # Store the ConLModel encoder instance.
    
    # This is the forward mehtod
    # when an object of Siamese_Encoder is called with some input image, x, execution starts from forward() method
    # No need to call this method manually
    def forward(self, x):
        """
        Performs a forward pass through the encoder.
        
        Parameters:
        - x (Tensor): Input image tensor, e.g., shape (batch_size, 3, height, width).

        Returns:
        - Tensor: 128-dimensional embeddings for the input images.
        """
        return self.encoder(x)


# ------------------------------------------------------------
# FinalModel Class
# ------------------------------------------------------------
"""2nd Stage: Downstream task (Classification) model used for final classification task"""
# Functionality: This model is used for final classification of input images
# details: I have added one classification layer after the encoder of ConLModel. I freezes the already trained encoder from 1st stage training,
# just trains the newly addes classifier layer
# Input: It takes trained enocder from 1st stage, and num of classes for creating FinalModel() object
# Output: When images are passed to the object of this class, it executes forward method automatically and 
# outputs classification score of each class for each input image
# e.g i/p image of shape(1, 3, 224, 224) where 1 in batch size, 3 is input channels, 224, 224 image resolution
# it will output of shape (1, num_classes)
class FinalModel(nn.Module):
    def __init__(self, encoder, num_class):
        super(FinalModel, self).__init__()
        self.encoder = encoder # trained encoder instance from 1 stage training.
        self.output = nn.Linear(128, num_class) # Added Classification layer with the appropriate output dimensions.
    
    # This is the forward mehtod
    # when an object of FinalModel is called with some input image, x, execution starts from forward() method
    # No need to call this method manually
    def forward(self, x):
        """
        Performs a forward pass through the encoder and classification layer.
        
        Parameters:
        - x (Tensor): Input image tensor, e.g., shape (batch_size, 3, height, width).

        Returns:
        - Tensor: Class scores for each input image.
        """
        x = self.encoder(x)
        # x = self.classifier(x)
        x = self.output(x)
        return x

