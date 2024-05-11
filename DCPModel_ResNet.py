import torch
import torch.nn as nn
import torchvision.models as models

class DCPModel(nn.Module):
    def __init__(self):
        super(DCPModel, self).__init__()

        # Load a pre-trained ResNet model as the base
        resnet = models.resnet18(pretrained=True)

        # Freeze all parameters of the base ResNet layers
        for param in resnet.parameters():
            param.requires_grad = False

        # Extract the last convolutional layer of the ResNet model
        self.base_layers = nn.Sequential(*list(resnet.children())[:-2])
        self.conv5 = resnet.layer4[-1]  # Get the last convolutional layer from the ResNet model

        # Define additional layers for the DCP model
        self.conv1 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=3, padding=1)  # Output 3 channels

        self.relu = nn.ReLU(inplace=True)
        
        self.upsample = nn.Upsample(size=(224, 224), mode='bilinear', align_corners=False)
        
    def forward(self, x):
        # Forward pass through the ResNet base layers
        x = self.base_layers(x)

        # Forward pass through the last convolutional layer
        x = self.conv5(x)

        # Upsample the feature maps to the desired output size
        x = self.upsample(x)

        # Apply additional convolutional layers
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)  # Output 3 channels

        return x
