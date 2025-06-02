import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class CatDogCNN(nn.Module):
    def __init__(self):
        super(CatDogCNN, self).__init__()
        # Load pre-trained ResNet18 with updated weights argument
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        # Replace the final layer (FC) to match our binary classification (Cat/Dog)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(num_ftrs, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.resnet(x)







 