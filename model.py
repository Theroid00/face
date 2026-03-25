import torch.nn as nn
import torchvision.models as models

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        # Load a pretrained ResNet18 model
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # ResNet usually takes 3-channel RGB images, but FER-2013 is 1-channel grayscale.
        # We can either let the transforms duplicate the channel to 3, or modify the first layer.
        # We'll modify the first layer to accept 1 channel.
        self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify the final classification layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.resnet(x)
