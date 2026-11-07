import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, fine_tune=False):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

        if fine_tune:
            for p in self.model.layer4.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)


class DenseNet121(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, fine_tune=False):
        super().__init__()

        weights = models.DenseNet121_Weights.DEFAULT if pretrained else None
        self.model = models.densenet121(weights=weights)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier = nn.Linear(
            self.model.classifier.in_features, num_classes
        )

        if fine_tune:
            for p in self.model.features.denseblock4.parameters():
                p.requires_grad = True
            for p in self.model.features.norm5.parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)


class VGG16(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, fine_tune=False):
        super().__init__()

        weights = models.VGG16_Weights.DEFAULT if pretrained else None
        self.model = models.vgg16(weights=weights)

        for p in self.model.parameters():
            p.requires_grad = False

        self.model.classifier[6] = nn.Linear(
            self.model.classifier[6].in_features, num_classes
        )

        if fine_tune:
            for p in self.model.features[24:].parameters():
                p.requires_grad = True

    def forward(self, x):
        return self.model(x)
