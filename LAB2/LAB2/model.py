"""Torchvision backbones, CIFAR-10 heads and the two training modes."""
import torch.nn as nn
from torchvision import models

WEIGHTS = {"resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
           "vgg16": models.VGG16_Weights.IMAGENET1K_V1,
           "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1}


class CNN(nn.Module):
    def __init__(self, architecture="resnet18", num_classes=10,
                 pretrained=True, fine_tune=False):
        super().__init__()
        if architecture not in WEIGHTS:
            raise ValueError(f"Unknown architecture: {architecture}")
        if type(num_classes) is not int or num_classes <= 0:
            raise ValueError("num_classes must be a positive integer")
        if type(pretrained) is not bool or type(fine_tune) is not bool:
            raise ValueError("pretrained and fine_tune must be bool")
        self.architecture = architecture
        self.network = getattr(models, architecture)(
            weights=WEIGHTS[architecture] if pretrained else None)
        self.network.requires_grad_(False)
        if architecture == "resnet18":
            self.network.fc = nn.Linear(self.network.fc.in_features, num_classes)
            stages = [self.network.layer4]
        elif architecture == "vgg16":
            self.network.classifier[6] = nn.Linear(self.network.classifier[6].in_features, num_classes)
            stages = [self.network.features[24:]]
        else:
            self.network.classifier = nn.Linear(self.network.classifier.in_features, num_classes)
            stages = [self.network.features.denseblock4, self.network.features.norm5]
        if fine_tune:
            for stage in stages:
                stage.requires_grad_(True)
        self.train()

    def forward(self, x):
        return self.network(x)

    def train(self, mode=True):
        super().train(mode)
        for module in self.network.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                module.eval()
            if self.architecture == "vgg16" and isinstance(module, nn.Dropout):
                module.eval()
        return self

    def parameter_groups(self, lr_head, lr_backbone=None):
        prefix = {"resnet18": "network.fc.", "vgg16": "network.classifier.6.",
                  "densenet121": "network.classifier."}[self.architecture]
        head, backbone = [], []
        for name, p in self.named_parameters():
            if p.requires_grad:
                (head if name.startswith(prefix) else backbone).append(p)
        groups = [dict(name="head", params=head, lr=lr_head)]
        if backbone:
            if lr_backbone is None:
                raise ValueError("Fine-tuning needs lr_backbone")
            groups.append(dict(name="backbone", params=backbone, lr=lr_backbone))
        return groups


def parameter_statistics(model):
    return dict(n_params=sum(p.numel() for p in model.parameters()),
                trainable_params=sum(p.numel() for p in model.parameters() if p.requires_grad))


def architecture_text(model):
    rows = [str(model), str(parameter_statistics(model)), "Parameter | Shape | Count | Trainable"]
    rows += [f"{n} | {tuple(p.shape)} | {p.numel()} | {p.requires_grad}"
             for n, p in model.named_parameters()]
    return "\n".join(rows) + "\n"
