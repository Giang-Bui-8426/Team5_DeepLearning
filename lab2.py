import torch
import torchvision.models as models

vgg16    = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)


# Các khối chính
for name, m in vgg16.named_children():
    print("VGG16   :", name, "->", type(m).__name__)
for name, m in resnet18.named_children():
    print("ResNet18:", name, "->", type(m).__name__)

# Từng layer lá 
for name, m in vgg16.named_modules():
    if len(list(m.children())) == 0:
        print("VGG16   ", name, m)
for name, m in resnet18.named_modules():
    if len(list(m.children())) == 0:
        print("ResNet18", name, m)

print(vgg16)
print(resnet18)