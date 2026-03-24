import timm
import torch.nn as nn


def get_convnext_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    """
    Loads ConvNeXt-Tiny pretrained on ImageNet.
    Lets timm handle the classification head directly.
    """
    model = timm.create_model(
        "convnext_tiny",
        pretrained=pretrained,
        num_classes=num_classes,    # timm sets the head directly
        drop_rate=dropout,
    )
    return model