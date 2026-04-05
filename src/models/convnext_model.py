"""
src/models/convnext_model.py
=============================
ConvNeXt Tiny for medicinal plant classification.

Captures local texture — leaf surface, fine veins, edge sharpness.
Complementary to Swin which captures global structure.
Val accuracy: 97.29% (single-run, 30 epochs)

NOTE: Do not replace best_convnext.pth with k-fold model soup checkpoint.
K-fold model soup scored 97.10% test accuracy — slightly worse than single-run.
"""

import timm


def get_convnext_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    """
    Load ConvNeXt-Tiny with ImageNet pretrained weights.

    Args:
        num_classes: Output classes (80)
        pretrained:  True for training, False when loading saved checkpoint
        dropout:     Classification head dropout rate

    Returns:
        timm ConvNeXt model ready for training or inference
    """
    return timm.create_model(
        "convnext_tiny",
        pretrained  = pretrained,
        num_classes = num_classes,
        drop_rate   = dropout,
    )
