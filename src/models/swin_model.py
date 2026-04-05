"""
src/models/swin_model.py
=========================
Swin Transformer Tiny for medicinal plant classification.

Uses timm to load ImageNet-1K pretrained weights.
Captures global leaf structure via hierarchical shifted-window attention.
Val accuracy: 98.26% (single-run, 30 epochs)

NOTE: Do not replace best_swin.pth with k-fold model soup checkpoint.
K-fold model soup scored 96.52% test accuracy — worse than single-run.
"""

import timm


def get_swin_model(num_classes: int, pretrained: bool = True, dropout: float = 0.3):
    """
    Load Swin-Tiny with ImageNet pretrained weights.

    Args:
        num_classes: Output classes (80)
        pretrained:  True for training, False when loading saved checkpoint
        dropout:     Classification head dropout rate

    Returns:
        timm Swin model ready for training or inference
    """
    return timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained  = pretrained,
        num_classes = num_classes,
        drop_rate   = dropout,
    )
