import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedStackingEnsemble(nn.Module):
    """
    Meta-classifier trained on top of Swin + ConvNeXt softmax outputs.
    Input : concatenated probabilities  [batch, 2 * num_classes]
    Output: final logits                [batch, num_classes]
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.meta = nn.Sequential(
            nn.Linear(2 * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits_swin, logits_convnext):
        prob_swin     = F.softmax(logits_swin,     dim=1)
        prob_convnext = F.softmax(logits_convnext, dim=1)
        combined      = torch.cat([prob_swin, prob_convnext], dim=1)
        return self.meta(combined)


def simple_average_ensemble(logits_swin, logits_convnext):
    prob_swin     = F.softmax(logits_swin,     dim=1)
    prob_convnext = F.softmax(logits_convnext, dim=1)
    return (prob_swin + prob_convnext) / 2


def weighted_average_ensemble(logits_swin, logits_convnext, w_swin=0.5, w_convnext=0.5):
    prob_swin     = F.softmax(logits_swin,     dim=1)
    prob_convnext = F.softmax(logits_convnext, dim=1)
    return w_swin * prob_swin + w_convnext * prob_convnext
