"""
src/models/ensemble.py
========================
Ensemble strategies for MediPlant AI.

CRITICAL RULE — applies to every function in this file:
    Always pass raw logits. NEVER pass pre-softmax probabilities.
    All functions apply F.softmax() internally.
    Double-softmax flattens distributions → wrong predictions.

STRATEGY COMPARISON (Final Local Evaluation - April 2026):
    ┌───────────────────────────┬───────────────┬─────────────────────────┐
    │ Strategy                  │ Test Accuracy │ Notes                   │
    ├───────────────────────────┼───────────────┼─────────────────────────┤
    │ ConvNeXt (individual)     │ 96.91%        │ Original single-run     │
    │ Swin (individual)         │ 98.07%        │ Original single-run     │
    │ Simple Average            │ 98.16% ← BEST │ Production Default      │
    │ Learned Stacking (OOF)    │ 97.87%        │ Leak-free meta-classifier│
    │ Learned Stacking (Soup)   │ 94.40%        │ Distribution Mismatch   │
    └───────────────────────────┴───────────────┴─────────────────────────┘

PRODUCTION DEFAULT: simple_avg (98.16%)
RESEARCH FINDING: Applying meta-classification to K-Fold "Model Soups" caused
a severe distribution mismatch (94.40%). Training the meta-classifier strictly 
on Out-of-Fold (OOF) logits resolved the data leakage, rebounding to 97.87%.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LearnedStackingEnsemble(nn.Module):
    """
    MLP meta-classifier trained on k-fold out-of-fold logits.
    Input:  logits_swin [batch, num_classes] — raw logits
            logits_conv [batch, num_classes] — raw logits
    Output: class logits [batch, num_classes]
    """

    def __init__(self, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        self.meta = nn.Sequential(
            nn.Linear(2 * num_classes, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, logits_swin: torch.Tensor,
                logits_conv: torch.Tensor) -> torch.Tensor:
        prob_swin = F.softmax(logits_swin, dim=1)
        prob_conv = F.softmax(logits_conv, dim=1)
        return self.meta(torch.cat([prob_swin, prob_conv], dim=1))


def simple_average_ensemble(logits_swin: torch.Tensor,
                              logits_conv: torch.Tensor) -> torch.Tensor:
    """
    Equal weight average of both models' softmax probabilities.
    Test accuracy: 98.16% — best strategy per local evaluation.
    """
    return (F.softmax(logits_swin, dim=1) + F.softmax(logits_conv, dim=1)) / 2


def weighted_average_ensemble(logits_swin: torch.Tensor,
                                logits_conv: torch.Tensor,
                                w_swin: float = 0.502,
                                w_convnext: float = 0.497) -> torch.Tensor:
    """
    Weighted average of softmax probabilities.
    Test accuracy: 98.16% — equal to simple average.
    """
    return (w_swin    * F.softmax(logits_swin, dim=1) +
            w_convnext * F.softmax(logits_conv, dim=1))