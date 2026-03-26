"""
Ensemble Training & Evaluation Script
Meta-classifier trained on train set probabilities.
Increased to 200 epochs for better convergence.

Usage:
    python scripts/train_ensemble.py
"""

import os
import sys
import json
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MediPlantDataset
from src.data.transforms import get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.models.ensemble import (
    LearnedStackingEnsemble,
    simple_average_ensemble,
    weighted_average_ensemble,
)


# ── Extract probability vectors ──────────────────────────────────────────────


@torch.no_grad()
def extract_probabilities(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in tqdm(loader, desc="Extracting features"):
        images = images.to(device)
        probs = F.softmax(model(images), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)

    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


# ── Train meta-classifier on train set ───────────────────────────────────────


def train_meta_classifier(
    probs_swin,
    probs_convnext,
    labels,
    num_classes,
    hidden_dim=256,
    epochs=200,
    lr=0.001,
):
    combined = torch.cat([probs_swin, probs_convnext], dim=1)
    dataset = TensorDataset(combined, labels)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    meta_model = LearnedStackingEnsemble(num_classes=num_classes, hidden_dim=hidden_dim)
    optimizer = AdamW(meta_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        meta_model.train()
        total_loss = 0.0

        for batch_combined, batch_labels in loader:
            optimizer.zero_grad()
            b_swin = batch_combined[:, :num_classes]
            b_convnext = batch_combined[:, num_classes:]
            loss = criterion(meta_model(b_swin, b_convnext), batch_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()
        avg_loss = total_loss / len(loader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in meta_model.state_dict().items()}

        if epoch % 25 == 0:
            print(
                f"  Meta epoch [{epoch:03d}/{epochs}] Loss: {avg_loss:.4f} | Best Loss: {best_loss:.4f}"
            )

    print(f"  Final best loss: {best_loss:.4f}")
    meta_model.load_state_dict(best_state)
    return meta_model


# ── Evaluate all strategies ───────────────────────────────────────────────────


def evaluate_strategy(name, preds, labels):
    acc = (preds == labels).float().mean().item()
    print(f"  {name:<40} Accuracy: {acc * 100:.2f}%")
    return acc


@torch.no_grad()
def evaluate_all(
    probs_swin, probs_convnext, labels, meta_model, num_classes, w_swin, w_conv
):
    print("\n── Ensemble Evaluation on Validation Set ──────────────────────")

    acc_swin = evaluate_strategy("Swin (individual)", probs_swin.argmax(1), labels)
    acc_conv = evaluate_strategy(
        "ConvNeXt (individual)", probs_convnext.argmax(1), labels
    )
    print()

    acc_avg = evaluate_strategy(
        "Simple Average",
        simple_average_ensemble(probs_swin, probs_convnext).argmax(1),
        labels,
    )
    acc_wavg = evaluate_strategy(
        f"Weighted Avg ({w_swin:.3f}/{w_conv:.3f})",
        weighted_average_ensemble(probs_swin, probs_convnext, w_swin, w_conv).argmax(1),
        labels,
    )

    meta_model.eval()
    acc_stack = evaluate_strategy(
        "Learned Stacking", meta_model(probs_swin, probs_convnext).argmax(1), labels
    )

    print("────────────────────────────────────────────────────────────────")

    return {
        "swin_individual": acc_swin,
        "convnext_individual": acc_conv,
        "simple_avg": acc_avg,
        "weighted_avg": acc_wavg,
        "learned_stacking": acc_stack,
        "weights": {"swin": w_swin, "convnext": w_conv},
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main():
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    num_classes = cfg["data"]["num_classes"]
    image_size = cfg["data"]["image_size"]
    batch_size = cfg["data"]["batch_size"]

    print(f"Device  : {device}")
    print(f"Classes : {num_classes}")

    val_transform = get_val_transforms(image_size)
    train_ds = MediPlantDataset("data/splits/train.csv", val_transform)
    val_ds = MediPlantDataset("data/splits/val.csv", val_transform)
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )

    # ── Load models ──
    print("\nLoading Swin model...")
    swin_model = get_swin_model(num_classes, pretrained=False)
    swin_model.load_state_dict(
        torch.load("outputs/checkpoints/best_swin.pth", map_location=device)
    )
    swin_model = swin_model.to(device)

    print("Loading ConvNeXt model...")
    conv_model = get_convnext_model(num_classes, pretrained=False)
    conv_model.load_state_dict(
        torch.load("outputs/checkpoints/best_convnext.pth", map_location=device)
    )
    conv_model = conv_model.to(device)

    # ── Extract train set probabilities (for meta training) ──
    print("\nExtracting Swin probabilities (train)...")
    train_probs_swin, train_labels = extract_probabilities(swin_model, train_dl, device)

    print("Extracting ConvNeXt probabilities (train)...")
    train_probs_conv, _ = extract_probabilities(conv_model, train_dl, device)

    # ── Extract val set probabilities (for evaluation) ──
    print("\nExtracting Swin probabilities (val)...")
    val_probs_swin, val_labels = extract_probabilities(swin_model, val_dl, device)

    print("Extracting ConvNeXt probabilities (val)...")
    val_probs_conv, _ = extract_probabilities(conv_model, val_dl, device)

    # ── Compute weights ──
    swin_val_acc = (val_probs_swin.argmax(1) == val_labels).float().mean().item()
    conv_val_acc = (val_probs_conv.argmax(1) == val_labels).float().mean().item()
    total = swin_val_acc + conv_val_acc
    w_swin = swin_val_acc / total
    w_conv = conv_val_acc / total

    # ── Train meta-classifier on train set with 200 epochs ──
    print("\nTraining meta-classifier (200 epochs)...")
    meta_model = train_meta_classifier(
        probs_swin=train_probs_swin,
        probs_convnext=train_probs_conv,
        labels=train_labels,
        num_classes=num_classes,
        hidden_dim=cfg["ensemble"]["meta_hidden_dim"],
        epochs=200,
        lr=0.001,
    )

    # ── Save everything ──
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs("outputs/results", exist_ok=True)

    torch.save(meta_model.state_dict(), "outputs/checkpoints/best_ensemble.pth")
    print("\nSaved -> outputs/checkpoints/best_ensemble.pth")

    with open("outputs/checkpoints/weighted_avg_config.json", "w") as f:
        json.dump({"w_swin": w_swin, "w_convnext": w_conv}, f, indent=2)
    print("Saved -> outputs/checkpoints/weighted_avg_config.json")

    # ── Evaluate all strategies on val set ──
    results = evaluate_all(
        val_probs_swin,
        val_probs_conv,
        val_labels,
        meta_model,
        num_classes,
        w_swin,
        w_conv,
    )

    with open("outputs/results/ensemble_val_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved -> outputs/results/ensemble_val_results.json")

    # ── Summary ──
    ensemble_only = {
        k: v
        for k, v in results.items()
        if k in ["simple_avg", "weighted_avg", "learned_stacking"]
    }
    best_strategy = max(ensemble_only, key=ensemble_only.get)

    print(f"\n{'='*52}")
    print(f"  Swin individual        : {results['swin_individual']     * 100:.2f}%")
    print(f"  ConvNeXt individual    : {results['convnext_individual'] * 100:.2f}%")
    print(f"  Simple Average         : {results['simple_avg']          * 100:.2f}%")
    print(f"  Weighted Average       : {results['weighted_avg']        * 100:.2f}%")
    print(f"  Learned Stacking       : {results['learned_stacking']    * 100:.2f}%")
    print(f"{'='*52}")
    print(
        f"  Best strategy : {best_strategy} ({ensemble_only[best_strategy] * 100:.2f}%)"
    )


if __name__ == "__main__":
    main()
