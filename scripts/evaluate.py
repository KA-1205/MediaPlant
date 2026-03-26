"""
Phase 3 — Final Test Set Evaluation
=====================================
Evaluates all models and ensemble strategies on the held-out test set.
Generates all data needed for a research paper:
  - Accuracy, F1, Precision, Recall
  - Per-class accuracy
  - Confusion matrix
  - Full comparison table
  - All results saved to outputs/results/

Usage:
    python scripts/evaluate.py
"""

import os
import sys
import json
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)
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

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/results/figures"


# ── Helpers ──────────────────────────────────────────────────────────────────

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


@torch.no_grad()
def extract_probabilities(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    for images, labels in tqdm(loader, desc="  Extracting"):
        images = images.to(device)
        probs  = F.softmax(model(images), dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(labels)
    return torch.cat(all_probs, dim=0), torch.cat(all_labels, dim=0)


def compute_metrics(y_true, y_pred, class_names):
    """Compute all metrics for a single model/strategy."""
    return {
        "accuracy":          round(accuracy_score(y_true, y_pred), 6),
        "f1_macro":          round(f1_score(y_true, y_pred, average="macro",     zero_division=0), 6),
        "f1_weighted":       round(f1_score(y_true, y_pred, average="weighted",  zero_division=0), 6),
        "precision_macro":   round(precision_score(y_true, y_pred, average="macro",    zero_division=0), 6),
        "recall_macro":      round(recall_score(y_true, y_pred, average="macro",       zero_division=0), 6),
    }


# ── Confusion Matrix ──────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred, class_names, model_name):
    """Save full confusion matrix as a heatmap figure."""
    cm       = confusion_matrix(y_true, y_pred)
    fig_path = os.path.join(FIGURES_DIR, f"confusion_matrix_{model_name}.png")

    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        annot=False,           # too many classes for per-cell annotation
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
    )
    plt.title(f"Confusion Matrix — {model_name}", fontsize=16, fontweight="bold", pad=20)
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(rotation=0,  fontsize=6)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved confusion matrix -> {fig_path}")
    return fig_path


# ── Per-Class Accuracy ────────────────────────────────────────────────────────

def save_per_class_accuracy(y_true, y_pred, class_names, model_name):
    """Save per-class accuracy to CSV and bar chart."""
    cm              = confusion_matrix(y_true, y_pred)
    per_class_acc   = cm.diagonal() / cm.sum(axis=1)

    # CSV
    df = pd.DataFrame({
        "class":    class_names,
        "accuracy": per_class_acc,
        "correct":  cm.diagonal(),
        "total":    cm.sum(axis=1),
    }).sort_values("accuracy", ascending=True)

    csv_path = os.path.join(RESULTS_DIR, f"per_class_accuracy_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"    Saved per-class accuracy -> {csv_path}")

    # Bar chart — bottom 20 hardest classes
    fig_path = os.path.join(FIGURES_DIR, f"per_class_accuracy_{model_name}.png")
    bottom20 = df.head(20)

    plt.figure(figsize=(12, 7))
    bars = plt.barh(bottom20["class"], bottom20["accuracy"] * 100, color="tomato", edgecolor="black", linewidth=0.5)
    plt.xlabel("Accuracy (%)", fontsize=12)
    plt.title(f"20 Hardest Classes — {model_name}", fontsize=14, fontweight="bold")
    plt.xlim(0, 100)
    for bar, acc in zip(bars, bottom20["accuracy"]):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{acc * 100:.1f}%", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved per-class chart   -> {fig_path}")

    return df


# ── Training Curves ───────────────────────────────────────────────────────────

def save_training_curves(log_dir="outputs/logs"):
    """Plot training curves for both models from saved CSVs."""
    for model_name in ["swin", "convnext"]:
        log_path = os.path.join(log_dir, f"{model_name}_training_log.csv")
        if not os.path.exists(log_path):
            print(f"    No log found for {model_name} — skipping curve")
            continue

        df       = pd.read_csv(log_path)
        fig_path = os.path.join(FIGURES_DIR, f"training_curve_{model_name}.png")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Loss
        ax1.plot(df["epoch"], df["train_loss"], label="Train Loss", color="royalblue")
        ax1.plot(df["epoch"], df["val_loss"],   label="Val Loss",   color="tomato")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_title(f"{model_name.upper()} — Loss Curve")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Accuracy
        ax2.plot(df["epoch"], df["train_acc"] * 100, label="Train Acc", color="royalblue")
        ax2.plot(df["epoch"], df["val_acc"]   * 100, label="Val Acc",   color="tomato")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_title(f"{model_name.upper()} — Accuracy Curve")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(f"{model_name.upper()} Training Curves", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved training curve -> {fig_path}")


# ── Comparison Table ──────────────────────────────────────────────────────────

def save_comparison_table(all_results):
    """Save final comparison table as CSV and styled figure."""
    rows = []
    for name, metrics in all_results.items():
        rows.append({
            "Method":             name,
            "Accuracy (%)":       round(metrics["accuracy"]        * 100, 2),
            "F1 Macro (%)":       round(metrics["f1_macro"]        * 100, 2),
            "F1 Weighted (%)":    round(metrics["f1_weighted"]     * 100, 2),
            "Precision Macro (%)":round(metrics["precision_macro"] * 100, 2),
            "Recall Macro (%)":   round(metrics["recall_macro"]    * 100, 2),
        })

    df       = pd.DataFrame(rows)
    csv_path = os.path.join(RESULTS_DIR, "final_comparison_table.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved comparison table -> {csv_path}")

    # Figure
    fig_path = os.path.join(FIGURES_DIR, "final_comparison_table.png")
    fig, ax  = plt.subplots(figsize=(14, len(rows) * 0.8 + 2))
    ax.axis("off")
    table = ax.table(
        cellText    = df.values,
        colLabels   = df.columns,
        cellLoc     = "center",
        loc         = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)

    # Highlight best row
    best_idx = df["Accuracy (%)"].idxmax()
    for col in range(len(df.columns)):
        table[best_idx + 1, col].set_facecolor("#d4edda")   # green highlight

    plt.title("Final Model Comparison — Test Set", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved comparison figure -> {fig_path}")

    return df


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ensure_dirs()

    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device      = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes = cfg["data"]["num_classes"]
    image_size  = cfg["data"]["image_size"]
    batch_size  = cfg["data"]["batch_size"]

    # Load class names
    class_map   = pd.read_csv("data/splits/class_map.csv")
    class_names = class_map.sort_values("label")["class_name"].tolist()

    print(f"Device      : {device}")
    print(f"Classes     : {num_classes}")
    print(f"Test set    : data/splits/test.csv")
    print("=" * 60)

    # ── Test DataLoader ──
    test_ds = MediPlantDataset("data/splits/test.csv", get_val_transforms(image_size))
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                         num_workers=cfg["data"]["num_workers"])

    # ── Load Models ──
    print("\n[1/5] Loading models...")
    swin_model = get_swin_model(num_classes, pretrained=False)
    swin_model.load_state_dict(torch.load("outputs/checkpoints/best_swin.pth", map_location=device))
    swin_model = swin_model.to(device)
    print("  Swin loaded")

    conv_model = get_convnext_model(num_classes, pretrained=False)
    conv_model.load_state_dict(torch.load("outputs/checkpoints/best_convnext.pth", map_location=device))
    conv_model = conv_model.to(device)
    print("  ConvNeXt loaded")

    meta_model = LearnedStackingEnsemble(num_classes=num_classes,
                                          hidden_dim=cfg["ensemble"]["meta_hidden_dim"])
    meta_model.load_state_dict(torch.load("outputs/checkpoints/best_ensemble.pth", map_location=device))
    meta_model.eval()
    print("  Ensemble meta-classifier loaded")

    with open("outputs/checkpoints/weighted_avg_config.json") as f:
        wavg_cfg = json.load(f)
    w_swin = wavg_cfg["w_swin"]
    w_conv = wavg_cfg["w_convnext"]
    print(f"  Weighted avg config loaded (w_swin={w_swin:.3f}, w_conv={w_conv:.3f})")

    # ── Extract Test Probabilities ──
    print("\n[2/5] Extracting test set probabilities...")
    test_probs_swin, test_labels = extract_probabilities(swin_model, test_dl, device)
    test_probs_conv, _           = extract_probabilities(conv_model,  test_dl, device)
    print(f"  Test samples: {len(test_labels)}")

    y_true = test_labels.numpy()

    # ── Evaluate All Strategies ──
    print("\n[3/5] Evaluating all strategies on test set...")

    strategies = {
        "Swin Transformer":    test_probs_swin.argmax(1).numpy(),
        "ConvNeXt":            test_probs_conv.argmax(1).numpy(),
        "Simple Average":      simple_average_ensemble(test_probs_swin, test_probs_conv).argmax(1).numpy(),
        "Weighted Average":    weighted_average_ensemble(test_probs_swin, test_probs_conv, w_swin, w_conv).argmax(1).numpy(),
        "Learned Stacking":    meta_model(test_probs_swin, test_probs_conv).argmax(1).numpy(),
    }

    all_results = {}
    for name, y_pred in strategies.items():
        metrics = compute_metrics(y_true, y_pred, class_names)
        all_results[name] = metrics
        print(f"  {name:<25} Acc: {metrics['accuracy']*100:.2f}%  F1: {metrics['f1_macro']*100:.2f}%")

    # Save all metrics to JSON
    json_path = os.path.join(RESULTS_DIR, "final_test_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved all metrics -> {json_path}")

    # ── Generate Figures ──
    print("\n[4/5] Generating figures...")

    # Training curves
    print("  Training curves:")
    save_training_curves()

    # Confusion matrix + per-class accuracy for best model
    best_model_name = max(all_results, key=lambda k: all_results[k]["accuracy"])
    best_preds      = strategies[best_model_name]

    print(f"\n  Confusion matrix for best model ({best_model_name}):")
    save_confusion_matrix(y_true, best_preds, class_names, best_model_name.replace(" ", "_"))

    print(f"\n  Per-class accuracy for best model ({best_model_name}):")
    per_class_df = save_per_class_accuracy(y_true, best_preds, class_names, best_model_name.replace(" ", "_"))

    # Also save per-class for Swin (individual baseline for paper)
    print(f"\n  Per-class accuracy for Swin (baseline):")
    save_per_class_accuracy(y_true, strategies["Swin Transformer"], class_names, "Swin_Transformer")

    # ── Final Comparison Table ──
    print("\n[5/5] Building final comparison table...")
    comparison_df = save_comparison_table(all_results)

    # ── Print Final Summary ──
    print("\n" + "=" * 60)
    print("FINAL TEST SET RESULTS")
    print("=" * 60)
    print(comparison_df.to_string(index=False))
    print("=" * 60)

    best_acc    = comparison_df["Accuracy (%)"].max()
    best_method = comparison_df.loc[comparison_df["Accuracy (%)"].idxmax(), "Method"]
    print(f"\nBest Method   : {best_method}")
    print(f"Best Accuracy : {best_acc}%")

    print("\nAll outputs saved to outputs/results/")
    print("\nOutputs summary:")
    print("  outputs/results/final_test_results.json          <- all metrics")
    print("  outputs/results/final_comparison_table.csv       <- paper table")
    print("  outputs/results/per_class_accuracy_*.csv         <- per class data")
    print("  outputs/results/figures/confusion_matrix_*.png   <- confusion matrix")
    print("  outputs/results/figures/per_class_accuracy_*.png <- hardest classes")
    print("  outputs/results/figures/training_curve_*.png     <- loss/acc curves")
    print("  outputs/results/figures/final_comparison_table.png <- visual table")


if __name__ == "__main__":
    main()