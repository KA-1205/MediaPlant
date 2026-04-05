"""
scripts/evaluate.py
=====================
Full test set evaluation — generates all paper figures and metrics.

K-Fold results reference (2026-03-30):
    Best strategy: Simple Average = 98.36%
    Uses original best_swin.pth + best_convnext.pth (NOT model soup)

Usage:
    python scripts/evaluate.py
"""

import os, sys, json, yaml
import torch, torch.nn.functional as F
import numpy as np, pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
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

R = "outputs/results"
FG = "outputs/results/figures"


@torch.no_grad()
def extract_logits(model, loader, device):
    """Raw logits — ensemble functions apply softmax internally."""
    model.eval()
    all_l, all_lb = [], []
    for imgs, labels in tqdm(loader, desc="  Extracting"):
        all_l.append(model(imgs.to(device)).cpu())
        all_lb.append(labels)
    return torch.cat(all_l), torch.cat(all_lb)


def metrics(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 6),
        "f1_macro": round(
            f1_score(y_true, y_pred, average="macro", zero_division=0), 6
        ),
        "f1_weighted": round(
            f1_score(y_true, y_pred, average="weighted", zero_division=0), 6
        ),
        "precision_macro": round(
            precision_score(y_true, y_pred, average="macro", zero_division=0), 6
        ),
        "recall_macro": round(
            recall_score(y_true, y_pred, average="macro", zero_division=0), 6
        ),
    }


def save_confusion_matrix(y_true, y_pred, class_names, name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(24, 20))
    sns.heatmap(
        cm,
        annot=False,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.3,
    )
    plt.title(f"Confusion Matrix — {name}", fontsize=16, fontweight="bold")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks(rotation=90, fontsize=6)
    plt.yticks(fontsize=6)
    plt.tight_layout()
    path = f"{FG}/confusion_matrix_{name.replace(' ','_')}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")


def save_per_class(y_true, y_pred, class_names, name):
    cm = confusion_matrix(y_true, y_pred)
    acc = cm.diagonal() / cm.sum(axis=1)
    df = pd.DataFrame(
        {
            "class": class_names,
            "accuracy": acc,
            "correct": cm.diagonal(),
            "total": cm.sum(axis=1),
        }
    ).sort_values("accuracy")
    df.to_csv(f"{R}/per_class_accuracy_{name.replace(' ','_')}.csv", index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(df.head(15)["class"], df.head(15)["accuracy"] * 100, color="steelblue")
    ax.set_title(f"15 Hardest Classes — {name}", fontweight="bold")
    ax.set_xlabel("Accuracy (%)")
    plt.tight_layout()
    plt.savefig(f"{FG}/per_class_{name.replace(' ','_')}.png", dpi=150)
    plt.close()
    print(f"    Saved → {R}/per_class_accuracy_{name.replace(' ','_')}.csv")


def save_training_curves():
    for m in ["swin", "convnext"]:
        path = f"outputs/logs/{m}_training_log.csv"
        if not os.path.exists(path):
            print(f"    No log for {m} — skip")
            continue
        df = pd.read_csv(path)
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4))
        a1.plot(df["epoch"], df["train_loss"], label="Train")
        a1.plot(df["epoch"], df["val_loss"], label="Val", linestyle="--")
        a1.set_title(f"{m.upper()} Loss")
        a1.legend()
        a1.grid(True, alpha=0.3)
        a2.plot(df["epoch"], df["train_acc"] * 100, label="Train")
        a2.plot(df["epoch"], df["val_acc"] * 100, label="Val", linestyle="--")
        a2.set_title(f"{m.upper()} Accuracy (%)")
        a2.legend()
        a2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{FG}/training_curve_{m}.png", dpi=150)
        plt.close()
        print(f"    Saved training curve → {m}")


def main():
    os.makedirs(R, exist_ok=True)
    os.makedirs(FG, exist_ok=True)
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)
    device = torch.device(
        cfg["training"]["device"] if torch.cuda.is_available() else "cpu"
    )
    nc = cfg["data"]["num_classes"]
    names = (
        pd.read_csv("data/splits/class_map.csv")
        .sort_values("label")["class_name"]
        .tolist()
    )

    test_dl = DataLoader(
        MediPlantDataset(
            "data/splits/test.csv", get_val_transforms(cfg["data"]["image_size"])
        ),
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"]["num_workers"],
    )
    print(f"Test: {len(test_dl.dataset)} images | Device: {device}")

    print("\n[1/4] Loading models...")
    swin = get_swin_model(nc, False)
    swin.load_state_dict(
        torch.load("outputs/checkpoints/best_swin.pth", map_location=device)
    )
    swin = swin.to(device).eval()
    conv = get_convnext_model(nc, False)
    conv.load_state_dict(
        torch.load("outputs/checkpoints/best_convnext.pth", map_location=device)
    )
    conv = conv.to(device).eval()

    meta_loaded = False
    if os.path.exists("outputs/checkpoints/best_ensemble.pth"):
        meta = LearnedStackingEnsemble(nc, cfg["ensemble"]["meta_hidden_dim"])
        meta.load_state_dict(
            torch.load("outputs/checkpoints/best_ensemble.pth", map_location=device)
        )
        meta.eval()
        meta_loaded = True
        print("  Loaded best_ensemble.pth")
    else:
        print("  best_ensemble.pth not found — skipping Learned Stacking")

    with open("outputs/checkpoints/weighted_avg_config.json") as f:
        wavg = json.load(f)

    print("\n[2/4] Extracting test logits...")
    sl, tl_ = extract_logits(swin, test_dl, device)
    cl, _ = extract_logits(conv, test_dl, device)
    y_true = tl_.numpy()

    print("\n[3/4] Evaluating strategies...")
    strats = {
        "Swin Transformer": sl.argmax(1).numpy(),
        "ConvNeXt": cl.argmax(1).numpy(),
        "Simple Average": simple_average_ensemble(sl, cl).argmax(1).numpy(),
        "Weighted Average": weighted_average_ensemble(
            sl, cl, wavg["w_swin"], wavg["w_convnext"]
        )
        .argmax(1)
        .numpy(),
    }
    if meta_loaded:
        strats["Learned Stacking"] = meta(sl, cl).argmax(1).numpy()

    results = {}
    for name, y_pred in strats.items():
        m = metrics(y_true, y_pred)
        results[name] = m
        print(f"  {name:<25} Acc:{m['accuracy']*100:.2f}%  F1:{m['f1_macro']*100:.2f}%")

    with open(f"{R}/final_test_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n[4/4] Generating figures...")
    save_training_curves()
    best_name = max(results, key=lambda k: results[k]["accuracy"])
    save_confusion_matrix(y_true, strats[best_name], names, best_name)
    save_per_class(y_true, strats[best_name], names, best_name)
    save_per_class(y_true, strats["Swin Transformer"], names, "Swin_Transformer")

    rows = [
        {
            "Method": k,
            "Accuracy (%)": round(v["accuracy"] * 100, 2),
            "F1 Macro (%)": round(v["f1_macro"] * 100, 2),
            "Precision (%)": round(v["precision_macro"] * 100, 2),
            "Recall (%)": round(v["recall_macro"] * 100, 2),
        }
        for k, v in results.items()
    ]
    pd.DataFrame(rows).to_csv(f"{R}/final_comparison_table.csv", index=False)

    print(f"\n{'='*55}\nFINAL TEST RESULTS\n{'='*55}")
    for r in rows:
        print(f"  {r['Method']:<25} {r['Accuracy (%)']:.2f}%")
    print(f"\nBest: {best_name} — {results[best_name]['accuracy']*100:.2f}%")
    print(f"All saved to {R}/")


if __name__ == "__main__":
    main()
