"""
Train Swin Transformer or ConvNeXt individually.
Saves epoch logs to CSV automatically so no data is lost.

Usage:
    python scripts/train.py --model swin
    python scripts/train.py --model convnext
"""

import argparse
import yaml
import csv
import os
import sys
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MediPlantDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.training.trainer import Trainer


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CSVLogger:
    """Saves every epoch result to CSV immediately — no data ever lost."""

    def __init__(self, filepath):
        self.filepath = filepath
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(self.filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "best_val_acc"])
        print(f"  Logger -> {self.filepath}")

    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, best_val_acc):
        with open(self.filepath, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(train_loss,   6),
                round(train_acc,    6),
                round(val_loss,     6),
                round(val_acc,      6),
                round(best_val_acc, 6),
            ])


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["project"]["seed"])

    device           = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes      = cfg["data"]["num_classes"]
    image_size       = cfg["data"]["image_size"]
    batch_size       = cfg["data"]["batch_size"]
    grad_accum_steps = cfg["training"]["grad_accum_steps"]
    model_cfg        = cfg[args.model]

    print(f"Model        : {args.model.upper()}")
    print(f"Device       : {device}")
    print(f"Classes      : {num_classes}")
    print(f"Batch size   : {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"Epochs       : {model_cfg['epochs']}")

    # ── Data ──────────────────────────────────────────────────────────────────
    train_ds = MediPlantDataset("data/splits/train.csv", get_train_transforms(image_size))
    val_ds   = MediPlantDataset("data/splits/val.csv",   get_val_transforms(image_size))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                          num_workers=cfg["data"]["num_workers"], pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    if args.model == "swin":
        model = get_swin_model(num_classes, pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"])
    else:
        model = get_convnext_model(num_classes, pretrained=model_cfg["pretrained"], dropout=model_cfg["dropout"])
    model = model.to(device)

    # ── Optimizer & Scheduler ─────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=model_cfg["learning_rate"], weight_decay=model_cfg["weight_decay"])
    scheduler = CosineAnnealingLR(optimizer, T_max=model_cfg["epochs"])
    trainer   = Trainer(
        model            = model,
        optimizer        = optimizer,
        scheduler        = scheduler,
        device           = device,
        use_amp          = cfg["training"]["amp"],
        grad_accum_steps = grad_accum_steps,
    )

    # ── Logger & checkpoint dir ───────────────────────────────────────────────
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    log_path = os.path.join(cfg["training"]["log_dir"], f"{args.model}_training_log.csv")
    logger   = CSVLogger(log_path)

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0

    for epoch in range(1, model_cfg["epochs"] + 1):
        train_loss, train_acc = trainer.train_epoch(train_dl)
        val_loss,   val_acc   = trainer.evaluate(val_dl)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path    = os.path.join(cfg["training"]["checkpoint_dir"], f"best_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            saved_flag = " << saved best"
        else:
            saved_flag = ""

        print(f"Epoch [{epoch:03d}/{model_cfg['epochs']}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"Best: {best_val_acc:.4f}{saved_flag}")

        # Write to CSV immediately after every epoch
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, best_val_acc)

    print(f"\nTraining complete.")
    print(f"Best Val Accuracy : {best_val_acc:.4f}")
    print(f"Checkpoint        : outputs/checkpoints/best_{args.model}.pth")
    print(f"Full log          : {log_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["swin", "convnext"], required=True)
    args = parser.parse_args()
    main(args)