"""
Train Swin Transformer or ConvNeXt individually.
Usage:
    python scripts/train.py --model swin
    python scripts/train.py --model convnext
"""

import argparse
import yaml
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import sys, os
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


def main(args):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    # Set seed for reproducibility
    set_seed(cfg["project"]["seed"])

    device            = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    num_classes       = cfg["data"]["num_classes"]
    image_size        = cfg["data"]["image_size"]
    batch_size        = cfg["data"]["batch_size"]
    grad_accum_steps  = cfg["training"]["grad_accum_steps"]   # ← pulled from config
    model_cfg         = cfg[args.model]

    print(f"Model       : {args.model.upper()}")
    print(f"Device      : {device}")
    print(f"Classes     : {num_classes}")
    print(f"Batch size  : {batch_size} (effective: {batch_size * grad_accum_steps})")
    print(f"Accum steps : {grad_accum_steps}")

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

    trainer = Trainer(
        model            = model,
        optimizer        = optimizer,
        scheduler        = scheduler,
        device           = device,
        use_amp          = cfg["training"]["amp"],
        grad_accum_steps = grad_accum_steps,   # ← passed to trainer
    )

    # ── Training Loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)

    for epoch in range(1, model_cfg["epochs"] + 1):
        train_loss, train_acc = trainer.train_epoch(train_dl)
        val_loss,   val_acc   = trainer.evaluate(val_dl)

        print(f"Epoch [{epoch:03d}/{model_cfg['epochs']}] "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(cfg["training"]["checkpoint_dir"], f"best_{args.model}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"  >> Saved best model -> {save_path}")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=["swin", "convnext"], required=True,
                        help="Which model to train")
    args = parser.parse_args()
    main(args)