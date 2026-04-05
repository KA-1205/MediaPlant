"""
scripts/train.py
=================
Train Swin or ConvNeXt with automatic CSV logging.
Logs every epoch immediately — no data lost if training crashes.

Usage:
    python scripts/train.py --model swin
    python scripts/train.py --model convnext
"""

import argparse, csv, os, sys, random, yaml
import torch, numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import MediPlantDataset
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.training.trainer import Trainer


def set_seed(s):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)


class CSVLogger:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch","train_loss","train_acc","val_loss","val_acc","best_val_acc"])
        self.path = path

    def log(self, epoch, tl, ta, vl, va, best):
        with open(self.path, "a", newline="") as f:
            csv.writer(f).writerow([epoch,round(tl,6),round(ta,6),round(vl,6),round(va,6),round(best,6)])


def main(args):
    with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
    set_seed(cfg["project"]["seed"])

    device = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    nc, sz, bs = cfg["data"]["num_classes"], cfg["data"]["image_size"], cfg["data"]["batch_size"]
    accum, mcfg = cfg["training"]["grad_accum_steps"], cfg[args.model]

    print(f"Model: {args.model.upper()} | Device: {device} | Effective batch: {bs*accum}")

    train_dl = DataLoader(MediPlantDataset("data/splits/train.csv", get_train_transforms(sz)),
                          batch_size=bs, shuffle=True, num_workers=cfg["data"]["num_workers"], pin_memory=True)
    val_dl   = DataLoader(MediPlantDataset("data/splits/val.csv",   get_val_transforms(sz)),
                          batch_size=bs, shuffle=False, num_workers=cfg["data"]["num_workers"], pin_memory=True)

    model = (get_swin_model(nc, mcfg["pretrained"], mcfg["dropout"])
             if args.model == "swin"
             else get_convnext_model(nc, mcfg["pretrained"], mcfg["dropout"])).to(device)

    optimizer = AdamW(model.parameters(), lr=mcfg["learning_rate"], weight_decay=mcfg["weight_decay"])
    trainer   = Trainer(model, optimizer, CosineAnnealingLR(optimizer, mcfg["epochs"]),
                        device, cfg["training"]["amp"], accum)

    os.makedirs(cfg["training"]["checkpoint_dir"], exist_ok=True)
    logger   = CSVLogger(f"{cfg['training']['log_dir']}/{args.model}_training_log.csv")
    best, pc = 0.0, 0

    for epoch in range(1, mcfg["epochs"]+1):
        tl, ta = trainer.train_epoch(train_dl)
        vl, va = trainer.evaluate(val_dl)
        saved  = ""
        if va > best:
            best = va; pc = 0
            torch.save(model.state_dict(), f"{cfg['training']['checkpoint_dir']}/best_{args.model}.pth")
            saved = " << saved"
        else:
            pc += 1
        print(f"Epoch [{epoch:03d}/{mcfg['epochs']}] Train:{ta:.4f} Val:{va:.4f} Best:{best:.4f}{saved}")
        logger.log(epoch, tl, ta, vl, va, best)
        if pc >= cfg["training"].get("early_stopping_patience", 999):
            print(f"Early stopping at epoch {epoch}"); break

    print(f"Done. Best Val Acc: {best:.4f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, choices=["swin","convnext"])
    main(p.parse_args())
