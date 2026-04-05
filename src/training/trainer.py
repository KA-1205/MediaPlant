"""
src/training/trainer.py
========================
Trainer with gradient accumulation and AMP support.

With batch_size=8 and grad_accum_steps=4:
    Effective batch = 32, VRAM usage = batch_size=8 worth
    Suitable for RTX 3050 4GB VRAM
"""

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, scheduler, device,
                 use_amp=True, grad_accum_steps=4):
        self.model            = model
        self.optimizer        = optimizer
        self.scheduler        = scheduler
        self.device           = device
        self.scaler           = GradScaler() if use_amp else None
        self.criterion        = nn.CrossEntropyLoss()
        self.grad_accum_steps = grad_accum_steps

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        self.optimizer.zero_grad()

        for step, (images, labels) in enumerate(tqdm(loader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)

            if self.scaler:
                with autocast():
                    out  = self.model(images)
                    loss = self.criterion(out, labels) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                out  = self.model(images)
                loss = self.criterion(out, labels) / self.grad_accum_steps
                loss.backward()

            if (step + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer); self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum_steps * images.size(0)
            _, preds    = out.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        if self.scheduler: self.scheduler.step()
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(self.device), labels.to(self.device)
            out            = self.model(images)
            loss           = self.criterion(out, labels)
            total_loss    += loss.item() * images.size(0)
            _, preds       = out.max(1)
            correct       += preds.eq(labels).sum().item()
            total         += labels.size(0)

        return total_loss / total, correct / total
