import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm


class Trainer:
    def __init__(self, model, optimizer, scheduler, device, use_amp=True, grad_accum_steps=4):
        self.model             = model
        self.optimizer         = optimizer
        self.scheduler         = scheduler
        self.device            = device
        self.scaler            = GradScaler() if use_amp else None
        self.criterion         = nn.CrossEntropyLoss()
        self.grad_accum_steps  = grad_accum_steps   # how many batches before weight update

    def train_epoch(self, loader):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        self.optimizer.zero_grad()  # zero once at the start

        for step, (images, labels) in enumerate(tqdm(loader, desc="Training")):
            images, labels = images.to(self.device), labels.to(self.device)

            if self.scaler:
                with autocast():
                    outputs = self.model(images)
                    # divide loss by accum steps so gradients scale correctly
                    loss    = self.criterion(outputs, labels) / self.grad_accum_steps
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(images)
                loss    = self.criterion(outputs, labels) / self.grad_accum_steps
                loss.backward()

            # only update weights every grad_accum_steps batches
            if (step + 1) % self.grad_accum_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad()

            # track real loss (multiply back to undo the division above)
            total_loss += loss.item() * self.grad_accum_steps * images.size(0)
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        if self.scheduler:
            self.scheduler.step()

        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(loader, desc="Evaluating"):
            images, labels = images.to(self.device), labels.to(self.device)
            outputs = self.model(images)
            loss    = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, preds    = outputs.max(1)
            correct    += preds.eq(labels).sum().item()
            total      += labels.size(0)

        return total_loss / total, correct / total