"""
Phase 4 — Inference Pipeline
==============================
Single image → plant name + confidence + top-3 + GradCAM

Usage:
    python scripts/inference.py --image path/to/leaf.jpg
    python scripts/inference.py --image path/to/leaf.jpg --strategy weighted_avg
"""

import os
import sys
import json
import yaml
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.models.ensemble import (
    LearnedStackingEnsemble,
    simple_average_ensemble,
    weighted_average_ensemble,
)

RESULTS_DIR = "outputs/inference"


# ── GradCAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    def __init__(self, model):
        self.model       = model
        self.gradients   = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        target_layer = self.model.stages[-1].blocks[-1]

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(forward_hook)
        target_layer.register_full_backward_hook(backward_hook)

    def generate(self, image_tensor, class_idx):
        self.model.eval()
        image_tensor = image_tensor.unsqueeze(0).requires_grad_(True)
        output       = self.model(image_tensor)
        self.model.zero_grad()
        output[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam     = (weights * self.activations).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam     = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Load Models ───────────────────────────────────────────────────────────────

def load_models(cfg, device):
    num_classes = cfg["data"]["num_classes"]

    swin = get_swin_model(num_classes, pretrained=False)
    swin.load_state_dict(torch.load("outputs/checkpoints/best_swin.pth", map_location=device))
    swin = swin.to(device).eval()

    conv = get_convnext_model(num_classes, pretrained=False)
    conv.load_state_dict(torch.load("outputs/checkpoints/best_convnext.pth", map_location=device))
    conv = conv.to(device).eval()

    meta = LearnedStackingEnsemble(num_classes=num_classes, hidden_dim=cfg["ensemble"]["meta_hidden_dim"])
    meta.load_state_dict(torch.load("outputs/checkpoints/best_ensemble.pth", map_location=device))
    meta.eval()

    with open("outputs/checkpoints/weighted_avg_config.json") as f:
        wavg_cfg = json.load(f)

    return swin, conv, meta, wavg_cfg


# ── Predict ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def predict(image_tensor, swin, conv, meta, wavg_cfg, strategy, device):
    """
    IMPORTANT: ensemble functions (simple_average_ensemble, weighted_average_ensemble)
    apply softmax internally — so we must pass RAW LOGITS, not probabilities.
    LearnedStackingEnsemble.forward() also applies softmax internally.
    """
    img = image_tensor.unsqueeze(0).to(device)

    # Get raw logits (no softmax here)
    logits_swin = swin(img).cpu()
    logits_conv = conv(img).cpu()

    if strategy == "simple_avg":
        # simple_average_ensemble applies softmax internally
        probs = simple_average_ensemble(logits_swin, logits_conv)

    elif strategy == "weighted_avg":
        # weighted_average_ensemble applies softmax internally
        probs = weighted_average_ensemble(
            logits_swin, logits_conv,
            w_swin     = wavg_cfg["w_swin"],
            w_convnext = wavg_cfg["w_convnext"],
        )

    elif strategy == "learned_stacking":
        # LearnedStackingEnsemble.forward() applies softmax internally
        # then returns logits from meta MLP — apply softmax at the end
        meta_logits = meta(logits_swin, logits_conv)
        probs       = F.softmax(meta_logits, dim=1)

    elif strategy == "swin":
        probs = F.softmax(logits_swin, dim=1)

    elif strategy == "convnext":
        probs = F.softmax(logits_conv, dim=1)

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return probs.squeeze(0)


# ── Top-3 ─────────────────────────────────────────────────────────────────────

def get_top3(probs, class_names):
    top3_vals, top3_idxs = torch.topk(probs, k=3)
    return [
        {
            "rank":       i + 1,
            "class":      class_names[idx.item()],
            "label_idx":  idx.item(),
            "confidence": round(val.item() * 100, 2),
        }
        for i, (val, idx) in enumerate(zip(top3_vals, top3_idxs))
    ]


# ── Save result figure ────────────────────────────────────────────────────────

def save_result_figure(original_image, cam, top3, image_path, strategy):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    base     = os.path.splitext(os.path.basename(image_path))[0]
    fig_path = os.path.join(RESULTS_DIR, f"{base}_{strategy}_result.png")

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.patch.set_facecolor("#f9faf9")

    # Original
    axes[0].imshow(original_image)
    axes[0].set_title("Original Leaf", fontsize=11, fontweight="bold", color="#1c2b20")
    axes[0].axis("off")

    # GradCAM overlay
    axes[1].imshow(original_image)
    axes[1].imshow(cam, cmap="jet", alpha=0.45)
    axes[1].set_title("GradCAM — ConvNeXt Focus", fontsize=11, fontweight="bold", color="#1c2b20")
    axes[1].axis("off")

    # Top-3 bar
    labels = [f"#{p['rank']}  {p['class']}" for p in top3]
    confs  = [p["confidence"] for p in top3]
    colors = ["#2f7d45", "#52a36a", "#89c997"]

    bars = axes[2].barh(labels[::-1], confs[::-1], color=colors[::-1], edgecolor="none", height=0.5)
    axes[2].set_xlabel("Confidence (%)", fontsize=9, color="#4a6b52")
    axes[2].set_title(f"Top-3 Predictions ({strategy})", fontsize=11, fontweight="bold", color="#1c2b20")
    axes[2].set_xlim(0, 110)
    axes[2].set_facecolor("#f9faf9")
    axes[2].spines[["top", "right"]].set_visible(False)
    axes[2].spines[["left", "bottom"]].set_color("#d0e0d2")
    axes[2].tick_params(colors="#4a6b52", labelsize=8)

    for bar, conf in zip(bars, confs[::-1]):
        axes[2].text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                     f"{conf:.1f}%", va="center", fontsize=9,
                     fontweight="bold", color="#2f7d45", fontfamily="monospace")

    plt.suptitle(
        f"MediPlant AI  ·  {top3[0]['class']}  ({top3[0]['confidence']:.1f}%)",
        fontsize=13, fontweight="bold", color="#1c2b20", y=1.02
    )
    fig.patch.set_facecolor("#f9faf9")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=150, bbox_inches="tight", facecolor="#f9faf9")
    plt.close()
    return fig_path


# ── Main ─────────────────────────────────────────────────────────────────────

def run_inference(image_path, strategy="weighted_avg"):
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    device      = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    image_size  = cfg["data"]["image_size"]

    class_map   = pd.read_csv("data/splits/class_map.csv")
    class_names = class_map.sort_values("label")["class_name"].tolist()

    print("Loading models...")
    swin, conv, meta, wavg_cfg = load_models(cfg, device)

    transform      = get_val_transforms(image_size)
    original_image = Image.open(image_path).convert("RGB").resize((224, 224))
    image_tensor   = transform(original_image)

    print(f"Running inference (strategy: {strategy})...")
    probs = predict(image_tensor, swin, conv, meta, wavg_cfg, strategy, device)
    top3  = get_top3(probs, class_names)

    print("Generating GradCAM...")
    gradcam  = GradCAM(conv)
    cam      = gradcam.generate(image_tensor.to(device), top3[0]["label_idx"])

    fig_path  = save_result_figure(original_image, cam, top3, image_path, strategy)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base      = os.path.splitext(os.path.basename(image_path))[0]
    json_path = os.path.join(RESULTS_DIR, f"{base}_{strategy}_result.json")
    result    = {"image": image_path, "strategy": strategy, "top3": top3}
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n" + "=" * 50)
    print(f"  Image    : {image_path}")
    print(f"  Strategy : {strategy}")
    print("  Top-3 Predictions:")
    for p in top3:
        print(f"    #{p['rank']}  {p['class']:<30}  {p['confidence']:.2f}%")
    print("=" * 50)
    print(f"  Figure -> {fig_path}")
    print(f"  JSON   -> {json_path}")

    return result, fig_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",    type=str, required=True)
    parser.add_argument("--strategy", type=str, default="weighted_avg",
                        choices=["simple_avg", "weighted_avg", "learned_stacking", "swin", "convnext"])
    args = parser.parse_args()
    run_inference(args.image, args.strategy)