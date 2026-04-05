"""
scripts/inference.py
=====================
Single image inference with GradCAM visualization.

Strategy performance (k-fold test results, 2026-03-30):
    simple_avg   98.36% ← default (best)
    weighted_avg 98.36% (equal to simple)
    swin         98.26% (individual)
    convnext     97.10% (individual, model soup)
    learned_stacking 94.40% (distribution mismatch — see ensemble.py)

Usage:
    python scripts/inference.py --image leaf.jpg
    python scripts/inference.py --image leaf.jpg --strategy weighted_avg
"""

import os, sys, json, yaml, argparse
import torch, torch.nn.functional as F
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.transforms import get_val_transforms
from src.models.swin_model import get_swin_model
from src.models.convnext_model import get_convnext_model
from src.models.ensemble import (LearnedStackingEnsemble,
                                  simple_average_ensemble, weighted_average_ensemble)

RESULTS_DIR = "outputs/inference"


class GradCAM:
    """GradCAM for ConvNeXt — shows which leaf regions influenced the prediction."""
    def __init__(self, model):
        self.model = model; self.grads = None; self.acts = None
        tl = model.stages[-1].blocks[-1]
        tl.register_forward_hook(lambda m,i,o: setattr(self,"acts",o.detach()))
        tl.register_full_backward_hook(lambda m,gi,go: setattr(self,"grads",go[0].detach()))

    def generate(self, tensor, class_idx):
        self.model.eval()
        out = self.model(tensor.unsqueeze(0).requires_grad_(True))
        self.model.zero_grad(); out[0, class_idx].backward()
        cam = F.relu((self.grads.mean(dim=(2,3),keepdim=True)*self.acts).sum(1,keepdim=True))
        cam = F.interpolate(cam,(224,224),mode="bilinear",align_corners=False).squeeze().numpy()
        return (cam-cam.min())/(cam.max()-cam.min()+1e-8)


def load_models(cfg, device):
    nc = cfg["data"]["num_classes"]
    swin = get_swin_model(nc, False)
    swin.load_state_dict(torch.load("outputs/checkpoints/best_swin.pth", map_location=device))
    swin = swin.to(device).eval()
    conv = get_convnext_model(nc, False)
    conv.load_state_dict(torch.load("outputs/checkpoints/best_convnext.pth", map_location=device))
    conv = conv.to(device).eval()
    meta = LearnedStackingEnsemble(nc, cfg["ensemble"]["meta_hidden_dim"])
    if os.path.exists("outputs/checkpoints/best_ensemble.pth"):
        meta.load_state_dict(torch.load("outputs/checkpoints/best_ensemble.pth", map_location=device))
    meta.eval()
    with open("outputs/checkpoints/weighted_avg_config.json") as f:
        wavg = json.load(f)
    return swin, conv, meta, wavg


@torch.no_grad()
def run_strategy(ls, lc, meta, wavg, strategy):
    if strategy == "simple_avg":
        return simple_average_ensemble(ls, lc)
    elif strategy == "weighted_avg":
        return weighted_average_ensemble(ls, lc, wavg["w_swin"], wavg["w_convnext"])
    elif strategy == "learned_stacking":
        return F.softmax(meta(ls, lc), dim=1)
    elif strategy == "swin":
        return F.softmax(ls, dim=1)
    return F.softmax(lc, dim=1)


def run_inference(image_path, strategy="simple_avg"):
    with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
    device      = torch.device(cfg["training"]["device"] if torch.cuda.is_available() else "cpu")
    class_names = pd.read_csv("data/splits/class_map.csv").sort_values("label")["class_name"].tolist()

    swin, conv, meta, wavg = load_models(cfg, device)
    t   = get_val_transforms(cfg["data"]["image_size"])
    img = Image.open(image_path).convert("RGB")
    tensor = t(img)

    with torch.no_grad():
        ls = swin(tensor.unsqueeze(0).to(device)).cpu()
        lc = conv(tensor.unsqueeze(0).to(device)).cpu()
    probs = run_strategy(ls, lc, meta, wavg, strategy).squeeze(0)

    top3_v, top3_i = torch.topk(probs, 3)
    top3 = [{"rank":i+1,"class":class_names[idx],"label_idx":idx.item(),
              "confidence":round(val.item()*100,2)}
            for i,(val,idx) in enumerate(zip(top3_v,top3_i))]

    cam       = GradCAM(conv).generate(tensor.to(device), top3[0]["label_idx"])
    img_disp  = img.resize((224,224))
    ps        = F.softmax(ls,dim=1).squeeze(0)
    pc_       = F.softmax(lc,dim=1).squeeze(0)
    sc        = round(ps[top3[0]["label_idx"]].item()*100,2)
    cc        = round(pc_[top3[0]["label_idx"]].item()*100,2)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(image_path))[0]
    fig, ax = plt.subplots(1,3,figsize=(16,5)); fig.patch.set_facecolor("#f9fafb")
    ax[0].imshow(img_disp); ax[0].set_title("Original Leaf",fontweight="bold"); ax[0].axis("off")
    ax[1].imshow(img_disp); ax[1].imshow(cam,cmap="jet",alpha=0.45)
    ax[1].set_title("GradCAM",fontweight="bold"); ax[1].axis("off")
    labels=[f"#{p['rank']} {p['class']}" for p in top3]; confs=[p["confidence"] for p in top3]
    ax[2].barh(labels[::-1],confs[::-1],color=["#374151","#6b7280","#9ca3af"][::-1],height=0.5)
    ax[2].set_xlim(0,115); ax[2].set_title(f"Top-3 · {strategy}",fontweight="bold")
    ax[2].text(0.02,-0.12,f"Swin:{sc:.1f}% Conv:{cc:.1f}%",transform=ax[2].transAxes,fontsize=8,color="#9ca3af")
    plt.suptitle(f"MediPlant AI · {top3[0]['class']} ({top3[0]['confidence']:.1f}%)",fontweight="bold",y=1.02)
    plt.tight_layout()
    fig_path = f"{RESULTS_DIR}/{base}_{strategy}_result.png"
    plt.savefig(fig_path,dpi=150,bbox_inches="tight",facecolor="#f9fafb"); plt.close()

    result = {"image":image_path,"strategy":strategy,"top3":top3,"swin_conf":sc,"conv_conf":cc}
    with open(f"{RESULTS_DIR}/{base}_{strategy}_result.json","w") as f: json.dump(result,f,indent=2)

    print(f"\n{'='*52}")
    for p in top3: print(f"  #{p['rank']} {p['class']:<30} {p['confidence']:.2f}%")
    print(f"  Swin:{sc:.1f}%  ConvNeXt:{cc:.1f}%")
    print(f"  Figure → {fig_path}")
    return result, fig_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image",    required=True)
    p.add_argument("--strategy", default="simple_avg",
                   choices=["simple_avg","weighted_avg","learned_stacking","swin","convnext"])
    a = p.parse_args(); run_inference(a.image, a.strategy)
