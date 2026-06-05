# =============================================================================
# generate_graphs.py — Generate real graphs from actual training data
# Saves PNG files to graphs/ folder for use in paper and frontend
# =============================================================================

import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

os.makedirs("graphs", exist_ok=True)

# Load data
with open("logs/metrics.json") as f:
    m = json.load(f)
with open("logs/comparison.json") as f:
    c = json.load(f)

TEAL   = "#45e4e4"
ORANGE = "#ffc29d"
PURPLE = "#c3c5dc"
BG     = "#111319"
CARD   = "#1e1f26"
TEXT   = "#e2e2eb"
MUTED  = "#bac9c9"

def dark_style(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(CARD)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    ax.title.set_color(TEXT)
    for spine in ax.spines.values():
        spine.set_edgecolor("#3c4949")

epochs = m["epoch"]

# ── 1. Loss Curve ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
dark_style(ax, fig)
ax.plot(epochs, m["train_loss"], color=TEAL,   lw=2,   label="Train Loss")
ax.plot(epochs, m["val_loss"],   color=ORANGE, lw=2, ls="--", label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("NT-Xent Loss")
ax.set_title("SimCLR Pre-Training: Loss Curve")
ax.legend(facecolor=CARD, edgecolor="#3c4949", labelcolor=TEXT, fontsize=9)
ax.grid(alpha=0.15, color="#3c4949")
plt.tight_layout()
plt.savefig("graphs/loss_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] graphs/loss_curve.png")

# ── 2. Contrastive Accuracy Curve ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
dark_style(ax, fig)
ax.plot(epochs, [v*100 for v in m["train_accuracy"]], color=TEAL,   lw=2,   label="Train Acc")
ax.plot(epochs, [v*100 for v in m["val_accuracy"]],   color=ORANGE, lw=2, ls="--", label="Val Acc")
ax.set_xlabel("Epoch"); ax.set_ylabel("Contrastive Top-1 Accuracy (%)")
ax.set_title("SimCLR Pre-Training: Contrastive Accuracy")
ax.legend(facecolor=CARD, edgecolor="#3c4949", labelcolor=TEXT, fontsize=9)
ax.grid(alpha=0.15, color="#3c4949")
plt.tight_layout()
plt.savefig("graphs/accuracy_curve.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] graphs/accuracy_curve.png")

# ── 3. Epoch Time ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
dark_style(ax, fig)
ax.bar(epochs, m["epoch_time"], color=TEAL, alpha=0.8, width=0.7)
ax.set_xlabel("Epoch"); ax.set_ylabel("Time (seconds)")
ax.set_title("SimCLR Pre-Training: Time per Epoch")
ax.grid(alpha=0.15, color="#3c4949", axis="y")
plt.tight_layout()
plt.savefig("graphs/epoch_time.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] graphs/epoch_time.png")

# ── 4. Model Comparison Bar Chart ─────────────────────────────────────────────
ORDER = ["resnet18_pretrained","resnet50_pretrained","efficientnet_b0","mobilenet_v3_small","final_model"]
NAMES = {
    "resnet18_pretrained": "ResNet18\n(ImageNet)",
    "resnet50_pretrained": "ResNet50\n(ImageNet)",
    "efficientnet_b0":     "EfficientNet-B0\n(ImageNet)",
    "mobilenet_v3_small":  "MobileNetV3\n(ImageNet)",
    "final_model":         "SimCLR+Linear\n(Ours, PCam only)",
}
COLORS = {
    "resnet18_pretrained": "#45e4e4",
    "resnet50_pretrained": "#37dbdb",
    "efficientnet_b0":     "#ffc29d",
    "mobilenet_v3_small":  "#ffb688",
    "final_model":         "#64c878",
}

keys   = [k for k in ORDER if k in c]
names  = [NAMES[k] for k in keys]
accs   = [c[k]["accuracy"] * 100 for k in keys]
colors = [COLORS[k] for k in keys]

fig, ax = plt.subplots(figsize=(10, 5))
dark_style(ax, fig)
bars = ax.barh(names, accs, color=colors, height=0.55)
ax.set_xlabel("Validation Accuracy (%)")
ax.set_title("Model Comparison: Linear Probe on PCam Validation Set")
ax.set_xlim(0, 100)
ax.grid(alpha=0.15, color="#3c4949", axis="x")

# Value labels
for bar, acc in zip(bars, accs):
    ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
            f"{acc:.2f}%", va="center", ha="left", color=TEXT, fontsize=9, fontweight="bold")

# Legend: supervised vs SSL
sup_patch = mpatches.Patch(color=TEAL,    label="ImageNet-pretrained (Supervised)")
ssl_patch = mpatches.Patch(color="#64c878", label="SSL from scratch (Ours)")
ax.legend(handles=[sup_patch, ssl_patch], facecolor=CARD, edgecolor="#3c4949",
          labelcolor=TEXT, fontsize=9, loc="lower right")

plt.tight_layout()
plt.savefig("graphs/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] graphs/model_comparison.png")

# ── 5. System Architecture Diagram ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
dark_style(ax, fig)
ax.set_xlim(0, 14); ax.set_ylim(0, 5)
ax.axis("off")
ax.set_title("HistoSight: High-Level System Architecture", color=TEXT, fontsize=13, fontweight="bold", pad=12)

def box(ax, x, y, w, h, label, sublabel="", color=TEAL, textcolor=BG):
    rect = plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="#3c4949", linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2 + (0.15 if sublabel else 0), label,
            ha="center", va="center", fontsize=9, fontweight="bold", color=textcolor, zorder=4)
    if sublabel:
        ax.text(x + w/2, y + h/2 - 0.25, sublabel,
                ha="center", va="center", fontsize=7.5, color=textcolor, alpha=0.85, zorder=4)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.5), zorder=5)

# Row 1: Data pipeline
box(ax, 0.2, 3.2, 2.0, 1.2, "PCam Dataset", "96×96 H&E patches", color="#282a30", textcolor=TEXT)
box(ax, 2.6, 3.2, 2.0, 1.2, "Macenko\nNormalization", "Stain invariance", color="#004e4e", textcolor=TEAL)
box(ax, 5.0, 3.2, 2.0, 1.2, "SimCLR\nAugmentation", "2 views / image", color="#004e4e", textcolor=TEAL)
box(ax, 7.4, 3.2, 2.0, 1.2, "ResNet-18\nEncoder", "512-d features", color=TEAL, textcolor=BG)
box(ax, 9.8, 3.2, 2.0, 1.2, "Projection\nHead (MLP)", "128-d embedding", color=TEAL, textcolor=BG)
box(ax, 12.0, 3.2, 1.8, 1.2, "NT-Xent\nLoss", "τ = 0.5", color="#433659", textcolor=TEXT)

arrow(ax, 2.2, 3.8, 2.6, 3.8)
arrow(ax, 4.6, 3.8, 5.0, 3.8)
arrow(ax, 7.0, 3.8, 7.4, 3.8)
arrow(ax, 9.4, 3.8, 9.8, 3.8)
arrow(ax, 11.8, 3.8, 12.0, 3.8)

# Row 2: Inference pipeline
box(ax, 0.2, 1.2, 2.0, 1.2, "Input Image", "JPEG/PNG/TIFF", color="#282a30", textcolor=TEXT)
box(ax, 2.6, 1.2, 2.0, 1.2, "Frozen\nEncoder", "best_simclr.pth", color="#282a30", textcolor=TEAL)
box(ax, 5.0, 1.2, 2.0, 1.2, "Linear\nClassifier", "512→2", color="#282a30", textcolor=TEAL)
box(ax, 7.4, 1.2, 2.0, 1.2, "Prediction", "Tumor / Normal", color="#64c878", textcolor=BG)
box(ax, 9.8, 1.2, 2.0, 1.2, "Grad-CAM\nSaliency", "Explainability", color="#433659", textcolor=TEXT)
box(ax, 12.0, 1.2, 1.8, 1.2, "FastAPI\nBackend", "/predict\n/gradcam", color="#282a30", textcolor=TEXT)

arrow(ax, 2.2, 1.8, 2.6, 1.8)
arrow(ax, 4.6, 1.8, 5.0, 1.8)
arrow(ax, 7.0, 1.8, 7.4, 1.8)
arrow(ax, 9.4, 1.8, 9.8, 1.8)
arrow(ax, 11.8, 1.8, 12.0, 1.8)

# Labels
ax.text(0.2, 4.55, "PRE-TRAINING PIPELINE", color=TEAL, fontsize=8, fontweight="bold", alpha=0.8)
ax.text(0.2, 2.55, "INFERENCE PIPELINE", color="#64c878", fontsize=8, fontweight="bold", alpha=0.8)

plt.tight_layout()
plt.savefig("graphs/system_architecture.png", dpi=150, bbox_inches="tight")
plt.close()
print("[SAVED] graphs/system_architecture.png")

print("\n[DONE] All graphs saved to graphs/")
