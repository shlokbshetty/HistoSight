# =============================================================================
# explainability.py — Interpretability Techniques for HistoSight
#
# Implements: Grad-CAM, SmoothGrad, Occlusion, LIME, SHAP
# Usage: python explainability.py --image <path> --method gradcam
# =============================================================================

import os, sys, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.pyplot as plt

# Import model loading from inference
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import inference

DEVICE = inference.DEVICE

# ======================
# 1. GRAD-CAM
# ======================
def gradcam(image_path, target_layer_name="layer4"):
    """Gradient-weighted Class Activation Mapping."""
    encoder, classifier = inference._load_models()
    encoder.eval()

    activations, gradients = {}, {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    target_layer = dict(encoder.named_modules()).get(target_layer_name)
    if not target_layer:
        raise ValueError(f"Layer '{target_layer_name}' not found.")
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img    = Image.open(image_path).convert("RGB")
        import stain_norm
        img    = stain_norm.normalize_image(img)
        tensor = inference.inference_transform(img).unsqueeze(0).to(DEVICE)

        feats  = encoder(tensor)
        logits = classifier(feats)
        pred_idx = logits.argmax(dim=1).item()

        encoder.zero_grad()
        classifier.zero_grad()
        logits[0, pred_idx].backward()

        weights = gradients["feat"].mean(dim=(2, 3), keepdim=True)
        cam     = (weights * activations["feat"]).sum(dim=1, keepdim=True)
        cam     = F.relu(cam)
        cam     = F.interpolate(cam, size=(96, 96), mode="bilinear", align_corners=False)
        cam     = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam, inference.CLASS_MAP[pred_idx]
    finally:
        fh.remove()
        bh.remove()

# ======================
# 2. SMOOTHGRAD
# ======================
def smoothgrad(image_path, n_samples=50, noise_level=0.15):
    """SmoothGrad: average gradients over noisy inputs."""
    encoder, classifier = inference._load_models()
    encoder.eval()

    img    = Image.open(image_path).convert("RGB")
    import stain_norm
    img    = stain_norm.normalize_image(img)
    tensor = inference.inference_transform(img).unsqueeze(0).to(DEVICE)

    grad_sum = torch.zeros_like(tensor)

    for _ in range(n_samples):
        noisy = tensor + torch.randn_like(tensor) * noise_level
        noisy.requires_grad = True

        feats  = encoder(noisy)
        logits = classifier(feats)
        pred_idx = logits.argmax(dim=1).item()

        encoder.zero_grad()
        classifier.zero_grad()
        logits[0, pred_idx].backward()

        grad_sum += noisy.grad.abs()

    grad_avg = grad_sum / n_samples
    saliency = grad_avg.squeeze().mean(dim=0).cpu().numpy()  # (H, W)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, inference.CLASS_MAP[pred_idx]

# ======================
# 3. OCCLUSION
# ======================
def occlusion(image_path, patch_size=16, stride=8):
    """Occlusion sensitivity: mask patches and measure confidence drop."""
    encoder, classifier = inference._load_models()
    encoder.eval()

    img    = Image.open(image_path).convert("RGB")
    import stain_norm
    img    = stain_norm.normalize_image(img)
    tensor = inference.inference_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats  = encoder(tensor)
        logits = classifier(feats)
        pred_idx = logits.argmax(dim=1).item()
        base_conf = torch.softmax(logits, dim=1)[0, pred_idx].item()

    H, W = 96, 96
    heatmap = np.zeros((H, W))

    for y in range(0, H - patch_size + 1, stride):
        for x in range(0, W - patch_size + 1, stride):
            occluded = tensor.clone()
            occluded[:, :, y:y+patch_size, x:x+patch_size] = 0  # black patch

            with torch.no_grad():
                feats  = encoder(occluded)
                logits = classifier(feats)
                conf   = torch.softmax(logits, dim=1)[0, pred_idx].item()

            drop = base_conf - conf
            heatmap[y:y+patch_size, x:x+patch_size] = max(heatmap[y:y+patch_size, x:x+patch_size].max(), drop)

    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    return heatmap, inference.CLASS_MAP[pred_idx]

# ======================
# 4. LIME (requires lime package)
# ======================
def lime_explain(image_path, num_samples=1000):
    """LIME: Local Interpretable Model-agnostic Explanations."""
    try:
        from lime import lime_image
        from skimage.segmentation import mark_boundaries
    except ImportError:
        print("[ERROR] LIME requires: pip install lime scikit-image")
        return None, None

    encoder, classifier = inference._load_models()
    encoder.eval()

    def predict_fn(images):
        """Batch prediction for LIME."""
        batch = torch.stack([
            inference.inference_transform(Image.fromarray(img.astype(np.uint8)))
            for img in images
        ]).to(DEVICE)
        with torch.no_grad():
            feats  = encoder(batch)
            logits = classifier(feats)
            probs  = torch.softmax(logits, dim=1).cpu().numpy()
        return probs

    img = Image.open(image_path).convert("RGB")
    import stain_norm
    img = stain_norm.normalize_image(img)
    img_np = np.array(img.resize((96, 96)))
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        img_np, predict_fn, top_labels=1, hide_color=0, num_samples=num_samples
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=10, hide_rest=False
    )
    return mask.astype(float), inference.CLASS_MAP[explanation.top_labels[0]]

# ======================
# 5. SHAP (requires shap package)
# ======================
def shap_explain(image_path, n_samples=100):
    """SHAP: SHapley Additive exPlanations (DeepExplainer)."""
    try:
        import shap
    except ImportError:
        print("[ERROR] SHAP requires: pip install shap")
        return None, None

    encoder, classifier = inference._load_models()
    encoder.eval()

    # Wrapper model
    class FullModel(nn.Module):
        def __init__(self, enc, clf):
            super().__init__()
            self.enc = enc
            self.clf = clf
        def forward(self, x):
            return self.clf(self.enc(x))

    model = FullModel(encoder, classifier).to(DEVICE)
    model.eval()

    # Disable in-place ReLU layers to prevent PyTorch autograd hook errors during SHAP explanation
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False

    img    = Image.open(image_path).convert("RGB")
    import stain_norm
    img    = stain_norm.normalize_image(img)
    tensor = inference.inference_transform(img).unsqueeze(0).to(DEVICE)

    # Background: random samples from training data (simplified: use zeros)
    background = torch.zeros(10, 3, 96, 96).to(DEVICE)

    explainer = shap.GradientExplainer(model, background)
    shap_values = explainer.shap_values(tensor)

    pred_idx = model(tensor).argmax(dim=1).item()
    if isinstance(shap_values, list):
        # shap_values is a list of length classes, each of shape (batch, C, H, W)
        shap_class = shap_values[pred_idx][0]
    else:
        # shap_values is an array of shape (batch, classes, C, H, W)
        shap_class = shap_values[0][pred_idx]

    saliency = np.abs(shap_class).mean(axis=0)  # (H, W)
    saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

    return saliency, inference.CLASS_MAP[pred_idx]

# ======================
# 6. VANILLA SALIENCY
# ======================
def saliency(image_path):
    """Vanilla Saliency Maps: gradients of predicted class score w.r.t input image."""
    encoder, classifier = inference._load_models()
    encoder.eval()

    img    = Image.open(image_path).convert("RGB")
    import stain_norm
    img    = stain_norm.normalize_image(img)
    tensor = inference.inference_transform(img).unsqueeze(0).to(DEVICE)
    tensor.requires_grad = True

    feats  = encoder(tensor)
    logits = classifier(feats)
    pred_idx = logits.argmax(dim=1).item()

    encoder.zero_grad()
    classifier.zero_grad()
    logits[0, pred_idx].backward()

    # Get absolute gradients w.r.t input image
    grad = tensor.grad.abs().squeeze()  # (C, H, W)
    saliency_map = grad.max(dim=0)[0].cpu().numpy()  # Max across color channels -> (H, W)
    
    # Normalize to [0, 1]
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)

    return saliency_map, inference.CLASS_MAP[pred_idx]

# ======================
# VISUALIZATION
# ======================
def visualize(image_path, heatmap, pred_class, method_name, save_path=None):
    """Overlay heatmap on original image and display."""
    img = Image.open(image_path).convert("RGB")
    import stain_norm
    img = stain_norm.normalize_image(img).resize((96, 96))
    img_np = np.array(img) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_np)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"{method_name} Heatmap")
    axes[1].axis("off")

    axes[2].imshow(img_np)
    axes[2].imshow(heatmap, cmap="jet", alpha=0.5)
    axes[2].set_title(f"Overlay (Pred: {pred_class})")
    axes[2].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[SAVE] {save_path}")
    else:
        plt.show()

# ======================
# CLI
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, required=True, help="Path to image")
    parser.add_argument("--method", type=str, default="gradcam",
                        choices=["gradcam", "smoothgrad", "occlusion", "lime", "shap", "saliency"],
                        help="Explainability method")
    parser.add_argument("--save",   type=str, help="Save path for visualization")
    args = parser.parse_args()

    print(f"[INFO] Running {args.method.upper()} on {args.image}")

    if args.method == "gradcam":
        heatmap, pred = gradcam(args.image)
    elif args.method == "smoothgrad":
        heatmap, pred = smoothgrad(args.image)
    elif args.method == "occlusion":
        heatmap, pred = occlusion(args.image)
    elif args.method == "lime":
        heatmap, pred = lime_explain(args.image)
    elif args.method == "shap":
        heatmap, pred = shap_explain(args.image)
    elif args.method == "saliency":
        heatmap, pred = saliency(args.image)

    if heatmap is not None:
        visualize(args.image, heatmap, pred, args.method.upper(), args.save)
    else:
        print(f"[ERROR] {args.method} failed.")
