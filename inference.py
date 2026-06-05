# =============================================================================
# inference.py — Single Image & Batch Prediction
#
# Used by FastAPI backend.
# predict_single(image_path) → {class, confidence}
# predict_batch(folder_path) → {predictions, accuracy, loss}
# =============================================================================

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image
from torch.cuda.amp import autocast

# ======================
# PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
BEST_SIMCLR     = os.path.join(CKPT_DIR, "best_simclr.pth")
BEST_CLASSIFIER = os.path.join(MODELS_DIR, "best_classifier.pth")

DEVICE    = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_MAP = {0: "Normal", 1: "Tumor"}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

# ======================
# TRANSFORM
# ======================
inference_transform = T.Compose([
    T.Resize((96, 96)),
    T.ToTensor(),
    T.Normalize(mean=[0.7011, 0.5859, 0.7054],
                std=[0.2128, 0.2438, 0.1955]),
])

# ======================
# MODEL LOADING (CACHED)
# ======================
_encoder    = None
_classifier = None


def _load_models():
    global _encoder, _classifier
    if _encoder is not None and _classifier is not None:
        return _encoder, _classifier

    for p, name in [(BEST_SIMCLR, "SimCLR checkpoint"), (BEST_CLASSIFIER, "Classifier")]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"{name} not found at {p}. "
                "Run train.py then fine_tune.py to generate checkpoints."
            )

    # Encoder
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    encoder = backbone.to(DEVICE)

    simclr_ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE)
    model_state = simclr_ckpt.get('model_state', simclr_ckpt.get('model', None))
    encoder_state = {
        k.replace("encoder.", ""): v
        for k, v in model_state.items()
        if k.startswith("encoder.")
    }
    encoder.load_state_dict(encoder_state, strict=True)
    encoder.eval()

    # Classifier
    classifier = nn.Linear(512, 2).to(DEVICE)
    clf_ckpt   = torch.load(BEST_CLASSIFIER, map_location=DEVICE)
    clf_state  = clf_ckpt.get('classifier_state', clf_ckpt)
    classifier.load_state_dict(clf_state)
    classifier.eval()

    _encoder    = encoder
    _classifier = classifier
    return _encoder, _classifier


def _predict_tensor(img_tensor: torch.Tensor):
    """Run inference on a single (1, C, H, W) tensor. Returns (class_name, confidence)."""
    encoder, classifier = _load_models()
    with torch.no_grad():
        with autocast(enabled=(DEVICE == "cuda")):
            features = encoder(img_tensor)
            logits   = classifier(features)
            probs    = torch.softmax(logits, dim=1)

    predicted_idx  = probs.argmax(dim=1).item()
    confidence     = probs[0, predicted_idx].item()
    class_name     = CLASS_MAP[predicted_idx]
    return class_name, confidence

# ======================
# PUBLIC API
# ======================
def predict_single(image_path: str) -> dict:
    """
    Predict a single image.

    Args:
        image_path: Path to an image file.

    Returns:
        {"class": "Tumor"|"Normal", "confidence": float}
    """
    img = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
    class_name, confidence = _predict_tensor(tensor)
    return {"class": class_name, "confidence": round(confidence, 4)}


def predict_batch(folder_path: str) -> dict:
    """
    Predict all images in a folder.

    Optionally reads ground-truth labels from a `labels.json` file in the
    folder with format: {"filename.png": 0_or_1, ...}

    Args:
        folder_path: Path to folder containing image files.

    Returns:
        {
          "predictions": [{"file": str, "class": str, "confidence": float}, ...],
          "accuracy": float | None,
          "loss": float | None
        }
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"Not a valid directory: {folder_path}")

    image_files = [
        f for f in os.listdir(folder_path)
        if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
    ]
    if not image_files:
        raise ValueError(f"No supported images found in: {folder_path}")

    # Load optional ground-truth labels
    labels_path = os.path.join(folder_path, "labels.json")
    ground_truth = {}
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            ground_truth = json.load(f)

    predictions = []
    correct = 0
    total   = 0
    total_loss = 0.0

    criterion = nn.CrossEntropyLoss(reduction="sum")
    encoder, classifier = _load_models()

    for fname in sorted(image_files):
        fpath = os.path.join(folder_path, fname)
        try:
            img    = Image.open(fpath).convert("RGB")
            tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                with autocast(enabled=(DEVICE == "cuda")):
                    features = encoder(tensor)
                    logits   = classifier(features)
                    probs    = torch.softmax(logits, dim=1)

            idx        = probs.argmax(dim=1).item()
            confidence = probs[0, idx].item()
            class_name = CLASS_MAP[idx]

            predictions.append({
                "file":       fname,
                "class":      class_name,
                "confidence": round(confidence, 4),
            })

            # Compute loss/accuracy if label available
            if fname in ground_truth:
                gt = ground_truth[fname]
                label_tensor = torch.tensor([gt], dtype=torch.long).to(DEVICE)
                loss_val = criterion(logits, label_tensor).item()
                total_loss += loss_val
                correct    += int(idx == gt)
                total      += 1

        except Exception as e:
            predictions.append({"file": fname, "error": str(e)})

    accuracy = (correct / total) if total > 0 else None
    avg_loss = (total_loss / total) if total > 0 else None

    return {
        "predictions": predictions,
        "accuracy":    round(accuracy, 4) if accuracy is not None else None,
        "loss":        round(avg_loss, 4) if avg_loss is not None else None,
    }


# ======================
# CLI TEST
# ======================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image",  type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    args = parser.parse_args()

    if args.image:
        result = predict_single(args.image)
        print(f"Prediction: {result['class']}  ({result['confidence']*100:.1f}%)")
    elif args.folder:
        result = predict_batch(args.folder)
        print(f"Predictions: {len(result['predictions'])} images")
        if result["accuracy"] is not None:
            print(f"Accuracy: {result['accuracy']*100:.2f}%")
        if result["loss"] is not None:
            print(f"Loss: {result['loss']:.4f}")
    else:
        print("Use --image <path> or --folder <path>")


# ======================
# GRAD-CAM SALIENCY MAP
# ======================
def generate_gradcam(image_path: str, target_layer_name: str = "layer4") -> dict:
    """
    Generate Grad-CAM saliency map for an input image.

    Args:
        image_path: Path to image file
        target_layer_name: Target conv layer (default: 'layer4' for ResNet18)

    Returns:
        {"class": str, "confidence": float, "saliency_map": list[list[float]] (96x96, 0-1)}
    """
    import torch.nn.functional as F_nn
    import numpy as np

    encoder, classifier = _load_models()
    encoder.eval()

    # Storage for hooks
    activations = {}
    gradients   = {}

    def fwd_hook(module, inp, out):
        activations["feat"] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients["feat"] = grad_out[0].detach()

    # Register hooks on target layer
    target_layer = dict(encoder.named_modules()).get(target_layer_name)
    if target_layer is None:
        raise ValueError(f"Layer '{target_layer_name}' not found in encoder.")
    fh = target_layer.register_forward_hook(fwd_hook)
    bh = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img    = Image.open(image_path).convert("RGB")
        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

        # Forward
        feats  = encoder(tensor)
        logits = classifier(feats)
        probs  = torch.softmax(logits, dim=1)
        pred_idx   = probs.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()

        # Backward on predicted class
        encoder.zero_grad()
        classifier.zero_grad()
        logits[0, pred_idx].backward()

        # Grad-CAM: global average pool gradients → weight activations
        weights = gradients["feat"].mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam     = (weights * activations["feat"]).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam     = F_nn.relu(cam)

        # Upsample to input size and normalize
        cam = F_nn.interpolate(cam, size=(96, 96), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return {
            "class":       CLASS_MAP[pred_idx],
            "confidence":  round(confidence, 4),
            "saliency_map": cam.tolist(),
        }
    finally:
        fh.remove()
        bh.remove()
