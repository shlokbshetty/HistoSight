# =============================================================================
# inference.py — Single Image & Batch Prediction
#
# Used by FastAPI backend and CLI diagnostics.
# predict_single(image_path, tta=False) → {class, confidence, probabilities, tta}
# predict_batch(folder_path, tta=False) → {predictions, accuracy, loss, tta}
# =============================================================================

import os
import sys
import json
import argparse
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models

from PIL import Image

# ======================
# PATHS
# ======================
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR        = os.path.join(BASE_DIR, "checkpoints")
MODELS_DIR      = os.path.join(BASE_DIR, "models")
BEST_SIMCLR     = os.path.join(CKPT_DIR, "best_simclr.pth")
BEST_CLASSIFIER = os.path.join(MODELS_DIR, "best_classifier.pth")
BEST_ENCODER    = os.path.join(MODELS_DIR, "best_encoder.pth")

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
CLASS_MAP  = {0: "Normal", 1: "Tumor"}

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
_encoder: Optional[nn.Module] = None
_classifier: Optional[nn.Module] = None


def _load_models():
    global _encoder, _classifier
    if _encoder is not None and _classifier is not None:
        return _encoder, _classifier

    # Check existence of classifier
    if not os.path.exists(BEST_CLASSIFIER):
        raise FileNotFoundError(
            f"Classifier checkpoint not found at {BEST_CLASSIFIER}. "
            "Run fine_tune.py to generate it."
        )

    # Encoder
    backbone = models.resnet18(weights=None)
    backbone.fc = nn.Identity()
    encoder = backbone.to(DEVICE)

    # Load either fine-tuned encoder or pre-trained SimCLR encoder
    if os.path.exists(BEST_ENCODER):
        print(f"[INFO] Loading fine-tuned encoder checkpoint from {BEST_ENCODER}")
        encoder_ckpt = torch.load(BEST_ENCODER, map_location=DEVICE, weights_only=True)
        model_state = encoder_ckpt.get('model_state', encoder_ckpt.get('model', encoder_ckpt))
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
        }
        encoder.load_state_dict(encoder_state, strict=True)
    elif os.path.exists(BEST_SIMCLR):
        print(f"[INFO] Loading pre-trained SimCLR encoder from {BEST_SIMCLR}")
        simclr_ckpt = torch.load(BEST_SIMCLR, map_location=DEVICE, weights_only=True)
        model_state = simclr_ckpt.get('model_state', simclr_ckpt.get('model', None))
        encoder_state = {
            k.replace("encoder.", ""): v
            for k, v in model_state.items()
            if k.startswith("encoder.")
        }
        encoder.load_state_dict(encoder_state, strict=True)
    else:
        raise FileNotFoundError(
            f"Neither fine-tuned encoder ({BEST_ENCODER}) nor pre-trained SimCLR checkpoint ({BEST_SIMCLR}) was found. "
            "Run train.py or fine_tune.py first."
        )

    encoder.eval()

    # Classifier
    classifier = nn.Linear(512, 2).to(DEVICE)
    clf_ckpt   = torch.load(BEST_CLASSIFIER, map_location=DEVICE, weights_only=True)
    clf_state  = clf_ckpt.get('classifier_state', clf_ckpt)
    classifier.load_state_dict(clf_state)
    classifier.eval()

    _encoder    = encoder
    _classifier = classifier
    return _encoder, _classifier


def get_dihedral_tta_batch(tensor: torch.Tensor) -> torch.Tensor:
    """
    Generate the 8 dihedral transformations (D4 symmetry group: 4 rotations x 2 flips)
    for an input tensor of shape (B, C, H, W).
    Returns tensor of shape (8*B, C, H, W).
    """
    views = []
    for flip in (False, True):
        x = torch.flip(tensor, dims=[-1]) if flip else tensor
        for k in range(4):
            views.append(torch.rot90(x, k, dims=(-2, -1)))
    return torch.cat(views, dim=0)


def _predict_tensor(img_tensor: torch.Tensor, tta: bool = False) -> Tuple[str, float, torch.Tensor]:
    """
    Run inference on a single (1, C, H, W) tensor.
    If tta=True, averages softmax probabilities over the 8 dihedral transformations (D4 group).
    Returns (class_name, confidence, probs_tensor_1x2).
    """
    encoder, classifier = _load_models()
    device_type = "cuda" if DEVICE == "cuda" else "cpu"

    with torch.no_grad():
        with torch.amp.autocast(device_type=device_type, enabled=(DEVICE == "cuda")):
            if tta:
                tta_batch = get_dihedral_tta_batch(img_tensor)
                features = encoder(tta_batch)
                logits = classifier(features)
                probs_all = torch.softmax(logits, dim=1)
                probs = probs_all.mean(dim=0, keepdim=True)
            else:
                features = encoder(img_tensor)
                logits   = classifier(features)
                probs    = torch.softmax(logits, dim=1)

    predicted_idx  = probs.argmax(dim=1).item()
    confidence     = probs[0, predicted_idx].item()
    class_name     = CLASS_MAP[predicted_idx]
    return class_name, confidence, probs

# ======================
# PUBLIC API
# ======================
def predict_single(image_path: str, tta: bool = False) -> Dict[str, Any]:
    """
    Predict tumor vs. normal tissue for a single histopathology patch.

    Args:
        image_path: Path to an image file.
        tta: Whether to apply 8-fold dihedral Test-Time Augmentation (D4 symmetry).

    Returns:
        {
            "class": "Tumor" | "Normal",
            "confidence": float,
            "probabilities": {"Normal": float, "Tumor": float},
            "tta": bool
        }
    """
    img = Image.open(image_path).convert("RGB")
    import stain_norm
    img = stain_norm.normalize_image(img)
    tensor = inference_transform(img).unsqueeze(0).to(DEVICE)
    class_name, confidence, probs = _predict_tensor(tensor, tta=tta)
    return {
        "class": class_name,
        "confidence": round(confidence, 4),
        "probabilities": {
            "Normal": round(probs[0, 0].item(), 4),
            "Tumor":  round(probs[0, 1].item(), 4),
        },
        "tta": tta,
    }


def predict_batch(folder_path: str, tta: bool = False) -> Dict[str, Any]:
    """
    Predict all histopathology patches in a directory.

    Optionally reads ground-truth labels from a `labels.json` file in the
    folder with format: {"filename.png": 0_or_1, ...}

    Args:
        folder_path: Path to directory containing image files.
        tta: Whether to apply 8-fold dihedral Test-Time Augmentation.

    Returns:
        {
          "predictions": [
              {
                  "file": str,
                  "class": str,
                  "confidence": float,
                  "probabilities": {"Normal": float, "Tumor": float}
              }, ...
          ],
          "accuracy": float | None,
          "loss": float | None,
          "tta": bool
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
    device_type = "cuda" if DEVICE == "cuda" else "cpu"

    for fname in sorted(image_files):
        fpath = os.path.join(folder_path, fname)
        try:
            img    = Image.open(fpath).convert("RGB")
            import stain_norm
            img    = stain_norm.normalize_image(img)
            tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

            class_name, confidence, probs = _predict_tensor(tensor, tta=tta)

            predictions.append({
                "file":          fname,
                "class":         class_name,
                "confidence":    round(confidence, 4),
                "probabilities": {
                    "Normal": round(probs[0, 0].item(), 4),
                    "Tumor":  round(probs[0, 1].item(), 4),
                },
            })

            # Compute loss/accuracy if label available
            if fname in ground_truth:
                gt = ground_truth[fname]
                label_tensor = torch.tensor([gt], dtype=torch.long).to(DEVICE)
                # Use log-probs as surrogate logits for calibrated cross-entropy
                log_probs = torch.log(probs.clamp(min=1e-7))
                loss_val = criterion(log_probs, label_tensor).item()
                pred_idx = probs.argmax(dim=1).item()
                total_loss += loss_val
                correct    += int(pred_idx == gt)
                total      += 1

        except Exception as e:
            predictions.append({"file": fname, "error": str(e)})

    accuracy = (correct / total) if total > 0 else None
    avg_loss = (total_loss / total) if total > 0 else None

    return {
        "predictions": predictions,
        "accuracy":    round(accuracy, 4) if accuracy is not None else None,
        "loss":        round(avg_loss, 4) if avg_loss is not None else None,
        "tta":         tta,
    }


# ======================
# GRAD-CAM SALIENCY MAP
# ======================
def generate_gradcam(image_path: str, target_layer_name: str = "layer4") -> Dict[str, Any]:
    """
    Generate Grad-CAM saliency map for an input image.

    Args:
        image_path: Path to image file
        target_layer_name: Target conv layer (default: 'layer4' for ResNet18)

    Returns:
        {
            "class": str,
            "confidence": float,
            "probabilities": {"Normal": float, "Tumor": float},
            "saliency_map": list[list[float]] (96x96, 0-1)
        }
    """
    encoder, classifier = _load_models()
    encoder.eval()
    device_type = "cuda" if DEVICE == "cuda" else "cpu"

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
        import stain_norm
        img    = stain_norm.normalize_image(img)
        tensor = inference_transform(img).unsqueeze(0).to(DEVICE)

        # Forward
        with torch.amp.autocast(device_type=device_type, enabled=(DEVICE == "cuda")):
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
        cam     = F.relu(cam)

        # Upsample to input size and normalize
        cam = F.interpolate(cam, size=(96, 96), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return {
            "class":         CLASS_MAP[pred_idx],
            "confidence":    round(confidence, 4),
            "probabilities": {
                "Normal": round(probs[0, 0].item(), 4),
                "Tumor":  round(probs[0, 1].item(), 4),
            },
            "saliency_map":  cam.tolist(),
        }
    finally:
        fh.remove()
        bh.remove()


# ======================
# CLI TEST
# ======================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HistoSight Patch Inference & Diagnostic Engine")
    parser.add_argument("--image",  type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to image folder")
    parser.add_argument("--tta",    action="store_true", help="Enable 8-fold dihedral Test-Time Augmentation (D4 group)")
    args = parser.parse_args()

    if args.image:
        result = predict_single(args.image, tta=args.tta)
        tta_str = " (TTA 8x enabled)" if args.tta else ""
        print(f"Prediction: {result['class']}  ({result['confidence']*100:.1f}%){tta_str}")
        print(f"Probabilities: Normal: {result['probabilities']['Normal']*100:.1f}%, Tumor: {result['probabilities']['Tumor']*100:.1f}%")
    elif args.folder:
        result = predict_batch(args.folder, tta=args.tta)
        tta_str = " [with TTA]" if args.tta else ""
        print(f"Predictions: {len(result['predictions'])} images{tta_str}")
        if result["accuracy"] is not None:
            print(f"Accuracy: {result['accuracy']*100:.2f}%")
        if result["loss"] is not None:
            print(f"Loss: {result['loss']:.4f}")
    else:
        print("Use --image <path> or --folder <path> [--tta]")

