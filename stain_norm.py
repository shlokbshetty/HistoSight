import os
import numpy as np
import h5py
from PIL import Image
import torchstain

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
DATA_DIR = os.path.join(BASE_DIR, "simclr", "ssl_splits")
TEMPLATE_PATH = os.path.join(CKPT_DIR, "template.png")

# Global normalizer cache
_normalizer = None

def get_normalizer():
    global _normalizer
    if _normalizer is not None:
        return _normalizer

    os.makedirs(CKPT_DIR, exist_ok=True)

    # 1. Load or extract reference template
    if not os.path.exists(TEMPLATE_PATH):
        print("[STAIN_NORM] Template image not found. Attempting to extract from dataset...")
        val_x_path = os.path.join(DATA_DIR, "pcam_ssl_val_labeled_x.h5")
        if os.path.exists(val_x_path):
            try:
                with h5py.File(val_x_path, 'r') as f:
                    # Choose a good representative patch. Index 0 is fine, or look for a non-blank one.
                    # Let's check a few indices and pick one that has a reasonable mean (not too white/blank)
                    best_img_arr = None
                    for idx in range(10):
                        arr = f['x'][idx]
                        mean_val = arr.mean()
                        # H&E images typically have background mean around 200-240.
                        # Tissues have lower means (e.g. 100-180). Let's pick one with mean < 200.
                        if mean_val < 200:
                            best_img_arr = arr
                            break
                    if best_img_arr is None:
                        best_img_arr = f['x'][0]
                
                img = Image.fromarray(best_img_arr)
                img.save(TEMPLATE_PATH)
                print(f"[STAIN_NORM] Extracted and saved reference template to: {TEMPLATE_PATH}")
            except Exception as e:
                print(f"[STAIN_NORM] Warning: Could not extract template from dataset: {e}")
        else:
            print(f"[STAIN_NORM] Warning: Dataset not found at {val_x_path}. Cannot automatically extract template.")

    # 2. Fit the normalizer
    # We use the 'numpy' backend for maximum robustness and stability in all environments
    normalizer = torchstain.normalizers.MacenkoNormalizer(backend='numpy')

    if os.path.exists(TEMPLATE_PATH):
        try:
            target_img = Image.open(TEMPLATE_PATH).convert("RGB")
            target_arr = np.array(target_img)
            normalizer.fit(target_arr)
            print("[STAIN_NORM] Macenko Normalizer successfully fitted with template.")
            _normalizer = normalizer
        except Exception as e:
            print(f"[STAIN_NORM] Error fitting normalizer with template: {e}")
            _normalizer = None
    else:
        print("[STAIN_NORM] Warning: Normalizer NOT fitted (no template available). Images will not be normalized.")
        _normalizer = None

    return _normalizer

def normalize_numpy(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a numpy image array (H, W, 3) in range [0, 255] using Macenko.
    Returns normalized numpy array (H, W, 3) in range [0, 255], uint8.
    """
    normalizer = get_normalizer()
    if normalizer is None:
        return arr

    try:
        # torchstain expects a numpy array of shape (H, W, 3) in [0, 255]
        norm, H, E = normalizer.normalize(I=arr, stains=True)
        # Ensure correct type and bounds
        norm = np.clip(norm, 0, 255).astype(np.uint8)
        return norm
    except Exception as e:
        # SVD can fail on pure background or corrupted images, return original as fallback
        return arr

def normalize_image(img: Image.Image) -> Image.Image:
    """
    Normalize a PIL Image. Returns normalized PIL Image.
    """
    arr = np.array(img.convert("RGB"))
    norm_arr = normalize_numpy(arr)
    return Image.fromarray(norm_arr)
