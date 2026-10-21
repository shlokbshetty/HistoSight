import os
import sys
import glob

# Ensure parent directory imports work
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

import explainability

def get_test_images():
    """Find a few test images from the dataset folder."""
    test_root = os.path.join(BASE_DIR, "pcam_test_100_per_class", "pcam_test_100_per_class")
    normal_imgs = glob.glob(os.path.join(test_root, "0_normal", "*.png"))
    tumor_imgs = glob.glob(os.path.join(test_root, "1_tumor", "*.png"))
    
    selected = []
    if normal_imgs:
        selected.append(("normal_sample", normal_imgs[0]))
    if tumor_imgs:
        selected.append(("tumor_sample", tumor_imgs[0]))
        
    return selected

def main():
    # Create the test output folder
    output_dir = os.path.join(BASE_DIR, "test")
    os.makedirs(output_dir, exist_ok=True)
    
    test_images = get_test_images()
    if not test_images:
        print("[ERROR] No test images found in pcam_test_100_per_class/pcam_test_100_per_class")
        return
        
    methods = {
        "gradcam": explainability.gradcam,
        "saliency": explainability.saliency,
        "smoothgrad": explainability.smoothgrad,
        "occlusion": explainability.occlusion,
        "lime": explainability.lime_explain,
        "shap": explainability.shap_explain
    }
    
    for name, img_path in test_images:
        print(f"\n[INFO] Processing image: {name} ({os.path.basename(img_path)})")
        for method_name, method_fn in methods.items():
            print(f"  -> Running {method_name.upper()}...")
            try:
                heatmap, pred_class = method_fn(img_path)
                if heatmap is not None:
                    save_path = os.path.join(output_dir, f"{name}_{method_name}.png")
                    explainability.visualize(
                        image_path=img_path,
                        heatmap=heatmap,
                        pred_class=pred_class,
                        method_name=method_name.upper(),
                        save_path=save_path
                    )
                    print(f"  [SUCCESS] Saved: {save_path}")
                else:
                    print(f"  [ERROR] {method_name} returned None heatmap.")
            except Exception as e:
                print(f"  [EXCEPTION] Failed running {method_name}: {e}")

if __name__ == "__main__":
    main()
