import os
import h5py
import numpy as np
from tqdm import tqdm
import stain_norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "simclr", "ssl_splits")

# Ensure template is extracted and normalizer is fitted
print("[PREPROCESS] Setting up normalizer...")
stain_norm.get_normalizer()

files_to_normalize = [
    ("pcam_ssl_val_labeled_x.h5", "pcam_ssl_val_labeled_x_normalized.h5"),
    ("pcam_ssl_test_labeled_x.h5", "pcam_ssl_test_labeled_x_normalized.h5"),
    ("pcam_ssl_train_unlabeled_x.h5", "pcam_ssl_train_unlabeled_x_normalized.h5")
]

def normalize_h5_file(src_name, dest_name, chunk_size=1000):
    src_path = os.path.join(DATA_DIR, src_name)
    dest_path = os.path.join(DATA_DIR, dest_name)

    if not os.path.exists(src_path):
        print(f"[PREPROCESS] Source file {src_name} not found. Skipping.")
        return

    if os.path.exists(dest_path):
        print(f"[PREPROCESS] Destination file {dest_name} already exists. Skipping.")
        return

    print(f"\n[PREPROCESS] Normalizing {src_name} -> {dest_name}...")
    
    with h5py.File(src_path, 'r') as f_src:
        x_src = f_src['x']
        total_samples = len(x_src)
        shape = x_src.shape
        dtype = x_src.dtype

        # Create destination H5 file with chunked storage for optimization
        with h5py.File(dest_path, 'w') as f_dest:
            x_dest = f_dest.create_dataset(
                'x', 
                shape=shape, 
                dtype=dtype, 
                chunks=(chunk_size, shape[1], shape[2], shape[3]) if total_samples >= chunk_size else None
            )

            # Process in chunks
            for i in tqdm(range(0, total_samples, chunk_size), desc=f"Normalizing {src_name}"):
                end_idx = min(i + chunk_size, total_samples)
                # Load chunk to RAM
                chunk = x_src[i:end_idx]
                
                # Normalize chunk on CPU
                normalized_chunk = np.empty_like(chunk)
                for idx in range(len(chunk)):
                    normalized_chunk[idx] = stain_norm.normalize_numpy(chunk[idx])
                
                # Write to destination
                x_dest[i:end_idx] = normalized_chunk

    print(f"[PREPROCESS] Successfully normalized {src_name} and saved to {dest_name}")

def main():
    for src_name, dest_name in files_to_normalize:
        try:
            normalize_h5_file(src_name, dest_name)
        except Exception as e:
            print(f"[PREPROCESS] Error processing {src_name}: {e}")

if __name__ == "__main__":
    main()
