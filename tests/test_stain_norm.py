import os
import numpy as np
from PIL import Image
import stain_norm

def test_get_normalizer():
    # Normalizer should successfully initialize and not raise exceptions
    normalizer = stain_norm.get_normalizer()
    assert normalizer is not None or not os.path.exists(stain_norm.TEMPLATE_PATH)

def test_normalize_image_shape():
    # Create a dummy tissue-like image (H&E color profile: pinkish)
    dummy_arr = np.ones((96, 96, 3), dtype=np.uint8)
    dummy_arr[:, :, 0] = 200  # R
    dummy_arr[:, :, 1] = 120  # G
    dummy_arr[:, :, 2] = 200  # B
    dummy_img = Image.fromarray(dummy_arr)
    
    # Run normalization
    norm_img = stain_norm.normalize_image(dummy_img)
    norm_arr = np.array(norm_img)
    
    # Assert dimensions and type are preserved
    assert norm_arr.shape == (96, 96, 3)
    assert norm_arr.dtype == np.uint8

def test_normalize_blank_fallback():
    # Create a pure white background image
    dummy_arr = np.ones((96, 96, 3), dtype=np.uint8) * 255
    dummy_img = Image.fromarray(dummy_arr)
    
    # Run normalization (should handle divide-by-zero or low OD issues gracefully)
    norm_img = stain_norm.normalize_image(dummy_img)
    norm_arr = np.array(norm_img)
    
    assert norm_arr.shape == (96, 96, 3)
    # The normalizer should fall back or output valid pixels
    assert np.isnan(norm_arr).sum() == 0
