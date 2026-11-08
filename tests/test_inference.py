import os
import torch
import numpy as np
from PIL import Image

import inference


def test_dihedral_tta_batch_shape():
    dummy = torch.randn(2, 3, 96, 96)
    tta_batch = inference.get_dihedral_tta_batch(dummy)
    assert tta_batch.shape == (16, 3, 96, 96)


def test_predict_single_output_structure():
    test_img_path = os.path.join(
        inference.BASE_DIR,
        "pcam_test_100_per_class",
        "pcam_test_100_per_class",
        "1_tumor",
        "test_patch_1.png"
    )
    if not os.path.exists(test_img_path):
        return

    res = inference.predict_single(test_img_path, tta=False)
    assert "class" in res
    assert res["class"] in {"Tumor", "Normal"}
    assert 0.0 <= res["confidence"] <= 1.0
    assert "probabilities" in res
    assert "Normal" in res["probabilities"]
    assert "Tumor" in res["probabilities"]
    assert round(res["probabilities"]["Normal"] + res["probabilities"]["Tumor"], 2) == 1.0
    assert res["tta"] is False


def test_predict_single_with_tta():
    test_img_path = os.path.join(
        inference.BASE_DIR,
        "pcam_test_100_per_class",
        "pcam_test_100_per_class",
        "1_tumor",
        "test_patch_1.png"
    )
    if not os.path.exists(test_img_path):
        return

    res_tta = inference.predict_single(test_img_path, tta=True)
    assert res_tta["tta"] is True
    assert res_tta["class"] in {"Tumor", "Normal"}
    assert 0.0 <= res_tta["confidence"] <= 1.0


def test_gradcam_output_shape():
    test_img_path = os.path.join(
        inference.BASE_DIR,
        "pcam_test_100_per_class",
        "pcam_test_100_per_class",
        "1_tumor",
        "test_patch_1.png"
    )
    if not os.path.exists(test_img_path):
        return

    gradcam_res = inference.generate_gradcam(test_img_path)
    assert "saliency_map" in gradcam_res
    saliency = np.array(gradcam_res["saliency_map"])
    assert saliency.shape == (96, 96)
    assert saliency.min() >= 0.0
    assert saliency.max() <= 1.0

