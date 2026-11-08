import torch
import numpy as np

import evaluate


def test_apply_dihedral_tta_shape():
    dummy = torch.randn(3, 3, 96, 96)
    tta_batch = evaluate.apply_dihedral_tta(dummy)
    assert tta_batch.shape == (24, 3, 96, 96)


def test_clinical_metric_calculations():
    # Synthetic ground truth and predictions
    # 5 Tumor (1), 5 Normal (0)
    y_true = np.array([1, 1, 1, 1, 0, 0, 0, 0, 1, 0])
    y_pred = np.array([1, 1, 1, 0, 0, 0, 0, 1, 1, 0])  # TP=4, FN=1, TN=4, FP=1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())

    assert tp == 4
    assert tn == 4
    assert fp == 1
    assert fn == 1

    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    precision = tp / (tp + fp)
    f1 = 2 * precision * sensitivity / (precision + sensitivity)

    assert accuracy == 0.8
    assert sensitivity == 0.8
    assert specificity == 0.8
    assert precision == 0.8
    assert round(f1, 4) == 0.8
