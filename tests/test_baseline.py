import numpy as np
import pandas as pd
import torch

from padufes.data import CATEGORICAL_COLUMNS, NUMERIC_COLUMNS, clean_features, make_preprocessor
from padufes.model import MultimodalClassifier
from padufes.prepare import patient_disjoint_split


def test_preprocessor_handles_missing_and_unseen_values():
    train = pd.DataFrame({**{c: ["yes", None] for c in CATEGORICAL_COLUMNS},
                          **{c: [1.0, np.nan] for c in NUMERIC_COLUMNS}})
    val = train.iloc[:1].copy()
    val["region"] = "new unseen category"
    p = make_preprocessor()
    x_train = p.fit_transform(clean_features(train))
    x_val = p.transform(clean_features(val))
    assert x_train.shape[1] == x_val.shape[1]
    assert np.isfinite(x_train).all()
    assert np.isfinite(x_val).all()


def test_forward_all_modes():
    images = torch.randn(2, 3, 224, 224)
    metadata = torch.randn(2, 20)
    for mode in ("image", "metadata", "multimodal"):
        model = MultimodalClassifier(20, mode=mode, pretrained=False, freeze_backbone=False)
        model.eval()
        with torch.inference_mode():
            logits = model(images, metadata)
        assert logits.shape == (2, 6)
        assert torch.isfinite(logits).all()


def test_splits_have_no_patient_overlap():
    classes = ["ACK", "BCC", "MEL", "NEV", "SCC", "SEK"]
    rows = [
        {"patient_id": f"p_{klass}_{i}", "diagnostic": klass, "img_id": f"image_{klass}_{i}"}
        for klass in classes for i in range(20)
    ]
    train, val, test = patient_disjoint_split(pd.DataFrame(rows), seed=42)
    assert not set(train.patient_id) & set(val.patient_id)
    assert not set(train.patient_id) & set(test.patient_id)
    assert not set(val.patient_id) & set(test.patient_id)
    assert len(train) + len(val) + len(test) == len(rows)
