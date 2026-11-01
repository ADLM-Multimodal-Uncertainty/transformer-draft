"""Evaluate the frozen best checkpoint once on the held-out test patients."""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, classification_report,
    confusion_matrix, f1_score,
)
from torch.utils.data import DataLoader

from src.padufes.data import CLASSES, SkinLesionDataset, build_image_index, clean_features, load_metadata
from src.padufes.model import MultimodalClassifier
from src.padufes.train import device_from_arg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pad_ufes_20"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/first_run"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = device_from_arg(args.device)
    # Only open artifacts you produced/trust. joblib files can execute code when loaded.
    ckpt = torch.load(args.run_dir / "best_model.pt", map_location="cpu", weights_only=True)
    if tuple(ckpt["classes"]) != CLASSES:
        raise ValueError("Checkpoint labels differ from configured labels")
    preprocessor = joblib.load(args.run_dir / "metadata_preprocessor.joblib")
    test_df = load_metadata(args.data_dir / "splits/test.csv")
    test_features = preprocessor.transform(clean_features(test_df))
    index = build_image_index(args.data_dir / "images")
    ds = SkinLesionDataset(test_df, index, test_features)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MultimodalClassifier(
        metadata_dim=ckpt["metadata_dim"], num_classes=len(CLASSES),
        mode=ckpt["mode"], pretrained=False, freeze_backbone=False,
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    actual, probabilities = [], []
    with torch.inference_mode():
        for image, metadata, label in loader:
            logits = model(image.to(device), metadata.to(device))
            probabilities.append(logits.softmax(dim=1).cpu().numpy())
            actual.extend(label.tolist())
    probabilities = np.concatenate(probabilities)
    predictions = probabilities.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    metrics = {
        "accuracy": accuracy_score(actual, predictions),
        "balanced_accuracy": balanced_accuracy_score(actual, predictions),
        "macro_f1": f1_score(actual, predictions, labels=labels, average="macro", zero_division=0),
        "classification_report": classification_report(
            actual, predictions, labels=labels, target_names=CLASSES,
            zero_division=0, output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(actual, predictions, labels=labels).tolist(),
        "class_order": list(CLASSES),
        "n_images": len(actual),
        "n_patients": int(test_df.patient_id.nunique()),
    }
    print(classification_report(actual, predictions, labels=labels, target_names=CLASSES, zero_division=0))
    print(f"Macro F1: {metrics['macro_f1']:.3f}; accuracy: {metrics['accuracy']:.3f}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    result = test_df[["patient_id", "img_id", "diagnostic"]].copy().reset_index(drop=True)
    result["predicted"] = [CLASSES[i] for i in predictions]
    for i, name in enumerate(CLASSES):
        result[f"p_{name}"] = probabilities[:, i]
    result.to_csv(args.run_dir / "test_predictions.csv", index=False)
    print(f"Saved test_metrics.json and test_predictions.csv in {args.run_dir}")


if __name__ == "__main__":
    main()
