"""Fit PAD-UFES-20 classification baseline. Uses only train data to fit preprocessing."""

import argparse
import csv
import random
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.padufes.data import (
    CLASSES, LABEL_TO_INDEX, SkinLesionDataset, build_image_index,
    clean_features, load_metadata, make_preprocessor,
)
from src.padufes.model import MultimodalClassifier


def device_from_arg(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def evaluate_loader(model, loader, device):
    model.eval()
    predicted, actual = [], []
    with torch.inference_mode():
        for image, metadata, label in loader:
            logits = model(image.to(device), metadata.to(device))
            predicted.extend(logits.argmax(1).cpu().tolist())
            actual.extend(label.tolist())
    return {
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(actual, predicted, labels=range(len(CLASSES)), average="macro", zero_division=0),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pad_ufes_20"))
    parser.add_argument("--out-dir", type=Path, default=Path("runs/first_run"))
    parser.add_argument("--mode", choices=("image", "metadata", "multimodal"), default="multimodal")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-pretrained", action="store_true", help="ResNet18 trained from scratch, no download")
    parser.add_argument("--finetune-backbone", action="store_true", help="Update ResNet18 weights; costs more memory")
    parser.add_argument("--weighted-loss", action="store_true", help="Inverse-frequency class-weighted loss")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if args.mode != "metadata" and args.no_pretrained and not args.finetune_backbone:
        parser.error("--no-pretrained requires --finetune-backbone (cannot freeze random weights)")
    device = device_from_arg(args.device)
    print(f"Device: {device}; mode: {args.mode}")

    train_df = load_metadata(args.data_dir / "splits/train.csv")
    val_df = load_metadata(args.data_dir / "splits/val.csv")
    if set(train_df.patient_id) & set(val_df.patient_id):
        raise ValueError("Train/val patient overlap detected")
    image_index = build_image_index(args.data_dir / "images")

    preprocessor = make_preprocessor()
    train_features = preprocessor.fit_transform(clean_features(train_df))
    val_features = preprocessor.transform(clean_features(val_df))
    tab_dim = train_features.shape[1]
    args.out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, args.out_dir / "metadata_preprocessor.joblib")

    train_ds = SkinLesionDataset(train_df, image_index, train_features, training=True)
    val_ds = SkinLesionDataset(val_df, image_index, val_features, training=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultimodalClassifier(
        metadata_dim=tab_dim, num_classes=len(CLASSES), mode=args.mode,
        pretrained=(not args.no_pretrained) if args.mode != "metadata" else False,
        freeze_backbone=(not args.finetune_backbone),
    ).to(device)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    weights = None
    if args.weighted_loss:
        counts = np.bincount([LABEL_TO_INDEX[x] for x in train_df.diagnostic], minlength=len(CLASSES))
        if np.any(counts == 0):
            raise ValueError("At least one class missing in train set")
        weights = torch.tensor(len(train_df) / (len(CLASSES) * counts), dtype=torch.float32, device=device)
        print(f"Class weights: {dict(zip(CLASSES, weights.cpu().tolist()))}")
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    history_path = args.out_dir / "history.csv"
    best_score = -1.0
    with history_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("epoch", "train_loss", "val_accuracy", "val_macro_f1"))
        writer.writeheader()
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss, n_items = 0.0, 0
            for image, metadata, label in train_loader:
                image, metadata, label = image.to(device), metadata.to(device), label.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(image, metadata)
                loss = loss_fn(logits, label)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(label)
                n_items += len(label)
            metrics = evaluate_loader(model, val_loader, device)
            record = {
                "epoch": epoch,
                "train_loss": round(total_loss / n_items, 5),
                "val_accuracy": round(metrics["accuracy"], 5),
                "val_macro_f1": round(metrics["macro_f1"], 5),
            }
            writer.writerow(record)
            f.flush()
            print(record)
            if metrics["macro_f1"] > best_score:
                best_score = metrics["macro_f1"]
                torch.save({
                    "model_state": model.state_dict(),
                    "mode": args.mode,
                    "metadata_dim": tab_dim,
                    "classes": list(CLASSES),
                    "epoch": epoch,
                    "val_macro_f1": best_score,
                }, args.out_dir / "best_model.pt")
    print(f"Best val macro-F1: {best_score:.4f}; saved artifacts in {args.out_dir}")


if __name__ == "__main__":
    main()
