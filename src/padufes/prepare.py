"""Generate patient-disjoint train/val/test CSVs from the original metadata."""

import argparse
from pathlib import Path

from sklearn.model_selection import GroupShuffleSplit

from src.padufes.data import CLASSES, load_metadata


def patient_disjoint_split(df, seed: int = 42, attempts: int = 100):
    """Approx. 70/15/15 by patients; require every class to occur in every split."""
    required = set(CLASSES)
    for attempt in range(attempts):
        split_seed = seed + attempt
        first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=split_seed)
        train_val_idx, test_idx = next(first.split(df, groups=df["patient_id"]))
        train_val = df.iloc[train_val_idx]
        second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=split_seed)
        train_sub, val_sub = next(second.split(train_val, groups=train_val["patient_id"]))
        train_df = train_val.iloc[train_sub].copy()
        val_df = train_val.iloc[val_sub].copy()
        test_df = df.iloc[test_idx].copy()
        parts = (train_df, val_df, test_df)
        if all(set(part["diagnostic"]) == required for part in parts):
            assert not (set(train_df.patient_id) & set(val_df.patient_id))
            assert not (set(train_df.patient_id) & set(test_df.patient_id))
            assert not (set(val_df.patient_id) & set(test_df.patient_id))
            return parts
    raise RuntimeError(
        f"Could not make a patient-disjoint split with all {len(CLASSES)} labels "
        f"in each partition after {attempts} attempts. Inspect class counts per patient."
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/pad_ufes_20"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    df = load_metadata(args.data_dir / "metadata.csv")
    splits = patient_disjoint_split(df, args.seed)
    output = args.data_dir / "splits"
    output.mkdir(parents=True, exist_ok=True)
    for name, part in zip(("train", "val", "test"), splits):
        part.to_csv(output / f"{name}.csv", index=False)
        counts = part.diagnostic.value_counts().reindex(CLASSES, fill_value=0)
        print(f"{name}: {len(part)} images, {part.patient_id.nunique()} patients; labels={counts.to_dict()}")
    print(f"Saved splits to: {output}")


if __name__ == "__main__":
    main()
