"""PAD-UFES-20 columns, preprocessing, image loading, and metrics-safe features."""

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset
from torchvision import transforms

# The paper's six diagnostic labels. This ordering is fixed across all runs.
CLASSES = ("ACK", "BCC", "MEL", "NEV", "SCC", "SEK")
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(CLASSES)}

NUMERIC_COLUMNS = ("age", "diameter_1", "diameter_2")
CATEGORICAL_COLUMNS = (
    "smoke", "drink", "background_father", "background_mother", "pesticide",
    "gender", "skin_cancer_history", "cancer_history", "has_piped_water",
    "has_sewage_system", "fitspatrick", "region", "itch", "grew",
    "hurt", "changed", "bleed", "elevation",
)
FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS

# Explicitly NOT features: patient_id, lesion_id, img_id, diagnostic, biopsed.
# 'biopsed' reflects how the ground truth was obtained and risks label leakage.


def load_metadata(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"patient_id": str, "img_id": str})
    df.columns = df.columns.str.strip()
    mandatory = {"patient_id", "img_id", "diagnostic", *FEATURE_COLUMNS}
    missing = mandatory.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected CSV columns: {sorted(missing)}")
    df["diagnostic"] = df["diagnostic"].astype(str).str.strip().str.upper()
    unknown = set(df["diagnostic"]) - set(CLASSES)
    if unknown:
        raise ValueError(f"Unexpected diagnostic labels: {sorted(unknown)}")
    if df["patient_id"].isna().any() or df["img_id"].isna().any():
        raise ValueError("Missing patient_id or img_id")
    if df["img_id"].duplicated().any():
        raise ValueError("Duplicate img_id rows in metadata")
    df["img_id"] = df["img_id"].str.removesuffix(".png")
    return df


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """Make numerical/strings consistent before sklearn; never use labels or IDs."""
    x = df.loc[:, FEATURE_COLUMNS].copy()
    for col in NUMERIC_COLUMNS:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    for col in CATEGORICAL_COLUMNS:
        # String conversion makes e.g. true/false/0/1 consistently categorical.
        x[col] = x[col].where(x[col].notna(), "Missing").astype(str)
    return x


def make_preprocessor() -> ColumnTransformer:
    """Fit on TRAIN ONLY. Missing values and previously unseen categories are supported."""
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer([
        ("num", numeric, list(NUMERIC_COLUMNS)),
        ("cat", categorical, list(CATEGORICAL_COLUMNS)),
    ], sparse_threshold=0)


def build_image_index(images_dir: str | Path) -> dict[str, Path]:
    """Find files under nested imgs_part_* directories by img_id stem."""
    images_dir = Path(images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Image folder not found: {images_dir}")
    mapping = {}
    for file in images_dir.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg"}:
            if file.stem in mapping:
                raise ValueError(f"Duplicate image stem {file.stem} under {images_dir}")
            mapping[file.stem] = file
    if not mapping:
        raise FileNotFoundError(f"No image files found under {images_dir}")
    return mapping


def image_transforms(training: bool):
    ops = [transforms.Resize((224, 224))]
    if training:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
    return transforms.Compose(ops)


class SkinLesionDataset(Dataset):
    def __init__(self, df, image_index, transformed_metadata, training=False):
        self.df = df.reset_index(drop=True)
        self.image_index = image_index
        self.metadata = np.asarray(transformed_metadata, dtype=np.float32)
        self.transform = image_transforms(training)
        if len(self.df) != len(self.metadata):
            raise ValueError("Rows in metadata and transformed features differ")
        missing = [str(k) for k in self.df["img_id"] if str(k) not in image_index]
        if missing:
            raise FileNotFoundError(
                f"Missing {len(missing)} image(s), first examples: {missing[:5]}"
            )
        self.labels = np.array([LABEL_TO_INDEX[x] for x in self.df["diagnostic"]], dtype=np.int64)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch

        path = self.image_index[str(self.df.iloc[idx]["img_id"])]
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img).convert("RGB")
            image = self.transform(img)
        metadata = torch.from_numpy(self.metadata[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, metadata, label
