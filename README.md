# PAD-UFES-20 image + metadata baseline

Minimal, runnable PyTorch starter for **The Value of a Second Modality**.

It trains the same six-class classifier in three modes: `image`, `metadata`, and `multimodal`. The point is to build trustworthy, comparable baselines before adding deep ensembles or uncertainty estimation.

## Architecture

```text
RGB clinical photo (3 x 224 x 224)       21 original metadata fields
              |                                     |
    pretrained ResNet18                         preprocessing
      (512 features)                   numeric: median + standardize
              |                        categorical: one-hot; missing
          Linear 128                               |
              |                               Linear 64
              +----------------+--------------------+
                               |
                       concatenate (192)
                               |
                      Linear 128 + ReLU
                               |
                        Linear 6 (logits)
```

The same code supports image-only and metadata-only ablations. Default backbone is **frozen and pretrained**; only the projection and classifier layers train. Run `--finetune-backbone --lr 0.0001` to fine-tune all layers later.

**Important safeguards**: `diagnostic` is the label. `patient_id`, `lesion_id`, and `img_id` are identifiers, not features. `biopsed` is deliberately **excluded** because it encodes diagnostic verification and could leak the label. Metadata preprocessing is fitted **only on training patients**. Images from the same patient never cross data partitions.

## Repository

```text
padufes_multimodal_starter/
├── pyproject.toml
├── README.md
├── .gitignore
├── src/padufes/
│   ├── __init__.py
│   ├── data.py       # schema, metadata preprocessing, image dataset
│   ├── prepare.py    # patient-disjoint train/val/test CSVs
│   ├── model.py      # ResNet18, metadata MLP, late fusion
│   ├── train.py      # training, checkpoint, validation metrics
│   └── evaluate.py   # test metrics and probability predictions
└── tests/
    └── test_baseline.py
```

## 1. Set up on Ubuntu / ADLM cluster

From the repository root, create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For GPU/cluster installations, ensure `torch` and `torchvision` are installed with the correct CUDA build, following the cluster's recommended instructions. The default pretrained ResNet18 weights download on the first image-based training run, so ensure internet access or pre-cache the weights. `metadata`-only mode does not need pretrained weights.

## 2. Place the original PAD-UFES-20 data

Download the dataset separately from https://data.mendeley.com/datasets/zr7vgbcyr2/1. Extract the images into nested subfolders as needed:

```text
data/
└── pad_ufes_20/
    ├── metadata.csv
    └── images/
        ├── imgs_part_1/
        │   └── PAT_...png
        ├── imgs_part_2/
        └── imgs_part_3/
```

`data/` is excluded by `.gitignore`, so neither images nor patient metadata are committed. The code searches recursively under `images/` and matches filenames by `img_id`.

## 3. Create patient-safe splits

```bash
python -m padufes.prepare --data-dir data/pad_ufes_20 --seed 42
```

Creates `splits/train.csv`, `splits/val.csv`, and `splits/test.csv`, approximately 70/15/15 by **patients**. Every label must be represented in every partition. Group separation matters because PAD-UFES-20 contains multiple photos for some lesions and patients. These splits are **group-shuffled, not optimally stratified**, so inspect the printed class distribution before running comparisons. Reuse the same split files across all experiment modes.

## 4. Train the multimodal baseline

```bash
python -m padufes.train \
    --data-dir data/pad_ufes_20 \
    --out-dir runs/multimodal \
    --mode multimodal \
    --epochs 10 --batch-size 32 --lr 0.001
```

To address PAD-UFES-20's class imbalance, try adding `--weighted-loss` and compare **validation macro-F1**. Treat it as an experiment, not a guaranteed improvement.

Train image-only on the **same data splits** for the research question:

```bash
python -m padufes.train \
    --data-dir data/pad_ufes_20 \
    --out-dir runs/image \
    --mode image \
    --epochs 10 --batch-size 32 --lr 0.001
```

Optional metadata-only diagnostic baseline:

```bash
python -m padufes.train \
    --data-dir data/pad_ufes_20 \
    --out-dir runs/metadata \
    --mode metadata \
    --epochs 10 --batch-size 32 --lr 0.001
```

Without a network connection to fetch ResNet18 weights you can run an image-based training **sanity check** with `--no-pretrained --finetune-backbone`; this will not be comparable to the pretrained baseline.

## 5. Evaluate on held-out test patients

Do this only after deciding on the final model/hyperparameters using validation:

```bash
python -m padufes.evaluate --data-dir data/pad_ufes_20 --run-dir runs/multimodal
python -m padufes.evaluate --data-dir data/pad_ufes_20 --run-dir runs/image
```

Each run folder includes:

- `best_model.pt`: best validation macro-F1 checkpoint
- `metadata_preprocessor.joblib`: preprocessing fitted on training rows
- `history.csv`: training loss and validation metrics per epoch
- `test_metrics.json`: class-level precision/recall/F1, macro-F1, balanced accuracy, confusion matrix
- `test_predictions.csv`: IDs, labels and per-class softmax probabilities for downstream analyses

Softmax scores are **not calibrated epistemic uncertainty estimates**. Later, replace a single trained model with an ensemble of independently initialized models and aggregate predicted probabilities. Compare the image-only and multimodal ensembles on exactly the same patients.

## 6. Run tests

```bash
pytest -q
```

## Next experiments (not implemented yet)

1. Stronger patient-group and label-stratified split selection, or grouped cross-validation; multiple random seeds and bootstrapped confidence intervals by patient.
2. Class imbalance handling and rare-class (`MEL`) F1/recall; calibration (NLL, ECE, Brier score).
3. A **deep ensemble** (e.g. five independently trained seeds per mode) with predictive entropy, expected entropy, and mutual information for approximate epistemic uncertainty.
4. Missing/conflicting metadata perturbations, image corruption, subgroup analyses, and a clinically motivated metadata subset that excludes socioeconomic/ancestry covariates.

## Research and use limitations

PAD-UFES-20 is small, class imbalanced, region-specific, and includes labels established in different ways. The starter reports **per-image**, not per-patient, metrics. Results are for academic research, **not clinical use**. Cite the source dataset and paper when reporting results: Pacheco et al., *PAD-UFES-20: A skin lesion dataset composed of patient data and clinical images collected from smartphones*, Data in Brief (2020), DOI: 10.1016/j.dib.2020.106221. Dataset: https://doi.org/10.17632/zr7vgbcyr2.1 (CC BY 4.0).
