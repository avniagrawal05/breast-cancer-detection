# Requirements

## Prerequisites

- Python 3.8+ recommended.
- GPU available for training (CUDA/cuDNN) recommended but not required for small experiments.

## Environment Setup

Create and activate a virtual environment and install dependencies.

### Using venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Using conda:

```bash
conda create -n breast-cancer python=3.9
conda activate breast-cancer
pip install -r requirements.txt
```

## Dataset

- Expect input images in a top-level `data/` directory, organized as `data/train/<class>/...` and `data/val/<class>/...` or a CSV that maps image paths to labels.
- Document the dataset source and licensing here when available. If using public datasets (e.g., DDSM, INbreast), add download/prep steps.

## Preprocessing

- Resize images to a consistent resolution (e.g., 224x224 or 512x512 depending on model).
- Normalize pixel values (ImageNet mean/std for transfer learning or 0-1 scaling for custom models).
- Optional augmentations: random rotations, flips, brightness/contrast jitter, random crops.

## Folder Structure (recommended)

- `data/` — raw and processed data
- `notebooks/` — exploratory notebooks and demo runs
- `scripts/` — utilities: data preparation, augmentation, and helpers
- `models/` — model definitions
- `configs/` — YAML/JSON configs for experiments
- `runs/` — training runs, checkpoints, logs
- `requirements.txt` — Python dependencies
