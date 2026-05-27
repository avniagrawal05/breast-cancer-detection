# breast-cancer-detection

## Project Overview

This repository contains code and notebooks for a breast cancer detection pipeline using medical images. The goal is to build a reproducible training and inference workflow that includes data preparation, model training, validation, and evaluation.

## Idea Depiction

- **Problem:** Detect presence/absence (or severity) of breast cancer from imaging data.
- **Goal:** Provide a clear, reproducible pipeline that trains a model on labeled images, evaluates it on hold-out data, and produces inference-ready artifacts.
- **Approach:** Preprocess images, augment and split dataset, train a CNN-based classifier (or transfer learning with ResNet/EfficientNet), validate with cross-validation, and report metrics (accuracy, AUC, sensitivity, specificity, confusion matrix).
- **Model / Architecture:** Start with transfer learning (e.g., EfficientNet/ResNet) and optionally iterate to lightweight or ensemble models depending on accuracy/latency tradeoffs.

## Dataset

- Expect input images in a top-level `data/` directory, organized as `data/train/<class>/...` and `data/val/<class>/...` or a CSV that maps image paths to labels.
- Document the dataset source and licensing here when available. If using public datasets (e.g., DDSM, INbreast), add download/prep steps.

## Preprocessing

- Resize images to a consistent resolution (e.g., 224x224 or 512x512 depending on model).
- Normalize pixel values (ImageNet mean/std for transfer learning or 0-1 scaling for custom models).
- Optional augmentations: random rotations, flips, brightness/contrast jitter, random crops.

## Code Execution Plan (Step-by-step)

1. Prerequisites
	- Python 3.8+ recommended.
	- GPU available for training (CUDA/cuDNN) recommended but not required for small experiments.

2. Environment setup
	- Create and activate a virtual environment (venv/conda) and install dependencies.

	Example (venv):

	```bash
	python -m venv .venv
	source .venv/bin/activate
	pip install -r requirements.txt
	```

3. Data preparation
	- Place images under `data/` or provide a CSV mapping file at `data/labels.csv`.
	- Run the data-prep script/notebook to split into `train/`, `val/`, and `test/` folders and to generate any preprocessing caches.

	Example:

	```bash
	python scripts/prepare_data.py --input data/raw --output data/processed --split 0.7 0.15 0.15
	```

4. Training
	- Configure training parameters (batch size, learning rate, number of epochs) in `configs/train.yaml` or via CLI args.
	- Start training and log checkpoints and metrics to `runs/` (or a logger like TensorBoard/Weights & Biases).

	Example:

	```bash
	python train.py --config configs/train.yaml
	# or
	python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/train.yaml
	```

5. Evaluation
	- Run the evaluation script on the `test/` set to compute metrics and save plots.

	Example:

	```bash
	python evaluate.py --checkpoint runs/exp1/checkpoint.pt --data data/processed/test
	```

6. Inference
	- Use `inference.py` or the provided notebook to run the model on new images and output predictions/visualizations (e.g., Grad-CAM).

	Example:

	```bash
	python inference.py --checkpoint runs/exp1/checkpoint.pt --input images/sample.jpg --output results/sample_pred.json
	```

7. Reproducibility
	- Save the training seed, environment (via `pip freeze > requirements.txt`), and configuration files alongside model checkpoints for reproducibility.

## Folder Structure (recommended)

- `data/` — raw and processed data
- `notebooks/` — exploratory notebooks and demo runs
- `scripts/` — utilities: data preparation, augmentation, and helpers
- `models/` — model definitions
- `configs/` — YAML/JSON configs for experiments
- `runs/` — training runs, checkpoints, logs
- `requirements.txt` — Python dependencies
- `README.md` — this file

## Quick Start (commands)

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Prepare data (example):

```bash
python scripts/prepare_data.py --input data/raw --output data/processed
```

3. Train model:

```bash
python train.py --config configs/train.yaml
```

4. Evaluate:

```bash
python evaluate.py --checkpoint runs/exp1/checkpoint.pt --data data/processed/test
```

## Metrics and Reporting

- Track loss/accuracy per epoch, AUC, precision/recall, confusion matrix.
- Save model checkpoints at the best validation AUC and keep the last checkpoint.

## Notes & Best Practices

- Use stratified splits to keep class balance across train/val/test.
- Start with transfer learning and a small subset of data to validate the pipeline before scaling up.
- Log hyperparameters and results for each run in a structured way (`runs/exp-<id>/` with `metrics.json`).

## Next Steps

- Add `requirements.txt` and example `configs/train.yaml` if missing.
- Implement `scripts/prepare_data.py`, `train.py`, `evaluate.py`, and `inference.py` if not yet present.

## Contact

For questions or help with the pipeline, open an issue or contact the repository owner.
