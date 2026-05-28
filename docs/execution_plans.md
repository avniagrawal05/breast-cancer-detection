# Execution Plans

## Step-by-step Code Execution

### 1. Prerequisites

- Python 3.8+ recommended.
- GPU available for training (CUDA/cuDNN) recommended but not required for small experiments.
- See [Requirements](requirements.md) for detailed setup.

### 2. Environment Setup

Create and activate a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

See [Requirements](requirements.md) for detailed environment setup.

### 3. Data Preparation

Place images under `data/` or provide a CSV mapping file at `data/labels.csv`.
Run the data-prep script/notebook to split into `train/`, `val/`, and `test/` folders and to generate any preprocessing caches.

```bash
python scripts/prepare_data.py --input data/raw --output data/processed --split 0.7 0.15 0.15
```

### 4. Training

Configure training parameters (batch size, learning rate, number of epochs) in `configs/train.yaml` or via CLI args.
Start training and log checkpoints and metrics to `runs/` (or a logger like TensorBoard/Weights & Biases).

```bash
python train.py --config configs/train.yaml
# or
python -m torch.distributed.launch --nproc_per_node=1 train.py --config configs/train.yaml
```

### 5. Evaluation

Run the evaluation script on the `test/` set to compute metrics and save plots.

```bash
python evaluate.py --checkpoint runs/exp1/checkpoint.pt --data data/processed/test
```

### 6. Inference

Use `inference.py` or the provided notebook to run the model on new images and output predictions/visualizations (e.g., Grad-CAM).

```bash
python inference.py --checkpoint runs/exp1/checkpoint.pt --input images/sample.jpg --output results/sample_pred.json
```

### 7. Reproducibility

Save the training seed, environment (via `pip freeze > requirements.txt`), and configuration files alongside model checkpoints for reproducibility.

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

## Next Steps

- Add `requirements.txt` and example `configs/train.yaml` if missing.
- Implement `scripts/prepare_data.py`, `train.py`, `evaluate.py`, and `inference.py` if not yet present.
