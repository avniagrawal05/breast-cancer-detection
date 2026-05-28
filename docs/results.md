# Results & Evaluation

## Metrics and Reporting

Track the following metrics per epoch:
- Loss/accuracy
- AUC (Area Under Curve)
- Precision/Recall
- Confusion matrix
- Sensitivity/Specificity

Save model checkpoints at the best validation AUC and keep the last checkpoint for comparison.

## Evaluation Process

Run the evaluation script on the `test/` set to compute metrics and save plots:

```bash
python evaluate.py --checkpoint runs/exp1/checkpoint.pt --data data/processed/test
```

Results will be saved in the `runs/` directory structure with:
- Metric summaries (JSON format)
- Visualizations (confusion matrices, ROC curves, etc.)
- Model predictions and confidence scores

## Notes & Best Practices

- **Stratified Splits:** Use stratified splits to keep class balance across train/val/test sets to prevent data leakage and ensure representative evaluation.
- **Pipeline Validation:** Start with transfer learning and a small subset of data to validate the pipeline before scaling up.
- **Structured Logging:** Keep hyperparameters and results for each run in a structured way (`runs/exp-<id>/` with `metrics.json`).
- **Reproducibility:** Save the training seed, environment (via `pip freeze > requirements.txt`), and configuration files alongside model checkpoints.
- **Cross-validation:** Consider k-fold cross-validation for more robust evaluation on smaller datasets.

## Inference & Deployment

Use `inference.py` or the provided notebook to run the model on new images and output:
- Predictions (class label and confidence score)
- Visualizations (e.g., Grad-CAM for model interpretability)
- Batch processing capabilities for high-throughput inference

Example:

```bash
python inference.py --checkpoint runs/exp1/checkpoint.pt --input images/sample.jpg --output results/sample_pred.json
```
