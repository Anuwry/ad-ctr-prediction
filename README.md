# Ad Ecosystem and Click-Through-Rate Prediction

CTR probability prediction with PyTorch feed-forward networks, LightGBM and isotonic calibration.

## Project context

Super AI Engineering Season 6, 2026. Individual competition project.

## Approach

Feature engineering, feed-forward models, LightGBM, weighted blending and isotonic probability-calibration experiments.

## Competition result

Competition **Normalized Cross-Entropy (NCE): 0.902722**. Submitted clicked values are probabilities in [0, 1]. Local ROC-AUC and log loss are different metrics.

Leaderboard results are from the project author's competition records, not newly generated benchmark results.

## Source files

- notebooks/ctr_fcn_lightgbm.ipynb
- notebooks/ctr_blend_experiments.ipynb

## Setup and use

1. Create a Python environment compatible with the historical notebook and selected model.
2. Install relevant dependencies from requirements.txt. GPU frameworks and Unsloth may require platform-specific CUDA instructions.
3. Provide authorized competition train/test CSV files and configure notebook paths. Customer rows and prediction files are excluded.
4. Start JupyterLab and run the selected notebook in order. Review experiment branches before a full training run.

    python -m venv .venv
    python -m pip install -r requirements.txt
    python -m jupyterlab

## Archive preparation

- Notebook outputs and execution metadata were cleared.
- Hardcoded credentials found in selected source were replaced with environment lookups.
- Datasets, personal records, submissions, model weights and local environments are excluded.
- Source was statically checked; training and external API calls were not rerun.

## Limitations

Multiple historical experiment variants. Internal scores and calibration results are not the official leaderboard metric. No exact final checkpoint is asserted.
