# hCaptcha pose classifier

Image-based binary classifier (TensorFlow CNN) for **sitting** vs **jumping** poses—the kind of visual challenge used in hCaptcha-style grids. This repo trains on local folders, evaluates accuracy, and can **select tile indices** that match a target pose (simulated challenge solving).

> **Note:** There is no browser or hCaptcha API integration. Use `predict.py --tiles-dir` on images you export from a challenge, or wire this model into your own automation.

## Project layout

| Path | Purpose |
|------|---------|
| `training/`, `validation/` | Raw labeled images (`jumping/`, `sitting/`) |
| `processed/` | Optional preprocessed copies (RGB, resize, blur)—created by convert |
| `models/pose_detection_model.h5` | Trained model (created by `train.py`) |
| `src/` | Config, preprocessing, model definition, inference helpers |

## Run on Windows

Double-click **`run.bat`** — one menu for setup, training, validation, and prediction.

**First time:** open `run.bat` → **1** (Setup) → **3** (Run all). For best accuracy, enter **150** when asked for epochs.

## Setup

```powershell
cd D:\Work\Project\hcaptcha-enterprise-solver-main
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python test.py
```

## Workflow

### 1. Preprocess (recommended)

Copies images into `processed/` with 128×128 RGB + Gaussian blur (does **not** overwrite raw data):

```powershell
python -m src.convert
# or convert.bat
```

### 2. Train

Uses `processed/training` and `processed/validation` when present; otherwise raw folders.

```powershell
python train.py
python train.py --epochs 150
python train.py --raw
```

### 3. Validate

```powershell
python validate.py
```

### 4. Predict

Single image:

```powershell
python predict.py --image validation\sitting\852e99efe430f4e0de5c79f64e03efe4ebb5e7c34a31b4f7f.png
```

hCaptcha-style tile folder (prints indices to click):

```powershell
python predict.py --tiles-dir path\to\tiles --target sitting --threshold 0.5
```

## Class labels

Keras `flow_from_directory` uses alphabetical order: `jumping` → 0, `sitting` → 1. The model sigmoid output is **P(sitting)**.

## Future work

- GAN/VAE synthetic data generation (see original TODO)
- Optional browser hook to crop challenge tiles automatically

## Reference

Training with strong augmentation can reach high validation accuracy on this dataset; see the sample chart in the upstream README image link.
