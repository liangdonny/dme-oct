# OCT-Based Prediction of Anti-VEGF Treatment Continuation in Diabetic Macular Edema

A deep learning model that classifies retinal OCT scans from patients with diabetic macular edema (DME) into two treatment decisions: **continue** or **stop** anti-VEGF injections.

This project was developed at **Olmeda Biosciences**.

![Training summary](results/training_summary.png)

## Background

Diabetic macular edema is a complication of diabetic retinopathy in which fluid builds up in the macula, the central part of the retina responsible for sharp vision. The standard treatment is a series of intravitreal anti-VEGF injections. Treatment is long-term and burdensome for patients, and deciding when injections can safely stop is a recurring clinical judgment made largely from optical coherence tomography (OCT) imaging.

This project tests whether a convolutional neural network can learn that decision directly from OCT images, framed as a binary classification problem:

| Label | Class | Meaning |
|------:|-------|---------|
| 0 | Continue Injection | Imaging suggests ongoing treatment is needed |
| 1 | Stop Injection | Imaging suggests treatment can be discontinued |

## Method

**Model:** ResNet-18 (MONAI implementation, 2D, 3 input channels, 2 output classes), trained from scratch with no pretrained weights.

**Preprocessing**
- Load OCT images and convert to 3-channel format (grayscale images are repeated across channels)
- Fixed region-of-interest crop (`SpatialCropd`)
- Resize to 224 x 224
- Intensity scaling to [0, 1]

**Training augmentation**
- Random rotation of up to +/-15 degrees (p = 0.5)
- Random horizontal flip (p = 0.5)
- Random zoom, 0.9 to 1.1 (p = 0.3)

**Training setup**

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam, lr = 1e-4 |
| Loss | Cross-entropy |
| Batch size | 128 |
| Gradient accumulation | 5 steps (effective batch size 640) |
| Epochs | 25 |
| Train / validation / test split | 70 / 15 / 15, stratified by label |
| Random seed | 42 |

**Patient-level splitting:** If `PATIENT_ID_PATTERN` is set in `DME_OCT.py`, the split uses `StratifiedGroupKFold` so that all scans from one patient land in the same split. This prevents the model from being validated or tested on near-duplicate scans of patients it was trained on.

**Evaluation:** Validation loss, ROC AUC (using the predicted probability of the "Stop Injection" class), and a confusion matrix are computed after every epoch. The checkpoint with the best validation AUC is saved and evaluated once on the held-out test set at the end of training.

## Results

| Metric (epoch 25) | Value |
|-------------------|-------|
| Validation ROC AUC | ~0.95 |
| Validation loss | ~0.33 |
| Training loss | ~0.42 |

- Validation AUC was near chance for the first five epochs, rose above 0.8 by epoch 8, and reached roughly 0.92 to 0.95 over epochs 18 to 25.
- Training loss decreased steadily from about 0.65 to 0.42.
- Validation loss was unstable throughout training (peaking around 1.35 at epoch 12) before settling lower in the final epochs, likely due to the small validation set and training from scratch.

> **Note:** These results come from the original version of the training code, which used an image-level 80/20 train/validation split, no test set, and had bugs in augmentation and gradient accumulation (see [Changelog](#changelog)). Results from the current code may differ.

## Repository Structure

```
.
├── DME_OCT.py                       # Data loading, preprocessing, training, and evaluation
├── requirements.txt
├── results/
│   └── training_summary.png       # Loss and AUC curves from the original 25-epoch run
└── README.md
```

Running the script writes the following to `outputs/`:
- `best_model.pt`: weights with the highest validation AUC
- `confusion_matrix_epoch_<n>.png`: validation confusion matrix for each epoch
- `confusion_matrix_test.png`: test-set confusion matrix
- `training_summary.png`: loss and AUC curves

## Data

The DME dataset used in this project is **not included** in this repository. The script expects the following layout:

```
<data_dir>/
├── Continue Injection/   *.jpg / *.jpeg / *.png
└── Stop Injection/       *.jpg / *.jpeg / *.png
```

The class label is taken from the folder name. By default the script reads from `data/DME_OCT`. To point it somewhere else, set the `DME_DATA_DIR` environment variable or edit `DATA_DIR` in `DME_OCT.py`.

If your file names contain a patient ID, set `PATIENT_ID_PATTERN` to a regular expression whose first capture group is that ID. For example, for files named `P012_OD_scan3.jpg`:

```python
PATIENT_ID_PATTERN = r"^(P\d+)_"
```

## Usage

```bash
git clone https://github.com/<your-username>/dme-oct.git
cd dme-oct
pip install -r requirements.txt
DME_DATA_DIR="path/to/data" python DME_OCT.py
```

On Windows PowerShell:

```powershell
$env:DME_DATA_DIR = "C:\path\to\data"
python DME_OCT.py
```

The script is organized into `# %%` cells, so it can also be run cell by cell in VS Code or Spyder. A CUDA GPU is used automatically if available.

## Changelog

Fixes made after the original run:
- **AUC calculation:** The original call passed two-column probabilities with `multi_class="ovr"`, which raises an error in current scikit-learn on binary labels. AUC is now computed from the "Stop Injection" probability.
- **Rotation range:** `RandRotated` takes radians, so the original `(-15, 15)` applied near-arbitrary rotations. It is now +/-15 degrees.
- **Gradient accumulation:** The loss is now divided by the number of batches in each accumulation group, so accumulated gradients are averaged rather than summed.
- **Data splitting:** Added a held-out test set and optional patient-level splitting.
- **Checkpointing:** The best model by validation AUC is now saved and used for test evaluation.
- **Confusion matrix:** Always uses both class labels, so it no longer breaks if a split contains only one class.
- **Portability:** Removed hardcoded local paths, made image file extensions case-insensitive, and removed an unused dataset loader.

## Limitations and Future Work

- **Retrain with the fixed code.** The reported results are from the original code. They should be regenerated with patient-level splitting and the held-out test set.
- **Single run.** Results come from one training run with one seed. Repeated runs or cross-validation would show how much the metrics vary.
- **Pretrained weights.** Starting from ImageNet-pretrained weights would likely speed up convergence and stabilize validation loss on a dataset of this size.

## Tech Stack

Python, PyTorch, MONAI, scikit-learn, NumPy, Matplotlib

## Disclaimer

This is a research project and is not intended for clinical use.
