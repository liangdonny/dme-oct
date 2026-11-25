# %%
import os
import re
import time

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, ScaleIntensityd, SpatialCropd,
    RandRotated, RandFlipd, RandZoomd, Resized, MapTransform, ToTensord
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism
from monai.networks.nets import resnet18

# %% Configuration
# Folder containing "Continue Injection/" and "Stop Injection/" subfolders.
# Set the DME_DATA_DIR environment variable or edit the default below.
DATA_DIR = os.environ.get("DME_DATA_DIR", "data/train")
OUTPUT_DIR = "outputs"

CLASS_DIRS = {"Continue Injection": 0, "Stop Injection": 1}
CLASS_NAMES = ["Continue", "Stop"]
IMG_EXTS = {".jpg", ".jpeg", ".png"}

# Regex that pulls a patient ID out of each file name, used to keep every
# patient's scans in a single split. The first capture group is the ID.
# Example: files named "P012_OD_scan3.jpg" -> r"^(P\d+)_"
# Leave as None to split by image (this can overstate performance if a
# patient has more than one scan).
PATIENT_ID_PATTERN = None

SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

BATCH_SIZE = 128
ACCUMULATION_STEPS = 5
LEARNING_RATE = 1e-4
EPOCHS = 25
VAL_INTERVAL = 1

os.makedirs(OUTPUT_DIR, exist_ok=True)
set_determinism(seed=SEED)

# %% Load file paths and labels
image_files, labels = [], []
for class_dir, label in CLASS_DIRS.items():
    folder = os.path.join(DATA_DIR, class_dir)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Missing class folder: {folder}")
    for name in sorted(os.listdir(folder)):
        if os.path.splitext(name)[1].lower() in IMG_EXTS:
            image_files.append(os.path.join(folder, name))
            labels.append(label)

if not image_files:
    raise FileNotFoundError(f"No images found under {DATA_DIR}")

image_files = np.array(image_files)
labels = np.array(labels)


def get_patient_ids(paths, pattern):
    if pattern is None:
        return None
    regex = re.compile(pattern)
    ids = []
    for p in paths:
        match = regex.search(os.path.basename(p))
        if match is None:
            raise ValueError(f"PATIENT_ID_PATTERN did not match file: {p}")
        ids.append(match.group(1))
    return np.array(ids)


patient_ids = get_patient_ids(image_files, PATIENT_ID_PATTERN)

# %% Train / validation / test split
def split_off(indices, y, groups, fraction, seed):
    """Split `indices` into (kept, held_out), stratified by label.
    If groups are given, no group appears on both sides."""
    if groups is None:
        return train_test_split(indices, test_size=fraction, stratify=y[indices], random_state=seed)
    n_splits = max(2, round(1 / fraction))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    kept, held_out = next(sgkf.split(np.zeros(len(indices)), y[indices], groups[indices]))
    return indices[kept], indices[held_out]


all_idx = np.arange(len(image_files))
trainval_idx, test_idx = split_off(all_idx, labels, patient_ids, TEST_FRACTION, SEED)
# Validation fraction is taken relative to the remaining train+val pool.
train_idx, val_idx = split_off(
    trainval_idx, labels, patient_ids, VAL_FRACTION / (1 - TEST_FRACTION), SEED
)

if patient_ids is None:
    print("WARNING: splitting by image, not by patient. Set PATIENT_ID_PATTERN if patients have multiple scans.")
else:
    assert not set(patient_ids[train_idx]) & set(patient_ids[val_idx])
    assert not set(patient_ids[train_idx]) & set(patient_ids[test_idx])
    assert not set(patient_ids[val_idx]) & set(patient_ids[test_idx])

for split_name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
    counts = np.bincount(labels[idx], minlength=2)
    print(f"{split_name:>5}: {len(idx)} images  (Continue={counts[0]}, Stop={counts[1]})")


def make_records(idx):
    return [{"image": str(image_files[i]), "label": int(labels[i])} for i in idx]


# %% Transforms
class EnsureThreeChannels(MapTransform):
    """Convert grayscale / grayscale+alpha / RGBA images to 3 channels."""

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            image = d[key]
            c = image.shape[0]
            if c in (1, 2):      # grayscale, or grayscale + alpha
                d[key] = image[:1].repeat(3, 1, 1)
            elif c > 3:          # RGBA or extra channels
                d[key] = image[:3]
        return d


# Note: MONAI's PIL reader returns images as (channels, width, height),
# so spatial axis 0 is the horizontal axis.
base_transforms = [
    LoadImaged(keys=["image"]),
    EnsureChannelFirstd(keys=["image"]),
    EnsureThreeChannels(keys=["image"]),
    SpatialCropd(keys=["image"], roi_start=[500, 0], roi_end=[5000, 500]),
    Resized(keys=["image"], spatial_size=(224, 224)),
    ScaleIntensityd(keys=["image"], minv=0.0, maxv=1.0),
]

max_angle = np.deg2rad(15)  # RandRotated expects radians
train_transforms = Compose(base_transforms + [
    RandRotated(keys=["image"], range_x=(-max_angle, max_angle), prob=0.5),
    RandFlipd(keys=["image"], spatial_axis=0, prob=0.5),  # horizontal flip
    RandZoomd(keys=["image"], min_zoom=0.9, max_zoom=1.1, prob=0.3),
    ToTensord(keys=["image"]),
])

eval_transforms = Compose(base_transforms + [ToTensord(keys=["image"])])

# %% Datasets and loaders
train_ds = Dataset(data=make_records(train_idx), transform=train_transforms)
val_ds = Dataset(data=make_records(val_idx), transform=eval_transforms)
test_ds = Dataset(data=make_records(test_idx), transform=eval_transforms)

# drop_last only if there is more than one full batch, so small datasets still train
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          drop_last=len(train_ds) > BATCH_SIZE)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

if len(train_loader) == 0:
    raise ValueError("Training set is empty.")

# %% Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = resnet18(
    spatial_dims=2,
    n_input_channels=3,
    num_classes=2,
    pretrained=False,
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.CrossEntropyLoss()


def evaluate(model, loader):
    """Return mean loss, AUC (None if only one class present), labels, preds, probs."""
    model.eval()
    total_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for batch in loader:
            inputs = batch["image"].to(device)
            targets = batch["label"].long().to(device)
            outputs = model(inputs)
            total_loss += loss_fn(outputs, targets).item()

            probs = torch.softmax(outputs, dim=1)[:, 1]  # P(Stop Injection)
            all_labels.extend(targets.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    mean_loss = total_loss / len(loader)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) == 2 else None
    return mean_loss, auc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


def save_confusion_matrix(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    disp.plot(cmap="Blues")
    plt.title(title)
    plt.savefig(path, bbox_inches="tight")
    plt.close(disp.figure_)


# %% Training
train_losses, val_losses, val_epochs = [], [], []
auc_values, auc_epochs = [], []
best_auc = -1.0
best_model_path = os.path.join(OUTPUT_DIR, "best_model.pt")
n_batches = len(train_loader)
start_time = time.time()

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    optimizer.zero_grad()

    progress = tqdm(enumerate(train_loader), total=n_batches, desc=f"Epoch {epoch}/{EPOCHS}")
    for i, batch in progress:
        inputs = batch["image"].to(device)
        targets = batch["label"].long().to(device)

        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

        # Scale by the size of this accumulation group so the summed gradient
        # equals the average over the group (the last group may be shorter).
        group_start = (i // ACCUMULATION_STEPS) * ACCUMULATION_STEPS
        group_size = min(ACCUMULATION_STEPS, n_batches - group_start)
        (loss / group_size).backward()
        epoch_loss += loss.item()

        if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == n_batches:
            optimizer.step()
            optimizer.zero_grad()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    epoch_loss /= n_batches
    train_losses.append(epoch_loss)
    print(f"Training loss: {epoch_loss:.4f}")

    if epoch % VAL_INTERVAL == 0:
        val_loss, val_auc, y_true, y_pred, _ = evaluate(model, val_loader)
        val_losses.append(val_loss)
        val_epochs.append(epoch)
        print(f"Validation loss: {val_loss:.4f}")

        if val_auc is not None:
            auc_values.append(val_auc)
            auc_epochs.append(epoch)
            print(f"Validation AUC: {val_auc:.4f}")
            if val_auc > best_auc:
                best_auc = val_auc
                torch.save(model.state_dict(), best_model_path)
                print("Saved new best model")
        else:
            print("Validation AUC skipped: validation set contains only one class")

        save_confusion_matrix(y_true, y_pred, f"Validation Confusion Matrix - Epoch {epoch}",
                              os.path.join(OUTPUT_DIR, f"confusion_matrix_epoch_{epoch}.png"))

print(f"Total training time: {time.time() - start_time:.2f} seconds")

# %% Training curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

ax1.set_title("Training and Validation Loss")
ax1.plot(range(1, EPOCHS + 1), train_losses, label="Training Loss")
ax1.plot(val_epochs, val_losses, label="Validation Loss")
ax1.set_xlabel("Epoch")
ax1.legend()

ax2.set_title("Validation AUC")
ax2.plot(auc_epochs, auc_values)
ax2.set_xlabel("Epoch")
ax2.set_ylabel("AUC")
ax2.set_ylim(0, 1)
ax2.grid(True)

fig.tight_layout()
fig.savefig(os.path.join(OUTPUT_DIR, "training_summary.png"))
plt.close(fig)

# %% Final evaluation on the held-out test set (best validation checkpoint)
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=device))
    print(f"Loaded best model (validation AUC {best_auc:.4f})")
else:
    print("No checkpoint saved; evaluating final-epoch model")

test_loss, test_auc, y_true, y_pred, _ = evaluate(model, test_loader)
print(f"Test loss: {test_loss:.4f}")
print(f"Test accuracy: {accuracy_score(y_true, y_pred):.4f}")
print(f"Test AUC: {test_auc:.4f}" if test_auc is not None else "Test AUC skipped: one class only")
save_confusion_matrix(y_true, y_pred, "Test Confusion Matrix",
                      os.path.join(OUTPUT_DIR, "confusion_matrix_test.png"))
