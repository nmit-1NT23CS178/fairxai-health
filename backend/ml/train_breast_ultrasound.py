from pathlib import Path
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# -------------------- CONFIG --------------------

SEED = 42
BATCH_SIZE = 16
EPOCHS = 8
IMAGE_SIZE = 224

CLASSES = ["benign", "malignant", "normal"]

ROOT = Path(__file__).resolve().parents[2]

DATASET_DIR = (
    ROOT / "data" / "raw" / "breast_cancer"
    / "extracted_images" / "Dataset_BUSI_with_GT"
)

OUTPUT_DIR = ROOT / "artifacts" / "breast_ultrasound"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = OUTPUT_DIR / "best_model.pth"
METRICS_PATH = OUTPUT_DIR / "metrics.json"

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Device: {DEVICE}")
print(f"Dataset: {DATASET_DIR}")
print(f"Output: {OUTPUT_DIR}")

# -------------------- COLLECT IMAGES --------------------

image_paths = []
labels = []

for label, class_name in enumerate(CLASSES):
    class_dir = DATASET_DIR / class_name

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Missing class folder: {class_dir}"
        )

    files = sorted(class_dir.glob("*.png"))

    # Exclude segmentation masks
    files = [
        path for path in files
        if "_mask" not in path.stem.lower()
    ]

    print(f"{class_name}: {len(files)} images")

    for path in files:
        image_paths.append(str(path))
        labels.append(label)

if not image_paths:
    raise RuntimeError(
        "No images found. Please check DATASET_DIR."
    )

if len(set(labels)) != len(CLASSES):
    raise RuntimeError("One or more classes have no images.")

# -------------------- STRATIFIED SPLIT --------------------
# Approximately 60% train, 20% validation, 20% test.

train_paths, temp_paths, train_labels, temp_labels = (
    train_test_split(
        image_paths,
        labels,
        test_size=0.40,
        random_state=SEED,
        stratify=labels,
    )
)

val_paths, test_paths, val_labels, test_labels = (
    train_test_split(
        temp_paths,
        temp_labels,
        test_size=0.50,
        random_state=SEED,
        stratify=temp_labels,
    )
)

print("\nDataset split:")
print(f"Training:   {len(train_paths)}")
print(f"Validation: {len(val_paths)}")
print(f"Testing:    {len(test_paths)}")

# -------------------- TRANSFORMS --------------------

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=7),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------- DATASET --------------------

class BreastUltrasoundDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        image_path = self.paths[index]
        label = self.labels[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label


train_dataset = BreastUltrasoundDataset(
    train_paths, train_labels, train_transform
)
val_dataset = BreastUltrasoundDataset(
    val_paths, val_labels, eval_transform
)
test_dataset = BreastUltrasoundDataset(
    test_paths, test_labels, eval_transform
)

# Balance the training samples to help with class imbalance.
class_counts = np.bincount(
    train_labels, minlength=len(CLASSES)
)

sample_weights = [
    1.0 / class_counts[label]
    for label in train_labels
]

sampler_generator = torch.Generator()
sampler_generator.manual_seed(SEED)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
    generator=sampler_generator,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# -------------------- MODEL --------------------

# Try ImageNet transfer learning. If weights cannot be
# downloaded or loaded, continue with random initialization.
try:
    model = resnet18(weights=ResNet18_Weights.DEFAULT)
    print("\nUsing pretrained ImageNet weights.")
except Exception as error:
    print(f"\nPretrained weights unavailable: {error}")
    print("Using randomly initialized ResNet18.")
    model = resnet18(weights=None)

# Freeze most layers to make CPU training faster.
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the final feature block and classification head.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

model.fc = nn.Linear(model.fc.in_features, len(CLASSES))

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=0.0001,
    weight_decay=0.0001,
)

# -------------------- EVALUATION HELPER --------------------

def evaluate(model, loader):
    model.eval()
    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, targets in loader:
            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            outputs = model(images)
            predictions = outputs.argmax(dim=1)

            all_labels.extend(targets.cpu().numpy().tolist())
            all_predictions.extend(
                predictions.cpu().numpy().tolist()
            )

    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(
        all_labels, all_predictions
    )
    macro_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0,
    )

    return accuracy, balanced_acc, macro_f1


# -------------------- TRAINING --------------------

best_val_f1 = -1.0

print("\nStarting training...\n")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    total_samples = 0

    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    train_loss = running_loss / max(total_samples, 1)

    val_accuracy, val_balanced_acc, val_macro_f1 = evaluate(
        model, val_loader
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Balanced Accuracy: {val_balanced_acc:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )

    # Save the checkpoint with the best validation macro F1.
    if val_macro_f1 > best_val_f1:
        best_val_f1 = val_macro_f1

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "image_size": IMAGE_SIZE,
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
            MODEL_PATH,
        )

        print("  Best model saved.")

# -------------------- FINAL TEST --------------------

print("\nLoading best model for final test...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model.load_state_dict(checkpoint["model_state_dict"])

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for images, targets in test_loader:
        images = images.to(DEVICE)

        outputs = model(images)
        predictions = outputs.argmax(dim=1)

        y_true.extend(targets.numpy().tolist())
        y_pred.extend(predictions.cpu().numpy().tolist())

test_accuracy = accuracy_score(y_true, y_pred)
test_balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
test_macro_f1 = f1_score(
    y_true, y_pred, average="macro", zero_division=0
)

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
    target_names=CLASSES,
    output_dict=True,
    zero_division=0,
)

matrix = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(CLASSES))),
)

metrics = {
    "classes": CLASSES,
    "total_images": len(image_paths),
    "train_images": len(train_paths),
    "validation_images": len(val_paths),
    "test_images": len(test_paths),
    "test_accuracy": float(test_accuracy),
    "test_balanced_accuracy": float(test_balanced_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
}

with open(METRICS_PATH, "w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=4)

print("\n========== FINAL TEST RESULTS ==========")
print(f"Test Accuracy:          {test_accuracy:.4f}")
print(f"Test Balanced Accuracy: {test_balanced_accuracy:.4f}")
print(f"Test Macro F1:          {test_macro_f1:.4f}")

print("\nPer-class report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(len(CLASSES))),
        target_names=CLASSES,
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(matrix)

print(f"\nModel saved to: {MODEL_PATH}")
print(f"Metrics saved to: {METRICS_PATH}")
print("\nTraining completed.")
print(
    "Educational prototype only; this model is not "
    "validated for medical diagnosis."
)