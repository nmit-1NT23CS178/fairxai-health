
"""
FairXAI - TB Image Classification
- Pretrained ResNet-18
- CPU-friendly transfer learning (frozen feature extractor)
- Class-weighted loss
- Training/validation progress
- Best checkpoint and evaluation metrics
- Grad-CAM visual explanations

Research prototype only; not for clinical diagnosis.
"""

from pathlib import Path
import json
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

IMAGE_ROOT = (
    ROOT / "data" / "raw" / "tb" / "tb_detection" / "images"
)
OUTPUT = ROOT / "artifacts" / "tb_image"
CAM_DIR = OUTPUT / "gradcam"

OUTPUT.mkdir(parents=True, exist_ok=True)
CAM_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 1  # First run: smoke test. Increase after successful run.
IMAGE_SIZE = 160
NUM_WORKERS = 0
DEVICE = torch.device("cpu")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Avoid excessive CPU thread contention on some laptops.
torch.set_num_threads(max(1, min(4, torch.get_num_threads())))

print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)


# --------------------------------------------------
# 2. Dataset and transforms
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

train_dir = IMAGE_ROOT / "train"
val_dir = IMAGE_ROOT / "val"

if not train_dir.exists() or not val_dir.exists():
    raise FileNotFoundError(
        f"Expected train and val folders under {IMAGE_ROOT}"
    )

train_ds = datasets.ImageFolder(
    train_dir, transform=train_transform
)
val_ds = datasets.ImageFolder(
    val_dir, transform=eval_transform
)

if train_ds.classes != val_ds.classes:
    raise ValueError(
        f"Train classes {train_ds.classes} differ from "
        f"validation classes {val_ds.classes}"
    )

classes = train_ds.classes
num_classes = len(classes)

print("\nClasses:", classes)
print("Training images:", len(train_ds))
print("Validation images:", len(val_ds))

train_counts = np.bincount(
    [label for _, label in train_ds.samples],
    minlength=num_classes,
)
val_counts = np.bincount(
    [label for _, label in val_ds.samples],
    minlength=num_classes,
)

print("Training class counts:", dict(zip(classes, train_counts.tolist())))
print("Validation class counts:", dict(zip(classes, val_counts.tolist())))

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=False,
)


# --------------------------------------------------
# 3. Pretrained ResNet-18 with frozen feature extractor
# --------------------------------------------------

print("\nLoading pretrained ResNet-18...", flush=True)

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Freeze convolutional feature extractor.
for param in model.parameters():
    param.requires_grad = False

# Replace and train only the classification head.
model.fc = nn.Linear(model.fc.in_features, num_classes)

for param in model.fc.parameters():
    param.requires_grad = True

model = model.to(DEVICE)

class_weights = (
    len(train_ds)
    / (num_classes * np.maximum(train_counts, 1))
)
class_weights = torch.tensor(
    class_weights, dtype=torch.float32, device=DEVICE
)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(
    model.fc.parameters(),
    lr=1e-3,
    weight_decay=1e-4,
)

print("Feature extractor frozen; classifier head trainable.", flush=True)


# --------------------------------------------------
# 4. Training and validation
# --------------------------------------------------

best_val_loss = float("inf")
best_epoch = 0
history = []
start_time = time.time()

for epoch in range(EPOCHS):
    epoch_start = time.time()
    model.train()

    # Keep frozen feature extractor in evaluation mode.
    model.eval()
    model.fc.train()

    running_loss = 0.0
    processed = 0

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} - training started",
        flush=True,
    )

    for batch_idx, (images, labels) in enumerate(train_loader, start=1):
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        # No gradients required for frozen backbone.
        with torch.no_grad():
            features = model.avgpool(
                model.layer4(
                    model.layer3(
                        model.layer2(
                            model.layer1(model.maxpool(model.relu(model.bn1(model.conv1(images)))))
                        )
                    )
                )
            )
            features = torch.flatten(features, 1)

        logits = model.fc(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        processed += images.size(0)

        if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(train_loader):
            print(
                f"\rTrain batch {batch_idx}/{len(train_loader)} "
                f"({processed}/{len(train_ds)} images) "
                f"loss={running_loss / processed:.4f}",
                end="\n",
                flush=True,
            )

    train_loss = running_loss / len(train_ds)

    # Validation
    model.eval()
    val_loss_sum = 0.0
    y_true = []
    y_pred = []
    y_prob = []

    print("Validation started...", flush=True)

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader, start=1):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)

            val_loss_sum += loss.item() * images.size(0)
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())
            y_prob.extend(probs.cpu().tolist())

            if batch_idx == 1 or batch_idx % 25 == 0 or batch_idx == len(val_loader):
                print(
                    f"\rValidation batch {batch_idx}/{len(val_loader)}",
                    end="\n",
                    flush=True,
                )

    val_loss = val_loss_sum / len(val_ds)
    val_accuracy = accuracy_score(y_true, y_pred)

    row = {
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "epoch_seconds": round(time.time() - epoch_start, 2),
    }
    history.append(row)

    print(
        f"Epoch {epoch + 1} complete | "
        f"train_loss={train_loss:.4f} | "
        f"val_loss={val_loss:.4f} | "
        f"val_accuracy={val_accuracy:.4f} | "
        f"time={row['epoch_seconds']:.1f}s",
        flush=True,
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_epoch = epoch + 1
        torch.save(model.state_dict(), OUTPUT / "best_model.pth")
        print("Saved best checkpoint.", flush=True)


# --------------------------------------------------
# 5. Final evaluation on the validation set
# --------------------------------------------------

checkpoint = OUTPUT / "best_model.pth"
if not checkpoint.exists():
    raise RuntimeError("No checkpoint was saved.")

model.load_state_dict(
    torch.load(checkpoint, map_location=DEVICE, weights_only=True)
)
model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for images, labels in val_loader:
        logits = model(images.to(DEVICE))
        probs = torch.softmax(logits, dim=1)

        y_true.extend(labels.tolist())
        y_pred.extend(logits.argmax(dim=1).cpu().tolist())
        y_prob.extend(probs.cpu().tolist())

report = classification_report(
    y_true,
    y_pred,
    labels=list(range(num_classes)),
    target_names=classes,
    output_dict=True,
    zero_division=0,
)

try:
    if num_classes == 2:
        auc = roc_auc_score(
            y_true, np.array(y_prob)[:, 1]
        )
    else:
        auc = roc_auc_score(
            y_true,
            np.array(y_prob),
            multi_class="ovr",
            average="macro",
            labels=list(range(num_classes)),
        )
except ValueError:
    auc = None

metrics = {
    "task": "three-class TB image dataset classification",
    "classes": classes,
    "train_images": len(train_ds),
    "validation_images": len(val_ds),
    "train_class_counts": dict(zip(classes, train_counts.tolist())),
    "validation_class_counts": dict(zip(classes, val_counts.tolist())),
    "validation_accuracy": float(accuracy_score(y_true, y_pred)),
    "validation_roc_auc": float(auc) if auc is not None else None,
    "confusion_matrix": confusion_matrix(
        y_true, y_pred, labels=list(range(num_classes))
    ).tolist(),
    "classification_report": report,
    "history": history,
    "best_epoch": best_epoch,
    "best_validation_loss": float(best_val_loss),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "epochs": EPOCHS,
    "seed": SEED,
    "device": str(DEVICE),
    "backbone": "ResNet-18 pretrained; frozen feature extractor",
    "training_seconds": round(time.time() - start_time, 2),
    "limitations": [
        "Research prototype; not clinically validated.",
        "Class labels must be verified against dataset documentation.",
        "Validation metrics are not external clinical validation.",
        "Frozen-backbone training is a fast baseline, not full fine-tuning.",
    ],
}

(OUTPUT / "metrics.json").write_text(
    json.dumps(metrics, indent=2), encoding="utf-8"
)

(OUTPUT / "training_history.json").write_text(
    json.dumps(history, indent=2), encoding="utf-8"
)

print("\nFinal validation accuracy:", round(metrics["validation_accuracy"], 4))
print("Validation ROC-AUC:", metrics["validation_roc_auc"])
print("\nClassification report:")
print(
    classification_report(
        y_true,
        y_pred,
        labels=list(range(num_classes)),
        target_names=classes,
        zero_division=0,
    )
)
print("Confusion matrix:")
print(metrics["confusion_matrix"])
print("\nSaved model:", checkpoint)
print("Saved metrics:", OUTPUT / "metrics.json")


# --------------------------------------------------
# 6. Grad-CAM explanation for one validation image
# --------------------------------------------------

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activations
        )
        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradients
        )

    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()


    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()


    def generate(self, image_tensor, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)

        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())

        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )

        cam = cam[0, 0]
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy(), class_idx

    def close(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

try:
    # Grad-CAM requires gradients, so do not wrap this section in no_grad().
    sample_path, sample_label = val_ds.samples[0]
    sample_image = Image.open(sample_path).convert("RGB")
    input_tensor = eval_transform(sample_image).unsqueeze(0).to(DEVICE)

    cam_tool = GradCAM(model, model.layer4[-1])
    input_tensor.requires_grad_(True)
    cam, predicted_idx = cam_tool.generate(input_tensor)
    cam_tool.close()

    # Display the original image with the heatmap overlay.
    image_np = np.array(
        sample_image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

    plt.figure(figsize=(8, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(image_np)
    plt.title(f"Original: {classes[sample_label]}")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(image_np)
    plt.imshow(cam, cmap="jet", alpha=0.45)
    plt.title(f"Grad-CAM: {classes[predicted_idx]}")
    plt.axis("off")

    plt.tight_layout()
    gradcam_path = CAM_DIR / "sample_gradcam.png"
    plt.savefig(gradcam_path, dpi=150, bbox_inches="tight")
    plt.close()

    print("Saved Grad-CAM:", gradcam_path)

except Exception as exc:
    print("Grad-CAM generation failed:", repr(exc))

print("\nTraining pipeline finished.")
print("Artifacts directory:", OUTPUT)
