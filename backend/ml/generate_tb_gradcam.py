
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import models, datasets, transforms

ROOT = Path(__file__).resolve().parents[2]
IMAGE_ROOT = ROOT / "data" / "raw" / "tb" / "tb_detection" / "images"
MODEL_PATH = ROOT / "artifacts" / "tb_image" / "best_model.pth"
OUTPUT_PATH = ROOT / "artifacts" / "tb_image" / "gradcam" / "sample_gradcam.png"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 160

VAL_DIR = IMAGE_ROOT / "val"
if not VAL_DIR.exists():
    raise FileNotFoundError(f"Validation folder not found: {VAL_DIR}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(VAL_DIR, transform=transform)
classes = dataset.classes

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model = model.to(DEVICE)
model.eval()

# Capture feature maps and their gradients from the last ResNet block.
saved = {}

def forward_hook(module, inputs, output):
    saved["activations"] = output
    output.register_hook(
        lambda grad: saved.__setitem__("gradients", grad)
    )

handle = model.layer4[-1].register_forward_hook(forward_hook)

image_tensor, true_label = dataset[0]
image_tensor = image_tensor.unsqueeze(0).to(DEVICE)
image_tensor.requires_grad_(True)

model.zero_grad(set_to_none=True)
logits = model(image_tensor)
predicted_idx = int(logits.argmax(dim=1).item())
logits[0, predicted_idx].backward()

handle.remove()

if "gradients" not in saved:
    raise RuntimeError("Grad-CAM gradients were not captured.")

activations = saved["activations"].detach()
gradients = saved["gradients"].detach()

weights = gradients.mean(dim=(2, 3), keepdim=True)
cam = torch.relu((weights * activations).sum(dim=1, keepdim=True))
cam = torch.nn.functional.interpolate(
    cam,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False,
)
cam = cam[0, 0]
cam = cam - cam.min()
cam = cam / (cam.max() + 1e-8)
cam = cam.cpu().numpy()

# Reopen the image for a clear, unnormalized display.
image_path, _ = dataset.samples[0]
original = Image.open(image_path).convert("RGB").resize(
    (IMAGE_SIZE, IMAGE_SIZE)
)
image_np = np.asarray(original)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

plt.figure(figsize=(9, 4))
plt.subplot(1, 2, 1)
plt.imshow(image_np)
plt.title(f"Original: {classes[true_label]}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(image_np)
plt.imshow(cam, cmap="jet", alpha=0.45)
plt.title(f"Grad-CAM: {classes[predicted_idx]}")
plt.axis("off")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
plt.close()

print("Grad-CAM generated successfully!")
print("Actual class:", classes[true_label])
print("Predicted class:", classes[predicted_idx])
print("Saved image:", OUTPUT_PATH)
