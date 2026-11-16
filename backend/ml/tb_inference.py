
from pathlib import Path
import io
import json
import os

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
IMAGE_DIR = Path(os.environ.get("IMAGE_ARTIFACT_DIR", ROOT / "artifacts"))
MODEL_PATH = IMAGE_DIR / "tb_image" / "best_model.pth"
METRICS_PATH = IMAGE_DIR / "tb_image" / "metrics.json"

# Small CPU-only hosts: one thread keeps memory and CPU use predictable.
torch.set_num_threads(int(os.environ.get("TORCH_NUM_THREADS", "1")))
DEVICE = torch.device("cpu")

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"TB model not found: {MODEL_PATH}")

# Class order and image size are read from the shipped metrics file, NOT from data/raw: the raw
# dataset is intentionally excluded from the Docker image, so depending on it broke production.
try:
    _metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
except Exception:
    _metrics = {}
CLASSES = _metrics.get("classes") or ["health", "sick", "tb"]
IMAGE_SIZE = int(_metrics.get("image_size", 160))

model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_tb_image(file_bytes):
    """Predict a class and return confidence plus Grad-CAM PNG bytes."""
    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError("Please upload a valid image file.") from exc

    display_image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    tensor.requires_grad_(True)

    captured = {}

    def forward_hook(module, inputs, output):
        captured["activation"] = output
        output.register_hook(
            lambda grad: captured.__setitem__("gradient", grad)
        )

    handle = model.layer4[-1].register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        probabilities = torch.softmax(logits.detach(), dim=1)[0]
        predicted_idx = int(probabilities.argmax().item())
        confidence = float(probabilities[predicted_idx].item())

        logits[0, predicted_idx].backward()

        if "gradient" not in captured:
            raise RuntimeError("Could not calculate Grad-CAM gradients.")

        weights = captured["gradient"].detach().mean(
            dim=(2, 3), keepdim=True
        )
        cam = torch.relu(
            (weights * captured["activation"].detach()).sum(
                dim=1, keepdim=True
            )
        )
        cam = torch.nn.functional.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        heatmap = cam.cpu().numpy()

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)

    original = np.asarray(display_image).astype(np.float32) / 255.0
    overlay = original.copy()
    overlay[:, :, 0] = np.maximum(overlay[:, :, 0], heatmap * 0.7)
    overlay[:, :, 1] *= (1 - heatmap * 0.45)
    overlay[:, :, 2] *= (1 - heatmap * 0.45)
    overlay = np.clip(overlay, 0, 1)

    # Compose the side-by-side image with Pillow: matplotlib pulls in tens of MB of native
    # libraries and is unnecessary for a fixed two-panel PNG on a memory-constrained host.
    original_u8 = (np.clip(original, 0, 1) * 255).astype(np.uint8)
    overlay_u8 = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)
    canvas = Image.new("RGB", (IMAGE_SIZE * 2, IMAGE_SIZE))
    canvas.paste(Image.fromarray(original_u8), (0, 0))
    canvas.paste(Image.fromarray(overlay_u8), (IMAGE_SIZE, 0))
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    buffer.seek(0)

    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": round(confidence, 4),
        "classes": CLASSES,
        "disclaimer": (
            "Research prototype only. This prediction is not a diagnosis "
            "and must not be used for clinical decisions."
        ),
        "gradcam_png_base64": __import__("base64").b64encode(
            buffer.getvalue()
        ).decode("ascii"),
    }
