from pathlib import Path
import io
import base64

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "artifacts" / "breast_ultrasound" / "best_model.pth"
)

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

CLASSES = ["benign", "malignant", "normal"]

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Breast ultrasound model not found: {MODEL_PATH}"
    )

# Load the separately trained breast ultrasound model.
checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=True,
)

CLASSES = checkpoint["classes"]
IMAGE_SIZE = checkpoint["image_size"]

model = models.resnet18(weights=None)
model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASSES),
)

model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=checkpoint["mean"],
        std=checkpoint["std"],
    ),
])


def predict_breast_ultrasound(file_bytes):
    """Predict an ultrasound image and return confidence and Grad-CAM."""

    try:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except Exception as exc:
        raise ValueError(
            "Please upload a valid ultrasound image."
        ) from exc

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
            raise RuntimeError(
                "Could not calculate Grad-CAM gradients."
            )

        weights = captured["gradient"].detach().mean(
            dim=(2, 3),
            keepdim=True,
        )

        cam = torch.relu(
            (
                weights * captured["activation"].detach()
            ).sum(dim=1, keepdim=True)
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

    # Create the Grad-CAM overlay in memory.
    original = np.asarray(display_image).astype(np.float32) / 255.0
    overlay = original.copy()

    overlay[:, :, 0] = np.maximum(
        overlay[:, :, 0],
        heatmap * 0.7,
    )
    overlay[:, :, 1] *= 1 - heatmap * 0.45
    overlay[:, :, 2] *= 1 - heatmap * 0.45

    overlay = np.clip(overlay, 0, 1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    buffer = io.BytesIO()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))

    axes[0].imshow(original)
    axes[0].set_title("Uploaded ultrasound")
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM explanation")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=120)
    plt.close(fig)

    buffer.seek(0)

    return {
        "predicted_class": CLASSES[predicted_idx],
        "confidence": round(confidence, 4),
        "classes": CLASSES,
        "probabilities": {
            CLASSES[i]: round(float(probabilities[i].item()), 4)
            for i in range(len(CLASSES))
        },
        "disclaimer": (
            "Research prototype only. This prediction is not a "
            "diagnosis and must not be used for clinical decisions."
        ),
        "gradcam_png_base64": base64.b64encode(
            buffer.getvalue()
        ).decode("ascii"),
    }
