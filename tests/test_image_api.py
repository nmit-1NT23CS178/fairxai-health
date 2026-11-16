"""Image module tests (TB X-ray + breast ultrasound).

These use the real checkpoints committed under artifacts/ - not synthetic model files - and verify
that class names come from shipped metadata rather than data/raw (which is absent in production).
"""
import io

import pytest
from PIL import Image


def _png(size=(224, 224)):
    buf = io.BytesIO()
    Image.new("RGB", size, (140, 120, 110)).save(buf, format="PNG")
    return buf.getvalue()


def test_models_are_reported_by_health(client):
    h = client.get("/api/health").get_json()["data"]
    assert h["image_models"]["tb_image"] is True
    assert h["image_models"]["breast_ultrasound"] is True
    assert h["image_models_loaded"] == 2


def test_tb_image_prediction_without_raw_dataset(client):
    r = client.post("/api/predict/tb-image", data={"file": (io.BytesIO(_png()), "xray.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    d = r.get_json()["data"]
    assert d["predicted_class"] in {"health", "sick", "tb"}
    assert d["classes"] == ["health", "sick", "tb"]
    assert 0.0 <= d["confidence"] <= 1.0
    assert d["gradcam_png_base64"]


def test_breast_ultrasound_prediction(client):
    r = client.post("/api/predict/breast-ultrasound", data={"file": (io.BytesIO(_png()), "bus.png")},
                    content_type="multipart/form-data")
    assert r.status_code == 200, r.get_json()
    d = r.get_json()["data"]
    assert d["predicted_class"] in {"benign", "malignant", "normal"}
    assert sum(d["probabilities"].values()) == pytest.approx(1.0, abs=1e-3)
    assert d["gradcam_png_base64"]


def test_image_endpoints_validate_upload(client):
    bad_type = client.post("/api/predict/tb-image", data={"file": (io.BytesIO(b"x"), "evil.exe")},
                           content_type="multipart/form-data")
    assert bad_type.status_code == 415

    missing = client.post("/api/predict/tb-image", data={}, content_type="multipart/form-data")
    assert missing.status_code == 400

    not_an_image = client.post("/api/predict/tb-image", data={"file": (io.BytesIO(b"not an image"), "x.png")},
                               content_type="multipart/form-data")
    assert not_an_image.status_code == 422
    assert not_an_image.get_json()["error"]["code"] == "invalid_image"


def test_image_endpoints_refuse_to_oom_a_small_instance(client, monkeypatch):
    """On a 512 MB instance the image models would exceed the memory limit; the endpoint must return a
    clear 503 instead of crashing the worker."""
    from backend.app.api import routes

    monkeypatch.setattr(routes, "_image_capacity", lambda: (False, 512))
    for path in ("/api/predict/tb-image", "/api/predict/breast-ultrasound"):
        r = client.post(path, data={"file": (io.BytesIO(_png()), "x.png")},
                        content_type="multipart/form-data")
        assert r.status_code == 503
        err = r.get_json()["error"]
        assert err["code"] == "image_capacity"
        assert err["limit_mb"] == 512
        assert err["required_mb"] >= 900
