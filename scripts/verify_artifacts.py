"""Validate every shipped model artifact end-to-end WITHOUT retraining (the same checks the
deployed service relies on). Run from the repo root:

    python scripts/verify_artifacts.py

Covers the four tabular disease models and, when PyTorch is installed, both image models
(TB X-ray 3-class and breast ultrasound 3-class) including their Grad-CAM output.
Exits non-zero if any artifact cannot be loaded or produce a prediction + explanations.
"""
import io
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from backend.ml import inference as inf  # noqa: E402
from backend.ml.diseases import DISEASES  # noqa: E402


def defaults(schema):
    out = {}
    for f in schema:
        if f["type"] == "number":
            out[f["name"]] = f.get("median", f.get("min_seen", 0))
        elif f["type"] == "category":
            out[f["name"]] = f.get("mode") or f["options"][0]
        else:
            out[f["name"]] = f["options"][0]
    return out


def check_tabular():
    failures, checked = [], 0
    for d in DISEASES:
        version = inf.latest_version(d)
        if not version:
            print(f"[SKIP] {d}: no trained artifact")
            continue
        checked += 1
        try:
            meta = inf.load_meta(d, version)
            res = inf.predict(d, defaults(meta["feature_schema"]), version=version, explain=True, lime=True)
            assert 0.0 <= res["probability"] <= 1.0
            assert res["explanations"]["shap"].get("contributions"), "SHAP missing"
            assert res["explanations"]["lime"].get("items"), "LIME missing"
            assert meta["test_metrics"]["roc_auc"] is not None
            print(f"[PASS] {d:14s} {meta['algorithm']:32s} p={res['probability']:.4f} "
                  f"thr={meta['threshold']:.4f} shap={len(res['explanations']['shap']['contributions'])} "
                  f"lime={len(res['explanations']['lime']['items'])}")
        except Exception as e:  # noqa: BLE001
            failures.append(d)
            print(f"[FAIL] {d}: {type(e).__name__}: {e}")
    return checked, failures


def check_images():
    try:
        from PIL import Image
        from backend.ml import tb_inference, breast_ultrasound_inference
    except Exception as e:  # noqa: BLE001
        print(f"[SKIP] image models: {type(e).__name__}: {e}")
        return 0, []

    buf = io.BytesIO()
    Image.new("RGB", (224, 224), (140, 120, 110)).save(buf, format="PNG")
    img = buf.getvalue()
    failures, checked = [], 0
    checks = [
        ("tb_image", tb_inference.predict_tb_image, {"health", "sick", "tb"}),
        ("breast_ultrasound", breast_ultrasound_inference.predict_breast_ultrasound,
         {"benign", "malignant", "normal"}),
    ]
    for name, fn, classes in checks:
        checked += 1
        try:
            out = fn(img)
            assert out["predicted_class"] in classes
            assert 0.0 <= out["confidence"] <= 1.0
            assert out["gradcam_png_base64"], "Grad-CAM missing"
            print(f"[PASS] {name:18s} class={out['predicted_class']:10s} "
                  f"conf={out['confidence']:.4f} gradcam=yes")
        except Exception as e:  # noqa: BLE001
            failures.append(name)
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")
    return checked, failures


def main():
    t_checked, t_fail = check_tabular()
    i_checked, i_fail = check_images()
    total, failed = t_checked + i_checked, len(t_fail) + len(i_fail)
    if total == 0:
        print("No artifacts found.")
        return 1
    print(f"\n{total - failed}/{total} shipped models verified.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
