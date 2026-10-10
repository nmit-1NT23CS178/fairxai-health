"""Inference: loads saved pipeline artifacts (never retrains), validates input against the disease's own
feature schema, applies the identical fitted preprocessing, and returns calibrated score + explanations."""
from __future__ import annotations
import math, threading, traceback
import numpy as np, pandas as pd, joblib
from .common import ARTIFACT_DIR, DISCLAIMER, jload
from .diseases import DISEASES
from . import explain as xai
from .train import apply_calibrator

_lock = threading.Lock(); _cache: dict = {}


class ModelUnavailable(Exception): pass
class ValidationFailed(Exception):
    def __init__(self, errors, warnings=None): self.errors, self.warnings = errors, warnings or []; super().__init__("; ".join(errors))


def list_versions(disease):
    d = ARTIFACT_DIR / disease
    return sorted([p.name for p in d.iterdir() if (p / "metadata.json").exists()]) if d.exists() else []


def latest_version(disease):
    f = ARTIFACT_DIR / disease / "LATEST.json"
    if f.exists():
        v = jload(f)["latest"]
        if (ARTIFACT_DIR / disease / v / "metadata.json").exists(): return v
    vs = list_versions(disease); return vs[-1] if vs else None


def load_meta(disease, version=None):
    version = version or latest_version(disease)
    if not version: raise ModelUnavailable(f"No trained model for '{disease}'.")
    p = ARTIFACT_DIR / disease / version / "metadata.json"
    if not p.exists(): raise ModelUnavailable(f"Model version {version} not found for '{disease}'.")
    return jload(p)


def load_model(disease, version=None):
    version = version or latest_version(disease)
    if not version: raise ModelUnavailable(f"No trained model for '{disease}'.")
    key = (disease, version)
    with _lock:
        if key not in _cache:
            p = ARTIFACT_DIR / disease / version / "model.joblib"
            if not p.exists(): raise ModelUnavailable(f"Model artifact missing: {disease}/{version}")
            try: _cache[key] = (joblib.load(p), jload(p.parent / "metadata.json"))
            except Exception as e: raise ModelUnavailable(f"Model artifact could not be loaded: {type(e).__name__}")
        return _cache[key]


def clear_cache():
    with _lock: _cache.clear(); xai._cache.clear()


def validate_inputs(schema, inputs: dict):
    errors, warnings, row = [], [], {}
    if not isinstance(inputs, dict): raise ValidationFailed(["'inputs' must be an object"])
    names = {f["name"] for f in schema}
    extra = [k for k in inputs if k not in names]
    if extra: warnings.append(f"Ignored unknown fields: {', '.join(map(str, extra[:5]))}")
    for f in schema:
        n, v = f["name"], inputs.get(f["name"])
        if v is None or (isinstance(v, str) and v.strip() == ""):
            if f.get("allow_missing"): row[n] = np.nan; warnings.append(f"{f['label']} not provided: filled with the training-data median/mode by the model pipeline.")
            else: errors.append(f"{f['label']} is required.")
            continue
        if f["type"] in ("number", "binary"):
            if isinstance(v, bool): errors.append(f"{f['label']} must be numeric."); continue
            try: x = float(v)
            except (TypeError, ValueError): errors.append(f"{f['label']} must be a number."); continue
            if not math.isfinite(x): errors.append(f"{f['label']} must be finite."); continue
            if f["type"] == "binary":
                if x not in [float(o) for o in f["options"]]: errors.append(f"{f['label']} must be one of {f['options']}."); continue
            else:
                if f.get("hard_min") is not None and not (f["hard_min"] <= x <= f["hard_max"]):
                    errors.append(f"{f['label']} must be between {f['hard_min']} and {f['hard_max']}."); continue
                span = f["max_seen"] - f["min_seen"]
                if span > 0 and not (f["min_seen"] - .5 * span <= x <= f["max_seen"] + .5 * span):
                    warnings.append(f"{f['label']}={x:g} is far outside the training range ({f['min_seen']:g}-{f['max_seen']:g}); the estimate may be unreliable.")
            row[n] = x
        else:
            s = str(v).strip(); opts = f["options"]
            m = next((o for o in opts if o == s), None) or next((o for o in opts if o.lower() == s.lower()), None)
            if m is None: errors.append(f"{f['label']} must be one of {opts}."); continue
            row[n] = m
    if errors: raise ValidationFailed(errors, warnings)
    return row, warnings


def predict(disease, inputs, version=None, explain=True, lime=False):
    bundle, meta = load_model(disease, version)
    row, warnings = validate_inputs(meta["feature_schema"], inputs)
    df = pd.DataFrame([row])[bundle["columns"]]
    raw = float(bundle["pipeline"].predict_proba(df)[0, 1])
    p = float(apply_calibrator(bundle["calibrator"], [raw])[0]); thr = meta["threshold"]
    flagged = p >= thr; t = meta["test_metrics"]
    near = abs(p - thr) < 0.1
    res = dict(disease=disease, disease_name=meta["disease_name"], model_version=meta["version"], algorithm=meta["algorithm"],
               probability=p, threshold=thr, threshold_rule=meta["threshold_rule"], flagged=bool(flagged),
               label=("Model score above screening threshold" if flagged else "Model score below screening threshold"),
               confidence=dict(calibration="Platt-scaled on validation data", test_brier=t["brier"], test_roc_auc=t["roc_auc"],
                               test_recall=t["recall"], test_specificity=t["specificity"], test_n=t["n"],
                               near_threshold=bool(near),
                               note=("Score is close to the decision threshold - treat as uncertain." if near else "Score is not near the decision threshold.")),
               warnings=warnings, limitations=meta["limitations"][:4], disclaimer=DISCLAIMER, explanations={})
    if explain:
        try: res["explanations"]["shap"] = xai.local_shap(bundle, df, key=(disease, meta["version"]))
        except Exception as e:
            traceback.print_exc(); res["explanations"]["shap"] = dict(error=f"SHAP explanation unavailable: {type(e).__name__}")
    if lime:
        try: res["explanations"]["lime"] = xai.local_lime(bundle, df)
        except Exception as e:
            traceback.print_exc(); res["explanations"]["lime"] = dict(error=f"LIME explanation unavailable: {type(e).__name__}")
    res["_validated_inputs"] = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()}
    return res
