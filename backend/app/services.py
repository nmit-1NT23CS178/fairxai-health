"""Service layer: registry sync, dashboard aggregation, background training, prediction persistence."""
from __future__ import annotations
import datetime as dt, threading, traceback, uuid, math
from pathlib import Path
from ..ml.common import REPORT_DIR, RAW_DIR, PROC_DIR, jload
from ..ml.diseases import DISEASES
from ..ml import inference as inf
from ..ml.data_prep import discover

now = lambda: dt.datetime.utcnow()
_runs: dict = {}; _run_lock = threading.Lock(); _training = set()


def _lean_metrics(m):
    return {k: v for k, v in (m or {}).items() if k not in ("roc_curve", "pr_curve", "calibration_curve")}


def sync_artifacts(db, cfg=None):
    """Mirror artifact metadata into MongoDB (binaries stay on disk; only references are stored)."""
    if not db.ok: return
    for d in DISEASES:
        for v in inf.list_versions(d):
            m = inf.load_meta(d, v)
            doc = dict(disease=d, version=v, algorithm=m["algorithm"], artifact_path=f"artifacts/{d}/{v}/model.joblib",
                       feature_schema=m["feature_schema"], validation_metrics=_lean_metrics(m["validation_metrics"]),
                       test_metrics=_lean_metrics(m["test_metrics"]), threshold=m["threshold"], trained_at=m["trained_at"],
                       dataset=m["dataset"], dataset_rows=m["dataset_rows"], status="active" if v == inf.latest_version(d) else "archived",
                       candidates=[dict(model=c["model"], cv_roc_auc_mean=c["cv_roc_auc_mean"], val_roc_auc=c["val_metrics"]["roc_auc"], train_time_s=c["train_time_s"], selected=c["selected"]) for c in m["candidates"]])
            db.col("model_registry").update_one({"disease": d, "version": v}, {"$set": doc}, upsert=True)
            f = m.get("fairness", {})
            db.col("fairness_audits").update_one({"disease": d, "model_version": v}, {"$set": dict(disease=d, model_version=v, dataset_version=m["dataset"], audited_at=m["trained_at"],
                    status=f.get("status"), findings=f.get("findings", []), limitations=f.get("limitations", []),
                    subgroup_sizes={a["attribute"]: {g["group"]: g["n"] for g in a["groups"]} for a in f.get("attributes", [])},
                    metrics={a["attribute"]: a.get("disparities") for a in f.get("attributes", [])})}, upsert=True)
        rp = REPORT_DIR / f"{d}_quality.json"
        if rp.exists():
            r = jload(rp)
            db.col("datasets").update_one({"disease": d}, {"$set": dict(disease=d, file=r["source_file"], schema=dict(numeric=r["numeric_features"], categorical=r["categorical_features"], target=r["target_column"]),
                    rows=r["cleaned_rows"], columns=r["cleaned_columns"], original_rows=r["original_rows"], quality_report_ref=f"data/reports/{d}_quality.json",
                    processed_at=r["generated_at"], status="processed")}, upsert=True)


def disease_status(d):
    spec = DISEASES[d]; info = discover(d); ver = inf.latest_version(d)
    out = dict(key=d, name=spec["name"], short=spec["short"], dataset_status=info["status"], dataset_file=Path(info["path"]).name if info.get("path") else None)
    if ver:
        m = inf.load_meta(d, ver); t = m["test_metrics"]
        out.update(status="trained", version=ver, algorithm=m["algorithm"], trained_at=m["trained_at"], threshold=m["threshold"],
                   test={k: t.get(k) for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier", "n")},
                   validation={k: m["validation_metrics"].get(k) for k in ("roc_auc", "recall", "pr_auc")},
                   top_features=[i["feature"] for i in m["global_explanation"].get("importance", [])[:3]],
                   fairness_status=m["fairness"]["status"], n_versions=len(inf.list_versions(d)))
    elif info["status"] == "tabular": out.update(status="untrained", message="Dataset found; training not yet run.")
    elif info["status"] == "image_only": out.update(status="image_pipeline_required", message=f"Only {info['n_images']} image files found; a CNN image pipeline is required (not part of the tabular trainer).")
    else: out.update(status="dataset_missing", message=f"No dataset found in data/raw/{d}/. Add a CSV/XLSX and train.")
    return out


def start_training(db, disease, search_iter=8):
    with _run_lock:
        if disease in _training: raise RuntimeError("Training already running for this disease.")
        _training.add(disease)
    run_id = uuid.uuid4().hex[:12]
    rec = dict(run_id=run_id, disease=disease, status="running", started_at=now().isoformat() + "Z", progress=[], error=None)
    _runs[run_id] = rec
    if db.ok: db.col("training_runs").insert_one(dict(rec))

    def work():
        from ..ml.train import train_disease
        def prog(msg):
            rec["progress"].append(f"{now().strftime('%H:%M:%S')} {msg}")
        try:
            m = train_disease(disease, progress=prog, search_iter=search_iter)
            inf.clear_cache(); sync_artifacts(db)
            rec.update(status="completed", version=m["version"], algorithm=m["algorithm"], candidates=[dict(model=c["model"], val_roc_auc=c["val_metrics"]["roc_auc"]) for c in m["candidates"]])
        except Exception as e:
            traceback.print_exc(); rec.update(status="failed", error=f"{type(e).__name__}: {str(e)[:300]}")
        finally:
            rec["finished_at"] = now().isoformat() + "Z"
            with _run_lock: _training.discard(disease)
            if db.ok: db.col("training_runs").update_one({"run_id": run_id}, {"$set": {k: v for k, v in rec.items() if k != "run_id"}})
    threading.Thread(target=work, daemon=True).start()
    return rec


def list_runs(db, limit=20):
    if db.ok: return list(db.col("training_runs").find({}, {"_id": 0}).sort("started_at", -1).limit(limit))
    return list(_runs.values())[-limit:]


def save_prediction(db, cfg, res):
    pid = uuid.uuid4().hex
    created = now()
    if db.ok:
        doc = dict(prediction_id=pid, created_at=created, expires_at=created + dt.timedelta(days=cfg.PREDICTION_RETENTION_DAYS),
                   disease=res["disease"], model_version=res["model_version"], algorithm=res["algorithm"], probability=res["probability"],
                   threshold=res["threshold"], flagged=res["flagged"], status="completed", warnings=res["warnings"],
                   explanation_meta=dict(shap_top=[(c["feature"], round(c["shap"], 4)) for c in res["explanations"].get("shap", {}).get("contributions", [])[:5]],
                                         lime_available="items" in res["explanations"].get("lime", {})))
        if cfg.STORE_PREDICTION_INPUTS: doc["inputs"] = res["_validated_inputs"]
        db.col("predictions").insert_one(doc)
    return pid


def query_predictions(db, disease=None, date_from=None, date_to=None, version=None, status=None, flagged=None, limit=200):
    if not db.ok: return []
    q = {}
    if disease: q["disease"] = disease
    if version: q["model_version"] = version
    if status: q["status"] = status
    if flagged in ("true", "false"): q["flagged"] = flagged == "true"
    if date_from or date_to:
        q["created_at"] = {}
        if date_from: q["created_at"]["$gte"] = dt.datetime.fromisoformat(date_from)
        if date_to: q["created_at"]["$lt"] = dt.datetime.fromisoformat(date_to) + dt.timedelta(days=1)
    rows = list(db.col("predictions").find(q, {"_id": 0}).sort("created_at", -1).limit(min(int(limit), 1000)))
    for r in rows:
        r["created_at"] = r["created_at"].isoformat() + "Z"; r.pop("expires_at", None)
    return rows
