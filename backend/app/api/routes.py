"""REST API (all responses: {ok: bool, data|error})."""
from __future__ import annotations
import datetime as dt, hmac, sys, platform
from functools import wraps
from pathlib import Path
from flask import Blueprint, jsonify, request, current_app as app, Response
from werkzeug.utils import secure_filename
from ...ml import inference as inf
from ...ml.common import REPORT_DIR, RAW_DIR, PROC_DIR, ARTIFACT_DIR, DISCLAIMER, jload
from ...ml.diseases import DISEASES
from ...ml.data_prep import discover, clean_dataset, DatasetError, TABULAR_EXT
from .. import services

bp = Blueprint("api", __name__)
ok = lambda data=None, code=200: (jsonify(ok=True, data=data), code)
err = lambda code, msg, http=400, **x: (jsonify(ok=False, error=dict(code=code, message=msg, **x)), http)


def admin(fn):
    @wraps(fn)
    def w(*a, **k):
        tok = app.cfg.ADMIN_TOKEN
        if not tok:
            if app.cfg.PRODUCTION: return err("admin_disabled", "ADMIN_TOKEN is not configured on the server.", 503)
        elif not hmac.compare_digest(request.headers.get("X-Admin-Token", ""), tok):
            return err("unauthorized", "Admin token required.", 401)
        return fn(*a, **k)
    return w


def _disease(d):
    if d not in DISEASES: raise KeyError(d)
    return d


@bp.errorhandler(KeyError)
def _unknown(e): return err("unknown_disease", f"Unknown disease {e}. Valid: {', '.join(DISEASES)}", 404)


@bp.errorhandler(inf.ModelUnavailable)
def _mu(e): return err("model_unavailable", str(e), 404)


@bp.errorhandler(inf.ValidationFailed)
def _vf(e): return err("validation_error", "Invalid input", 422, details=e.errors, warnings=e.warnings)


@bp.errorhandler(DatasetError)
def _de(e): return err("dataset_error", str(e), 422)


@bp.get("/health")
def health():
    db = app.db
    return ok(dict(status="ok", database=dict(mode=db.mode, connected=db.ok, persistent=db.mode == "mongodb", note=db.error),
                   models_loaded=sum(1 for d in DISEASES if inf.latest_version(d)), time=dt.datetime.utcnow().isoformat() + "Z"))


@bp.get("/system")
def system():
    import sklearn, shap, xgboost, fairlearn, lime, pandas
    return ok(dict(python=platform.python_version(), versions=dict(scikit_learn=sklearn.__version__, xgboost=xgboost.__version__, shap=shap.__version__, fairlearn=fairlearn.__version__, pandas=pandas.__version__),
                   database=dict(mode=app.db.mode, note=app.db.error), retention_days=app.cfg.PREDICTION_RETENTION_DAYS, store_inputs=app.cfg.STORE_PREDICTION_INPUTS,
                   admin_protected=bool(app.cfg.ADMIN_TOKEN), max_upload_mb=app.cfg.MAX_CONTENT_LENGTH // 1048576, environment=app.cfg.ENV, disclaimer=DISCLAIMER))


@bp.get("/diseases")
def diseases(): return ok([services.disease_status(d) for d in DISEASES])


@bp.get("/dashboard/summary")
def dashboard():
    ds = [services.disease_status(d) for d in DISEASES]
    trained = [d for d in ds if d["status"] == "trained"]
    quality = {}
    for d in DISEASES:
        p = REPORT_DIR / f"{d}_quality.json"
        if p.exists():
            r = jload(p); quality[d] = {k: r[k] for k in ("original_rows", "cleaned_rows", "original_columns", "cleaned_columns", "duplicate_rows_removed", "imbalance_ratio", "target_distribution", "generated_at")}
            quality[d]["missing_cells_before"] = sum(r["missing_before"].values()); quality[d]["missing_cells_after"] = sum(r["missing_after"].values())
    pred_stats = _pred_stats()
    return ok(dict(diseases=ds, supported=len(DISEASES), trained=len(trained), last_trained=max([d["trained_at"] for d in trained], default=None),
                   quality=quality, predictions=pred_stats, runs=services.list_runs(app.db, 5), database=dict(mode=app.db.mode, persistent=app.db.mode == "mongodb"),
                   fairness={d["key"]: d["fairness_status"] for d in trained}, disclaimer=DISCLAIMER))


@bp.get("/models")
def models():
    d, st = request.args.get("disease"), request.args.get("status")
    if app.db.ok:
        q = {k: v for k, v in dict(disease=d, status=st).items() if v}
        rows = list(app.db.col("model_registry").find(q, {"_id": 0, "feature_schema": 0}).sort("trained_at", -1))
    else:
        rows = [dict(disease=x, version=v) for x in DISEASES for v in inf.list_versions(x)]
    return ok(rows)


@bp.get("/models/<disease>/metrics")
def metrics(disease):
    m = inf.load_meta(_disease(disease), request.args.get("version"))
    keys = ("disease", "disease_name", "version", "algorithm", "trained_at", "dataset", "dataset_rows", "split", "candidates", "selection_reason", "threshold", "threshold_rule",
            "validation_metrics", "test_metrics", "calibration", "class_balance_method", "training_seconds", "limitations", "error_costs", "seed", "target_column", "target_encoding")
    out = {k: m[k] for k in keys if k in m}; out["versions"] = inf.list_versions(disease)
    return ok(out)


@bp.get("/models/<disease>/schema")
def schema(disease):
    m = inf.load_meta(_disease(disease), request.args.get("version"))
    return ok(dict(version=m["version"], fields=m["feature_schema"], disclaimer=DISCLAIMER))


@bp.post("/predict/<disease>")
def predict(disease):
    _disease(disease)
    body = request.get_json(silent=True)
    if not isinstance(body, dict) or "inputs" not in body: return err("bad_request", "JSON body with an 'inputs' object is required.")
    res = inf.predict(disease, body["inputs"], version=body.get("model_version"), explain=body.get("explain", True), lime=bool(body.get("lime", False)))
    try: res["prediction_id"] = services.save_prediction(app.db, app.cfg, res)
    except Exception as e:
        app.logger.error("prediction not persisted: %s", type(e).__name__); res["prediction_id"] = None; res["warnings"].append("Result could not be saved to history.")
    res.pop("_validated_inputs", None)
    return ok(res)

@bp.post("/predict/tb-image")
def predict_tb_image_route():
    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        return err(
            "bad_request",
            "Upload a chest X-ray image using the 'file' field."
        )

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp"}
    if Path(uploaded.filename).suffix.lower() not in allowed_extensions:
        return err(
            "bad_file_type",
            "Allowed image types: PNG, JPG, JPEG, WEBP.",
            415
        )

    try:
        image_bytes = uploaded.read()
        if not image_bytes:
            return err("empty_file", "The uploaded image is empty.")

        from ...ml.tb_inference import predict_tb_image

        result = predict_tb_image(image_bytes)
        return ok(result)

    except ValueError as exc:
        return err("invalid_image", str(exc), 422)
    except Exception:
        app.logger.exception("TB image prediction failed")
        return err(
            "prediction_failed",
            "Could not process this image. Check the server logs.",
            500
        )
@bp.post("/predict/breast-ultrasound")
def predict_breast_ultrasound_route():
    uploaded = request.files.get("file")

    if uploaded is None or not uploaded.filename:
        return err(
            "bad_request",
            "Upload a breast ultrasound image using the 'file' field."
        )

    allowed_extensions = {".png", ".jpg", ".jpeg", ".webp"}

    if Path(uploaded.filename).suffix.lower() not in allowed_extensions:
        return err(
            "bad_file_type",
            "Allowed image types: PNG, JPG, JPEG, WEBP.",
            415
        )

    try:
        image_bytes = uploaded.read()

        if not image_bytes:
            return err("empty_file", "The uploaded image is empty.")

        from ...ml.breast_ultrasound_inference import (
            predict_breast_ultrasound,
        )

        result = predict_breast_ultrasound(image_bytes)
        return ok(result)

    except ValueError as exc:
        return err("invalid_image", str(exc), 422)

    except Exception:
        app.logger.exception("Breast ultrasound prediction failed")
        return err(
            "prediction_failed",
            "Could not process this image. Check the server logs.",
            500
        )

@bp.get("/explain/<disease>/global")
def explain_global(disease):
    m = inf.load_meta(_disease(disease), request.args.get("version"))
    return ok(dict(version=m["version"], algorithm=m["algorithm"], explanation=m["global_explanation"]))


@bp.get("/fairness/<disease>")
def fairness(disease):
    m = inf.load_meta(_disease(disease), request.args.get("version"))
    f = dict(m["fairness"]); f["model_version"] = m["version"]; f["threshold"] = m["threshold"]
    sg = request.args.get("subgroup")
    if sg: f["attributes"] = [a for a in f.get("attributes", []) if a["attribute"] == sg]
    return ok(f)


@bp.get("/datasets")
def datasets():
    out = []
    for d in DISEASES:
        i = discover(d); r = REPORT_DIR / f"{d}_quality.json"
        out.append(dict(disease=d, name=DISEASES[d]["name"], status=i["status"], file=Path(i["path"]).name if i.get("path") else None, has_report=r.exists()))
    return ok(out)


@bp.get("/datasets/<disease>/report")
def ds_report(disease):
    p = REPORT_DIR / f"{_disease(disease)}_quality.json"
    if not p.exists():
        return err("no_report", services.disease_status(disease).get("message") or "Dataset has not been cleaned yet. Run cleaning first.", 404)
    return ok(jload(p))


@bp.post("/datasets/<disease>/upload")
@admin
def ds_upload(disease):
    _disease(disease); f = request.files.get("file")
    if not f or not f.filename: return err("bad_request", "Attach a file in form field 'file'.")
    name = secure_filename(f.filename)
    if Path(name).suffix.lower() not in TABULAR_EXT: return err("bad_type", f"Allowed types: {', '.join(sorted(TABULAR_EXT))}", 415)
    dest = RAW_DIR / disease; dest.mkdir(parents=True, exist_ok=True)
    for old in dest.glob("*"):  # keep exactly one active raw file; previous remains in git history/backups
        if old.suffix.lower() in TABULAR_EXT: old.rename(old.with_name(old.name + ".replaced"))
    f.save(dest / name)
    return ok(dict(saved=name, next="POST /api/datasets/%s/clean" % disease), 201)


@bp.post("/datasets/<disease>/clean")
@admin
def ds_clean(disease):
    r = clean_dataset(_disease(disease)); services.sync_artifacts(app.db)
    if app.db.ok:
        app.db.col("datasets").update_one({"disease": disease}, {"$set": dict(disease=disease, file=r["source_file"], rows=r["cleaned_rows"], columns=r["cleaned_columns"], original_rows=r["original_rows"],
                                           quality_report_ref=f"data/reports/{disease}_quality.json", processed_at=r["generated_at"], status="processed")}, upsert=True)
    return ok({k: r[k] for k in ("original_rows", "cleaned_rows", "target_column", "target_distribution", "limitations")})


@bp.post("/train/<disease>")
@admin
def train(disease):
    if discover(_disease(disease))["status"] != "tabular": return err("dataset_missing", services.disease_status(disease).get("message"), 422)
    try: return ok(services.start_training(app.db, disease), 202)
    except RuntimeError as e: return err("conflict", str(e), 409)


@bp.get("/train/runs")
def runs(): return ok(services.list_runs(app.db, int(request.args.get("limit", 20))))


def _pred_stats():
    if not app.db.ok: return dict(total=0, by_disease={}, daily=[], flagged=0)
    c = app.db.col("predictions"); rows = list(c.find({}, {"_id": 0, "disease": 1, "created_at": 1, "flagged": 1}))
    by, daily = {}, {}
    for r in rows:
        by[r["disease"]] = by.get(r["disease"], 0) + 1; k = r["created_at"].strftime("%Y-%m-%d"); daily[k] = daily.get(k, 0) + 1
    return dict(total=len(rows), by_disease=by, daily=[dict(date=k, count=v) for k, v in sorted(daily.items())], flagged=sum(1 for r in rows if r.get("flagged")))


@bp.get("/predictions")
def predictions():
    a = request.args
    try: rows = services.query_predictions(app.db, a.get("disease"), a.get("from"), a.get("to"), a.get("model_version"), a.get("status"), a.get("flagged"), a.get("limit", 200))
    except ValueError: return err("bad_request", "Dates must be YYYY-MM-DD.")
    daily = {}
    for r in rows: daily[r["created_at"][:10]] = daily.get(r["created_at"][:10], 0) + 1
    return ok(dict(items=rows, count=len(rows), daily=[dict(date=k, count=v) for k, v in sorted(daily.items())], database_mode=app.db.mode))


@bp.delete("/predictions/<pid>")
@admin
def del_pred(pid):
    if not app.db.ok: return err("db_unavailable", "Database unavailable.", 503)
    r = app.db.col("predictions").delete_one({"prediction_id": pid})
    return ok(dict(deleted=r.deleted_count)) if r.deleted_count else err("not_found", "No such prediction.", 404)


@bp.delete("/predictions")
@admin
def purge():
    if not app.db.ok: return err("db_unavailable", "Database unavailable.", 503)
    days = int(request.args.get("older_than_days", app.cfg.PREDICTION_RETENTION_DAYS))
    r = app.db.col("predictions").delete_many({"created_at": {"$lt": dt.datetime.utcnow() - dt.timedelta(days=days)}})
    return ok(dict(deleted=r.deleted_count, older_than_days=days))


@bp.get("/reports/<disease>")
def report(disease):
    from ...ml.report import build_markdown
    md = build_markdown(_disease(disease), request.args.get("version"))
    if request.args.get("format") == "json": return ok(dict(markdown=md))
    return Response(md, mimetype="text/markdown", headers={"Content-Disposition": f"attachment; filename=fairxai_{disease}_report.md"})
