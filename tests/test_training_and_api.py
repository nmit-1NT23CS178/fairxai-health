import time, json
import numpy as np, pandas as pd, joblib
from backend.ml.common import ARTIFACT_DIR
from backend.ml import inference as inf


def test_training_artifacts_and_leakage_safe_structure(trained_app):
    v = inf.latest_version("diabetes"); meta = inf.load_meta("diabetes")
    assert v and meta["split"]["train"] + meta["split"]["validation"] + meta["split"]["test"] == meta["dataset_rows"]
    assert abs(meta["split"]["test"] / meta["dataset_rows"] - 0.2) < 0.02
    b = joblib.load(ARTIFACT_DIR / "diabetes" / v / "model.joblib")
    prep = b["pipeline"].named_steps["prep"]
    assert hasattr(prep, "transformers_") and "imp" in dict(prep.named_transformers_["num"].steps)   # imputation inside the pipeline
    assert "selection_reason" in meta and "Selection never used test data" in meta["selection_reason"]
    assert meta["test_metrics"]["n"] == meta["split"]["test"] and meta["validation_metrics"]["n"] == meta["split"]["validation"]
    assert {c["model"] for c in meta["candidates"]} >= {"Logistic Regression (baseline)", "Random Forest", "XGBoost"}


def test_serialization_roundtrip_matches_inference(trained_app):
    inf.clear_cache(); b, m = inf.load_model("diabetes")
    row = {f["name"]: (f["median"] if f["type"] == "number" else f["mode"] if f["type"] == "category" else f["options"][0]) for f in m["feature_schema"]}
    r1 = inf.predict("diabetes", row, explain=False)["probability"]; inf.clear_cache(); r2 = inf.predict("diabetes", row, explain=False)["probability"]
    assert r1 == r2 and 0 <= r1 <= 1


def sample_inputs(client):
    f = client.get("/api/models/diabetes/schema").get_json()["data"]["fields"]
    return {x["name"]: (x["median"] if x["type"] == "number" else x["mode"] if x["type"] == "category" else x["options"][0]) for x in f}


def test_health_and_diseases(client):
    h = client.get("/api/health").get_json()["data"]; assert h["status"] == "ok" and h["database"]["mode"] in ("memory", "mongodb")
    ds = {d["key"]: d for d in client.get("/api/diseases").get_json()["data"]}
    assert ds["diabetes"]["status"] == "trained" and ds["tb"]["status"] == "dataset_missing" and set(ds) == {"cvd", "diabetes", "breast_cancer", "ckd", "tb"}


def test_schema_is_disease_specific(client):
    names = [f["name"] for f in client.get("/api/models/diabetes/schema").get_json()["data"]["fields"]]
    assert "glucose" in names and "patient_id" not in names and "outcome" not in names


def test_predict_with_shap_and_lime_and_persistence(client):
    r = client.post("/api/predict/diabetes", json=dict(inputs=sample_inputs(client), lime=True)); d = r.get_json()["data"]
    assert r.status_code == 200 and 0 <= d["probability"] <= 1 and d["disclaimer"] and d["prediction_id"]
    assert d["explanations"]["shap"]["contributions"] and d["explanations"]["lime"]["items"]
    assert "_validated_inputs" not in d
    hist = client.get("/api/predictions?disease=diabetes").get_json()["data"]
    assert hist["count"] >= 1 and hist["items"][0]["prediction_id"] == d["prediction_id"]
    assert client.get("/api/predictions?disease=ckd").get_json()["data"]["count"] == 0
    assert client.get("/api/predictions?from=2999-01-01").get_json()["data"]["count"] == 0
    assert client.get("/api/predictions?from=bad").status_code == 400


def test_invalid_and_incomplete_inputs(client):
    i = sample_inputs(client); bad = dict(i, glucose=-5); r = client.post("/api/predict/diabetes", json=dict(inputs=bad))
    assert r.status_code == 422 and r.get_json()["error"]["code"] == "validation_error"
    inc = {k: v for k, v in i.items() if k != "bmi"}; assert client.post("/api/predict/diabetes", json=dict(inputs=inc)).status_code in (200, 422)
    assert client.post("/api/predict/diabetes", json=dict(inputs=dict(i, bmi="abc"))).status_code == 422
    assert client.post("/api/predict/diabetes", json=dict(inputs=dict(i, bmi=float("nan")))).status_code in (422, 400)
    assert client.post("/api/predict/diabetes", data="not json", content_type="application/json").status_code == 400
    assert client.post("/api/predict/diabetes", json=dict(inputs=dict(i, sex="X"))).status_code == 422


def test_model_unavailable_and_unknown_disease(client):
    r = client.post("/api/predict/tb", json=dict(inputs={})); assert r.status_code == 404 and r.get_json()["error"]["code"] == "model_unavailable"
    assert client.get("/api/models/nope/metrics").status_code == 404
    assert client.get("/api/models/tb/metrics").status_code == 404


def test_metrics_explain_fairness_reports_dashboard(client):
    m = client.get("/api/models/diabetes/metrics").get_json()["data"]; assert m["test_metrics"]["roc_auc"] and m["candidates"]
    g = client.get("/api/explain/diabetes/global").get_json()["data"]["explanation"]; assert g["importance"] and g["beeswarm"]
    f = client.get("/api/fairness/diabetes").get_json()["data"]
    assert f["status"] == "available" and {a["attribute"] for a in f["attributes"]} == {"sex", "age_group"} and f["limitations"]
    assert all("n" in g for a in f["attributes"] for g in a["groups"])
    assert client.get("/api/reports/diabetes").data.decode().startswith("# FairXAI report")
    s = client.get("/api/dashboard/summary").get_json()["data"]; assert s["trained"] == 1 and s["supported"] == 5 and s["predictions"]["total"] >= 1
    assert client.get("/api/models?disease=diabetes").get_json()["data"][0]["algorithm"]


def test_database_failure_does_not_crash(trained_app):
    from backend.app.db import Database
    saved = trained_app.db.db; trained_app.db.db = None
    try:
        c = trained_app.test_client(); inputs = sample_inputs(c)
        r = c.post("/api/predict/diabetes", json=dict(inputs=inputs)); assert r.status_code == 200
        assert any("could not be saved" in w for w in r.get_json()["data"]["warnings"]) or r.get_json()["data"]["prediction_id"]
        assert c.get("/api/predictions").get_json()["data"]["count"] == 0
        assert c.get("/api/dashboard/summary").status_code == 200 and c.delete("/api/predictions").status_code == 503
    finally: trained_app.db.db = saved


def test_mongodb_misconfiguration_reports_status_without_credentials():
    from backend.app.db import Database
    class Cfg: MONGO_URI = "mongodb://user:secretpw@127.0.0.1:1/?serverSelectionTimeoutMS=300"; MONGO_DB = "x"; ALLOW_MEMORY_DB = False
    d = Database(Cfg); assert not d.ok and "secretpw" not in str(d.error) and "failed" in d.error


def test_admin_protection_and_upload_validation(trained_app):
    from backend.app import create_app
    class Cfg:
        from backend.app.config import Config as C
        ENV = "development"; PRODUCTION = False; MONGO_URI = ""; MONGO_DB = "t"; ALLOW_MEMORY_DB = True; ADMIN_TOKEN = "s3cret"; CORS_ORIGINS = []; MAX_CONTENT_LENGTH = 1024; PREDICTION_RETENTION_DAYS = 1; STORE_PREDICTION_INPUTS = False
    app = create_app(Cfg); c = app.test_client()
    assert c.post("/api/train/diabetes").status_code == 401 and c.delete("/api/predictions").status_code == 401
    assert c.post("/api/train/diabetes", headers={"X-Admin-Token": "wrong"}).status_code == 401
    import io
    bad = c.post("/api/datasets/ckd/upload", headers={"X-Admin-Token": "s3cret"}, data={"file": (io.BytesIO(b"x"), "evil.exe")}, content_type="multipart/form-data"); assert bad.status_code == 415
    big = c.post("/api/datasets/ckd/upload", headers={"X-Admin-Token": "s3cret"}, data={"file": (io.BytesIO(b"x" * 5000), "a.csv")}, content_type="multipart/form-data"); assert big.status_code == 413
    assert c.post("/api/train/tb", headers={"X-Admin-Token": "s3cret"}).status_code == 422          # no dataset -> clear error, no crash


def test_empty_state_endpoints(trained_app):
    from backend.app import create_app
    class Cfg:
        ENV = "development"; PRODUCTION = False; MONGO_URI = ""; MONGO_DB = "empty"; ALLOW_MEMORY_DB = True; ADMIN_TOKEN = ""; CORS_ORIGINS = []; MAX_CONTENT_LENGTH = 1 << 20; PREDICTION_RETENTION_DAYS = 1; STORE_PREDICTION_INPUTS = True
    c = create_app(Cfg).test_client()
    assert c.get("/api/predictions").get_json()["data"]["items"] == [] and c.get("/api/train/runs").status_code == 200
    assert c.get("/api/datasets/tb/report").status_code == 404 and c.get("/api/fairness/tb").status_code == 404


def test_end_to_end_upload_clean_train_predict(trained_app, raw_dir):
    """Dataset upload -> clean -> train (background) -> register -> predict -> explain -> fairness -> history -> dashboard."""
    import io
    from tests.conftest import synthetic_frame
    c = trained_app.test_client(); df = synthetic_frame(500, seed=7).rename(columns={"Outcome": "classification"})
    df["classification"] = df["classification"].map({0: "notckd", 1: "ckd"})
    up = c.post("/api/datasets/ckd/upload", data={"file": (io.BytesIO(df.to_csv(index=False).encode()), "synthetic_ckd.csv")}, content_type="multipart/form-data"); assert up.status_code == 201
    assert c.post("/api/datasets/ckd/clean").status_code == 200
    assert c.get("/api/datasets/ckd/report").get_json()["data"]["target_column"] == "classification"
    run = c.post("/api/train/ckd"); assert run.status_code == 202
    for _ in range(120):
        st = [r for r in c.get("/api/train/runs").get_json()["data"] if r["disease"] == "ckd"][0]
        if st["status"] != "running": break
        time.sleep(2)
    assert st["status"] == "completed", st
    f = c.get("/api/models/ckd/schema").get_json()["data"]["fields"]
    inp = {x["name"]: (x["median"] if x["type"] == "number" else x["mode"] if x["type"] == "category" else x["options"][0]) for x in f}
    p = c.post("/api/predict/ckd", json=dict(inputs=inp)).get_json()["data"]; assert p["disease"] == "ckd" and p["explanations"]["shap"]["contributions"]
    assert c.get("/api/fairness/ckd").get_json()["data"]["status"] == "available"
    assert any(m["disease"] == "ckd" for m in c.get("/api/models").get_json()["data"])
    assert c.get("/api/dashboard/summary").get_json()["data"]["trained"] == 2
