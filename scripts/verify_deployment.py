"""End-to-end verification of a RUNNING deployment (local or public).

    python scripts/verify_deployment.py --base-url https://your-app.onrender.com
    python scripts/verify_deployment.py --base-url https://your-app.onrender.com --admin-token ...

It exercises health, every trained disease module, SHAP/LIME, fairness, history persistence and
error handling. Inputs are the schema medians/modes learned from the training data (a "typical"
record) - no results are fabricated. Exit code 0 = all checks passed.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request


class Checker:
    def __init__(self, base, timeout=120):
        self.base = base.rstrip("/")
        self.timeout = timeout
        self.passed = 0
        self.failed = []

    def call(self, path, method="GET", payload=None, headers=None):
        data = json.dumps(payload).encode() if payload is not None else None
        h = {"Content-Type": "application/json", "Accept": "application/json"}
        if headers:
            h.update(headers)
        req = urllib.request.Request(self.base + path, data=data, headers=h, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return r.status, json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            try:
                return e.code, json.loads(body)
            except json.JSONDecodeError:
                return e.code, {"raw": body}

    def check(self, name, fn):
        try:
            detail = fn()
            self.passed += 1
            print(f"[PASS] {name}" + (f" - {detail}" if detail else ""))
            return True
        except Exception as e:  # noqa: BLE001
            self.failed.append(name)
            print(f"[FAIL] {name} - {type(e).__name__}: {e}")
            return False

    def skip(self, name, detail):
        print(f"[SKIP] {name} - {detail}")

    def get_ok(self, path):
        st, body = self.call(path)
        assert st == 200, f"HTTP {st}: {body}"
        assert body.get("ok") is True, body
        return body["data"]


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


def run(base, admin_token=None, expect_mongo=True, check_images=True):
    c = Checker(base)
    state = {}

    def health():
        d = c.get_ok("/api/health")
        state["health"] = d
        assert d["status"] in ("ok", "degraded"), d
        assert d["models_loaded"] >= 1, f"models_loaded={d['models_loaded']}"
        db = d["database"]
        assert db["connected"], db
        if expect_mongo:
            assert db["mode"] == "mongodb", f"database mode is {db['mode']}, expected mongodb"
            assert db["persistent"] is True, db
        return (f"status={d['status']} db={db['mode']} models={d['models_loaded']} "
                f"image_models={d.get('image_models_loaded')} commit={d.get('commit')}")

    c.check("health", health)

    def ready():
        st, body = c.call("/api/ready")
        assert st == 200, f"HTTP {st}: {body}"
        d = body["data"]
        return f"db={d['database']['mode']} persistent={d['database']['persistent']}"

    c.check("readiness", ready)

    def diseases():
        ds = c.get_ok("/api/diseases")
        state["trained"] = [x["key"] for x in ds if x["status"] == "trained"]
        assert state["trained"], "no trained diseases reported"
        return ", ".join(state["trained"])

    c.check("disease registry", diseases)

    def dashboard():
        d = c.get_ok("/api/dashboard/summary")
        assert d["supported"] >= 1 and d["trained"] >= 1
        return f"supported={d['supported']} trained={d['trained']} predictions={d['predictions']['total']}"

    c.check("dashboard summary", dashboard)

    for d in state.get("trained", []):
        def schema(d=d):
            s = c.get_ok(f"/api/models/{d}/schema")
            assert s["fields"], "empty schema"
            state[d] = defaults(s["fields"])
            return f"{len(s['fields'])} fields, version {s['version']}"

        c.check(f"{d}: schema", schema)

        def predict(d=d):
            st, body = c.call(f"/api/predict/{d}", "POST", {"inputs": state[d], "lime": True})
            assert st == 200, f"HTTP {st}: {body}"
            p = body["data"]
            assert 0.0 <= p["probability"] <= 1.0, p["probability"]
            assert p["explanations"]["shap"].get("contributions"), "SHAP missing"
            assert p["explanations"]["lime"].get("items"), "LIME missing"
            assert p["prediction_id"], "prediction was not persisted"
            state[d + "_pid"] = p["prediction_id"]
            return (f"p={p['probability']:.4f} flagged={p['flagged']} "
                    f"shap={len(p['explanations']['shap']['contributions'])} lime={len(p['explanations']['lime']['items'])}")

        c.check(f"{d}: prediction + SHAP + LIME", predict)

        def global_xai(d=d):
            g = c.get_ok(f"/api/explain/{d}/global")
            assert g["explanation"]["importance"], "no global importance"
            return f"{len(g['explanation']['importance'])} features"

        c.check(f"{d}: global SHAP", global_xai)

        def fairness(d=d):
            f = c.get_ok(f"/api/fairness/{d}")
            assert f["status"] in ("available", "not_available"), f["status"]
            return f"status={f['status']} attributes={[a['attribute'] for a in f.get('attributes', [])]}"

        c.check(f"{d}: fairness audit", fairness)

        def metrics(d=d):
            m = c.get_ok(f"/api/models/{d}/metrics")
            assert m["test_metrics"]["roc_auc"] is not None
            return f"AUC={m['test_metrics']['roc_auc']:.3f} recall={m['test_metrics']['recall']:.3f}"

        c.check(f"{d}: metrics", metrics)

        def history(d=d):
            h = c.get_ok(f"/api/predictions?disease={d}&limit=50")
            ids = [x["prediction_id"] for x in h["items"]]
            assert state[d + "_pid"] in ids, "just-written prediction not found in history"
            return f"{h['count']} record(s), database={h['database_mode']}"

        c.check(f"{d}: history persistence", history)

        def invalid(d=d):
            fields = c.get_ok(f"/api/models/{d}/schema")["fields"]
            bad = dict(state[d])
            f = next((x for x in fields if x["type"] in ("number", "binary") and x.get("hard_min") is not None), None)
            if f:
                bad[f["name"]] = f["hard_min"] - 1
                what = f"{f['name']}={bad[f['name']]}"
            else:
                num = next((x for x in fields if x["type"] in ("number", "binary")), None)
                if not num:
                    return "no numeric field to invalidate"
                bad[num["name"]] = "not-a-number"
                what = f"{num['name']}='not-a-number'"
            st, body = c.call(f"/api/predict/{d}", "POST", {"inputs": bad})
            assert st == 422, f"HTTP {st}: {body}"
            assert body["error"]["code"] == "validation_error"
            return f"rejected {what} with 422"

        c.check(f"{d}: invalid input rejected", invalid)

    def image_models():
        import io
        from PIL import Image
        buf = io.BytesIO()
        Image.new("RGB", (224, 224), (140, 120, 110)).save(buf, format="PNG")
        png = buf.getvalue()
        boundary = "----fairxaiverify"
        for path, allowed in (
            ("/api/predict/tb-image", {"health", "sick", "tb"}),
            ("/api/predict/breast-ultrasound", {"benign", "malignant", "normal"}),
        ):
            body = (
                f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"probe.png\"\r\n"
                "Content-Type: image/png\r\n\r\n"
            ).encode() + png + f"\r\n--{boundary}--\r\n".encode()
            req = urllib.request.Request(
                c.base + path, data=body,
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=c.timeout) as r:
                    payload = json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                raise AssertionError(f"{path} HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
            assert payload.get("ok"), payload
            data = payload["data"]
            assert data["predicted_class"] in allowed, data
            assert data["gradcam_png_base64"], f"{path}: Grad-CAM missing"
        return "TB X-ray + breast ultrasound + Grad-CAM OK"

    if check_images:
        h = state["health"]
        if h.get("image_models_available", True):
            c.check("image models (TB X-ray, breast ultrasound)", image_models)
        else:
            c.skip("image models (TB X-ray, breast ultrasound)",
                   f"deployed instance reports image_models_available=false "
                   f"(limit={h.get('image_capacity_mb')} MB, needs>={h.get('image_models_min_mb')} MB); "
                   "tabular modules are unaffected")

    def summary_after():
        d = c.get_ok("/api/dashboard/summary")
        assert d["predictions"]["total"] >= len(state.get("trained", []))
        return f"stored predictions={d['predictions']['total']}"

    c.check("predictions recorded in dashboard", summary_after)

    def unknown():
        st, body = c.call("/api/predict/not-a-disease", "POST", {"inputs": {}})
        assert st == 404 and body["error"]["code"] == "unknown_disease", body
        return "404 unknown_disease"

    c.check("unknown disease rejected", unknown)

    def no_model():
        st, body = c.call("/api/models/tb/metrics")
        assert st == 404 and body["error"]["code"] == "model_unavailable", body
        return "404 model_unavailable"

    c.check("untrained module reported honestly", no_model)

    if admin_token:
        def admin_protected():
            st, _ = c.call("/api/train/diabetes", "POST")
            assert st == 401, f"expected 401 without token, got {st}"
            return "401 without token"

        c.check("admin authorization", admin_protected)

    print(f"\n{len(c.failed)} failed, {c.passed} passed")
    return 1 if c.failed else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--admin-token", default=None)
    ap.add_argument("--skip-mongo-check", action="store_true")
    ap.add_argument("--skip-images", action="store_true", help="skip TB X-ray / ultrasound checks")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--retries", type=int, default=0, help="retry the whole run when a free host is cold-starting")
    a = ap.parse_args()
    code = 1
    for attempt in range(a.retries + 1):
        if attempt:
            print(f"\nRetrying ({attempt}/{a.retries}) after cold start...")
            time.sleep(20)
        code = run(a.base_url, a.admin_token, expect_mongo=not a.skip_mongo_check,
                   check_images=not a.skip_images)
        if code == 0:
            break
    sys.exit(code)
