"""Production-configuration behaviour: degraded DB reporting, readiness, security headers and SPA routing."""
from backend.app import create_app


class ProdCfg:
    ENV = "production"
    PRODUCTION = True
    DEBUG = False
    LOG_LEVEL = "CRITICAL"
    MONGO_URI = ""
    MONGO_DB = "fairxai_test"
    ALLOW_MEMORY_DB = False
    ADMIN_TOKEN = ""
    CORS_ORIGINS = ["https://example.com"]
    MAX_CONTENT_LENGTH = 1 << 20
    PREDICTION_RETENTION_DAYS = 90
    STORE_PREDICTION_INPUTS = False
    COMMIT = "test-commit"


def make_client():
    app = create_app(ProdCfg)
    app.config["TESTING"] = True
    return app.test_client()


def test_production_without_mongo_is_reported_degraded():
    c = make_client()
    h = c.get("/api/health")
    assert h.status_code == 200
    d = h.get_json()["data"]
    assert d["status"] == "degraded"
    assert d["database"]["mode"] == "unavailable"
    assert d["database"]["persistent"] is False
    assert d["database"]["required"] is True
    assert d["environment"] == "production"


def test_readiness_fails_without_persistent_database():
    c = make_client()
    r = c.get("/api/ready")
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "not_ready"


def test_admin_endpoints_disabled_in_production_without_token():
    c = make_client()
    r = c.post("/api/train/diabetes")
    assert r.status_code == 503
    assert r.get_json()["error"]["code"] == "admin_disabled"


def test_security_headers_and_no_store_on_api():
    c = make_client()
    r = c.get("/api/health")
    assert r.headers["X-Content-Type-Options"] == "nosniff"
    assert r.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in r.headers["Content-Security-Policy"]
    assert r.headers["Cache-Control"] == "no-store"
    assert "Server" not in r.headers


def test_cors_allows_configured_origin_only():
    c = make_client()
    r = c.options("/api/health", headers={"Origin": "https://example.com", "Access-Control-Request-Method": "GET"})
    assert r.headers.get("Access-Control-Allow-Origin") == "https://example.com"
    r2 = c.options("/api/health", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "GET"})
    assert r2.headers.get("Access-Control-Allow-Origin") is None


def test_spa_fallback_but_missing_assets_are_404():
    c = make_client()
    assert c.get("/dashboard").status_code == 200
    assert c.get("/missing.js").status_code == 404
    assert c.get("/api/nope").get_json()["error"]["code"] == "not_found"
