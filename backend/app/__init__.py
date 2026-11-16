import logging
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from werkzeug.middleware.proxy_fix import ProxyFix

from .config import Config
from .db import Database
from . import services

FRONT = Path(__file__).resolve().parents[2] / "frontend"
ASSET_SUFFIXES = {".css", ".js", ".png", ".jpg", ".jpeg", ".svg", ".ico", ".webp", ".woff", ".woff2", ".map", ".json", ".txt"}
log = logging.getLogger("fairxai")


def create_app(cfg=Config):
    logging.basicConfig(level=getattr(logging, getattr(cfg, "LOG_LEVEL", "INFO"), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = Flask(__name__, static_folder=str(FRONT), static_url_path="")
    app.config.from_object(cfg); app.cfg = cfg
    # Render (and most PaaS) terminate TLS at a single proxy in front of gunicorn.
    app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1)
    CORS(app, resources={r"/api/*": {"origins": cfg.CORS_ORIGINS}})
    app.db = Database(cfg)

    if getattr(cfg, "PRODUCTION", False):
        if not getattr(cfg, "MONGO_URI", ""):
            log.error("PRODUCTION without MONGO_URI: prediction history will NOT be persisted.")
        if not getattr(cfg, "ADMIN_TOKEN", ""):
            log.warning("ADMIN_TOKEN is not set: upload/clean/train/delete endpoints are disabled.")
        if getattr(cfg, "ALLOW_MEMORY_DB", False):
            log.warning("ALLOW_MEMORY_DB is enabled in production: data is NOT persistent.")

    try:
        services.sync_artifacts(app.db)
    except Exception as e:
        log.warning("registry sync failed: %s", e)

    from .api.routes import bp
    app.register_blueprint(bp, url_prefix="/api")

    @app.before_request
    def _start():
        request._t0 = time.time()
        # Cheap while connected; rate-limited reconnect after an Atlas blip.
        if request.path.startswith("/api") and app.db.db is None:
            app.db.ensure()

    @app.after_request
    def _headers(r):
        if request.path.startswith("/api"):
            logging.getLogger("fairxai.req").info(
                "%s %s %s %.0fms", request.method, request.path, r.status_code, (time.time() - request._t0) * 1000
            )
            r.headers["Cache-Control"] = "no-store"
        r.headers.setdefault("X-Content-Type-Options", "nosniff")
        r.headers.setdefault("X-Frame-Options", "DENY")
        r.headers.setdefault("Referrer-Policy", "no-referrer")
        r.headers.setdefault("Permissions-Policy", "geolocation=(), microphone=(), camera=()")
        r.headers.setdefault("Content-Security-Policy", "frame-ancestors 'none'; base-uri 'self'; object-src 'none'")
        if request.is_secure:
            r.headers.setdefault("Strict-Transport-Security", "max-age=31536000; includeSubDomains")
        r.headers.pop("Server", None)
        return r

    @app.errorhandler(404)
    def nf(e):
        if request.path.startswith("/api"):
            return jsonify(ok=False, error=dict(code="not_found", message="Endpoint not found")), 404
        if Path(request.path).suffix.lower() in ASSET_SUFFIXES:
            return jsonify(ok=False, error=dict(code="not_found", message="File not found")), 404
        return send_from_directory(FRONT, "index.html")

    @app.errorhandler(413)
    def big(e):
        return jsonify(ok=False, error=dict(code="too_large", message=f"Upload exceeds {cfg.MAX_CONTENT_LENGTH // 1048576} MB")), 413

    @app.errorhandler(Exception)
    def boom(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException):
            return jsonify(ok=False, error=dict(code=e.name.lower().replace(" ", "_"), message=e.description)), e.code
        log.exception("unhandled")
        return jsonify(ok=False, error=dict(code="internal_error", message="Internal server error")), 500

    @app.route("/")
    def index():
        return send_from_directory(FRONT, "index.html")

    return app
