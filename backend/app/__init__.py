import logging, time
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from .config import Config
from .db import Database
from . import services

FRONT = __import__("pathlib").Path(__file__).resolve().parents[2] / "frontend"


def create_app(cfg=Config):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    app = Flask(__name__, static_folder=str(FRONT), static_url_path="")
    app.config.from_object(cfg); app.cfg = cfg
    CORS(app, resources={r"/api/*": {"origins": cfg.CORS_ORIGINS}})
    app.db = Database(cfg)
    try: services.sync_artifacts(app.db)
    except Exception as e: logging.getLogger("fairxai").warning("registry sync failed: %s", e)
    from .api.routes import bp
    app.register_blueprint(bp, url_prefix="/api")

    @app.before_request
    def _t(): request._t0 = time.time()

    @app.after_request
    def _log(r):
        if request.path.startswith("/api"):
            logging.getLogger("fairxai.req").info("%s %s %s %.0fms", request.method, request.path, r.status_code, (time.time() - request._t0) * 1000)
        r.headers.setdefault("X-Content-Type-Options", "nosniff"); r.headers.setdefault("Referrer-Policy", "no-referrer")
        return r

    @app.errorhandler(404)
    def nf(e):
        if request.path.startswith("/api"): return jsonify(ok=False, error=dict(code="not_found", message="Endpoint not found")), 404
        return send_from_directory(FRONT, "index.html")

    @app.errorhandler(413)
    def big(e): return jsonify(ok=False, error=dict(code="too_large", message=f"Upload exceeds {cfg.MAX_CONTENT_LENGTH // 1048576} MB")), 413

    @app.errorhandler(Exception)
    def boom(e):
        from werkzeug.exceptions import HTTPException
        if isinstance(e, HTTPException): return jsonify(ok=False, error=dict(code=e.name.lower().replace(" ", "_"), message=e.description)), e.code
        logging.getLogger("fairxai").exception("unhandled")
        return jsonify(ok=False, error=dict(code="internal_error", message="Internal server error")), 500

    @app.route("/")
    def index(): return send_from_directory(FRONT, "index.html")
    return app
