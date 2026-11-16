import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


def _origins(raw: str) -> list:
    return [o.strip().rstrip("/") for o in raw.split(",") if o.strip()]


class Config:
    ENV = os.environ.get("APP_ENV", "development")
    PRODUCTION = ENV == "production"
    DEBUG = _flag("FLASK_DEBUG", False)
    LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()

    MONGO_URI = os.environ.get("MONGO_URI", "")
    MONGO_DB = os.environ.get("MONGO_DB", "fairxai")
    # Never silently fall back to a non-persistent database in production.
    ALLOW_MEMORY_DB = _flag("ALLOW_MEMORY_DB", not PRODUCTION)

    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
    CORS_ORIGINS = _origins(os.environ.get("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000"))

    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 25)) * 1024 * 1024
    PREDICTION_RETENTION_DAYS = int(os.environ.get("PREDICTION_RETENTION_DAYS", 90))
    STORE_PREDICTION_INPUTS = _flag("STORE_PREDICTION_INPUTS", True)

    COMMIT = os.environ.get("RENDER_GIT_COMMIT") or os.environ.get("GIT_COMMIT") or "unknown"
