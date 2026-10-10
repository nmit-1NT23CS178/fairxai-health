import os
from dotenv import load_dotenv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env", override=False)

class Config:
    ENV = os.environ.get("APP_ENV", "development")
    PRODUCTION = ENV == "production"
    MONGO_URI = os.environ.get("MONGO_URI", "")
    MONGO_DB = os.environ.get("MONGO_DB", "fairxai")
    ALLOW_MEMORY_DB = os.environ.get("ALLOW_MEMORY_DB", "0" if PRODUCTION else "1") == "1"
    ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "")
    CORS_ORIGINS = [o.strip() for o in os.environ.get("CORS_ORIGINS", "http://localhost:5000,http://127.0.0.1:5000").split(",") if o.strip()]
    MAX_CONTENT_LENGTH = int(os.environ.get("MAX_UPLOAD_MB", 25)) * 1024 * 1024
    PREDICTION_RETENTION_DAYS = int(os.environ.get("PREDICTION_RETENTION_DAYS", 90))
    STORE_PREDICTION_INPUTS = os.environ.get("STORE_PREDICTION_INPUTS", "1") == "1"
