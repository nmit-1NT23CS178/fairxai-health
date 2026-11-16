"""MongoDB access layer (PyMongo). Falls back to an in-memory mongomock database ONLY when allowed
(dev/tests) and always reports the mode, so the UI never pretends data is persistent when it is not.

The wrapper is deliberately defensive for production: a transient Atlas outage must not crash the
API, and the connection is retried (rate-limited) so the app recovers without a restart.
"""
import logging
import threading
import time

log = logging.getLogger("fairxai.db")
COLLECTIONS = ["datasets", "model_registry", "predictions", "fairness_audits", "training_runs"]


class Database:
    RETRY_SECONDS = 15.0

    def __init__(self, cfg):
        self.cfg = cfg
        self.mode, self.error, self.db = "unavailable", None, None
        self._lock = threading.Lock()
        self._last_attempt = 0.0
        self._connect()

    def _connect(self):
        uri = getattr(self.cfg, "MONGO_URI", "") or ""
        allow_memory = bool(getattr(self.cfg, "ALLOW_MEMORY_DB", False))
        try:
            if uri:
                from pymongo import MongoClient
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=5000,
                    connectTimeoutMS=5000,
                    socketTimeoutMS=20000,
                    retryWrites=True,
                    appname="fairxai",
                )
                client.admin.command("ping")
                self.db = client[getattr(self.cfg, "MONGO_DB", "fairxai")]
                self.mode, self.error = "mongodb", None
                log.info("MongoDB connected (db=%s)", getattr(self.cfg, "MONGO_DB", "fairxai"))
            elif allow_memory:
                import mongomock
                self.db, self.mode = mongomock.MongoClient()[getattr(self.cfg, "MONGO_DB", "fairxai")], "memory"
                self.error = "MONGO_URI not set: using NON-persistent in-memory database (development only)."
            else:
                self.error = "MONGO_URI not set."
        except Exception as e:  # never leak credentials/URI
            self.error = f"MongoDB connection failed ({type(e).__name__})."
            log.error(self.error)
            if allow_memory:
                import mongomock
                self.db, self.mode = mongomock.MongoClient()[getattr(self.cfg, "MONGO_DB", "fairxai")], "memory"
        if self.db is not None:
            self._setup(self.cfg)

    def ensure(self):
        """Try to (re)connect if the database is currently unavailable. Rate-limited; never raises."""
        if self.db is not None:
            return True
        # Only a real MongoDB connection is retried; a deliberately detached in-memory DB (tests)
        # stays detached so failures are observable instead of silently recovered.
        if self.mode not in ("mongodb", "unavailable"):
            return False
        with self._lock:
            if self.db is not None:
                return True
            if time.monotonic() - self._last_attempt < self.RETRY_SECONDS:
                return False
            self._last_attempt = time.monotonic()
            self.mode, self.db = "unavailable", None
            self._connect()
        return self.db is not None

    @property
    def ok(self):
        return self.db is not None

    def ping(self):
        """Verify the live connection (real MongoDB only) and drop it when it is gone."""
        if self.db is None:
            return False
        if self.mode != "mongodb":
            return True
        try:
            self.db.client.admin.command("ping")
            return True
        except Exception as e:
            self.error = f"MongoDB ping failed ({type(e).__name__})."
            log.error(self.error)
            self.db, self.mode = None, "unavailable"
            return False

    def _setup(self, cfg):
        d = self.db
        try:
            d.predictions.create_index([("created_at", -1)]); d.predictions.create_index([("disease", 1), ("model_version", 1)])
            d.predictions.create_index("prediction_id", unique=True)
            if self.mode == "mongodb":
                d.predictions.create_index("expires_at", expireAfterSeconds=0)  # TTL retention
            d.model_registry.create_index([("disease", 1), ("version", -1)], unique=True)
            d.fairness_audits.create_index([("disease", 1), ("model_version", 1)])
            d.training_runs.create_index([("started_at", -1)]); d.datasets.create_index("disease", unique=True)
            if self.mode == "mongodb":
                try:
                    d.command("collMod", "predictions", validator={"$jsonSchema": {"bsonType": "object", "required": ["prediction_id", "disease", "model_version", "created_at"]}}) \
                        if "predictions" in d.list_collection_names() else d.create_collection("predictions", validator={"$jsonSchema": {"bsonType": "object", "required": ["prediction_id", "disease", "model_version", "created_at"]}})
                except Exception as e:
                    log.warning("schema validator not applied: %s", type(e).__name__)
        except Exception as e:
            log.error("index setup failed: %s", e)

    def col(self, name):
        if not self.ok and not self.ensure():
            raise RuntimeError("database unavailable")
        return self.db[name]
