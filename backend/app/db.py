"""MongoDB access layer (PyMongo). Falls back to an in-memory mongomock database ONLY when allowed (dev/tests)
and always reports the mode, so the UI never pretends data is persistent when it is not."""
import logging
log = logging.getLogger("fairxai.db")
COLLECTIONS = ["datasets", "model_registry", "predictions", "fairness_audits", "training_runs"]


class Database:
    def __init__(self, cfg):
        self.mode, self.error, self.db = "unavailable", None, None
        try:
            if cfg.MONGO_URI:
                from pymongo import MongoClient
                c = MongoClient(cfg.MONGO_URI, serverSelectionTimeoutMS=3000)
                c.admin.command("ping"); self.db, self.mode = c[cfg.MONGO_DB], "mongodb"
            elif cfg.ALLOW_MEMORY_DB:
                import mongomock
                self.db, self.mode = mongomock.MongoClient()[cfg.MONGO_DB], "memory"
                self.error = "MONGO_URI not set: using NON-persistent in-memory database (development only)."
            else:
                self.error = "MONGO_URI not set."
        except Exception as e:  # never leak credentials/URI
            self.error = f"MongoDB connection failed ({type(e).__name__})."
            log.error(self.error)
            if cfg.ALLOW_MEMORY_DB:
                import mongomock
                self.db, self.mode = mongomock.MongoClient()[cfg.MONGO_DB], "memory"
        if self.db is not None: self._setup(cfg)

    @property
    def ok(self): return self.db is not None

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
                except Exception as e: log.warning("schema validator not applied: %s", type(e).__name__)
        except Exception as e:
            log.error("index setup failed: %s", e)

    def col(self, name):
        if not self.ok: raise RuntimeError("database unavailable")
        return self.db[name]
