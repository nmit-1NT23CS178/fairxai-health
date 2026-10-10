"""Shared paths, constants and helpers for the FairXAI ML layer."""
from __future__ import annotations
import json, os, re
from pathlib import Path

ROOT = Path(os.environ.get("FAIRXAI_ROOT", Path(__file__).resolve().parents[2]))
DATA_DIR = Path(os.environ.get("DATA_DIR", ROOT / "data"))
RAW_DIR, PROC_DIR, REPORT_DIR = DATA_DIR / "raw", DATA_DIR / "processed", DATA_DIR / "reports"
ARTIFACT_DIR = Path(os.environ.get("ARTIFACT_DIR", ROOT / "artifacts"))
SEED = int(os.environ.get("RANDOM_SEED", 42))
MIN_GROUP_SIZE = 30  # subgroups smaller than this are flagged "unreliable"

DISCLAIMER = ("AI-generated estimate from a research prototype. It is NOT a medical diagnosis or "
              "treatment recommendation and has not been clinically validated.")


def norm_col(name: str) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return s or "col"


def jdump(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_default))


def jload(path: Path):
    return json.loads(Path(path).read_text())


def _default(o):
    import numpy as np
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return None if np.isnan(o) else float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    if isinstance(o, np.ndarray): return o.tolist()
    if hasattr(o, "isoformat"): return o.isoformat()
    return str(o)
