"""Dataset discovery, inspection and cleaning. Raw files are never modified; cleaned output is written to
data/processed and a quality report to data/reports. Imputation/scaling/encoding are NOT done here - they
live inside the sklearn Pipeline and are fitted on training data only (no leakage)."""
from __future__ import annotations
import re, json, datetime as dt
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from .common import RAW_DIR, PROC_DIR, REPORT_DIR, norm_col, jdump, jload
from .diseases import DISEASES, POS_TOKENS, NEG_TOKENS, DOMAIN_RULES

TABULAR_EXT = {".csv", ".xlsx", ".xls", ".tsv"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".dcm"}
MISSING_TOKENS = ["?", "", " ", "na", "n/a", "nan", "null", "none", "\t?", "\t"]


class DatasetError(Exception):
    pass


def discover(disease: str) -> dict:
    """Locate the raw dataset for a disease. Returns status: tabular | image_only | missing."""
    folder = RAW_DIR / disease
    files = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.exists() else []
    tab = [p for p in files if p.suffix.lower() in TABULAR_EXT]
    imgs = [p for p in files if p.suffix.lower() in IMAGE_EXT]
    if tab:
        return dict(status="tabular", path=tab[0], all_files=[str(p) for p in tab])
    if imgs:
        return dict(status="image_only", path=None, n_images=len(imgs))
    return dict(status="missing", path=None)


def load_raw(path: Path) -> pd.DataFrame:
    suf = Path(path).suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".tsv":
        return pd.read_csv(path, sep="\t", na_values=MISSING_TOKENS, skipinitialspace=True)
    with open(path, "r", errors="ignore") as fh:
        head = fh.readline()
    sep = ";" if head.count(";") > head.count(",") else ","
    return pd.read_csv(path, sep=sep, na_values=MISSING_TOKENS, skipinitialspace=True, low_memory=False)


def _meta(disease: str) -> dict:
    p = RAW_DIR / disease / "meta.json"
    return jload(p) if p.exists() else {}


def detect_target(cols: list[str], disease: str, override: str | None) -> str:
    if override:
        o = norm_col(override)
        if o in cols: return o
        raise DatasetError(f"meta.json target '{override}' not found in columns")
    for alias in DISEASES[disease]["target_aliases"]:
        if alias in cols: return alias
    raise DatasetError(f"Could not identify target column among {cols[:15]}... "
                       f"Create data/raw/{disease}/meta.json with {{\"target\": \"<column>\"}}.")


def encode_target(s: pd.Series, name: str, pos_override=None) -> tuple[pd.Series, dict]:
    """Map target to {0,1}. Raises instead of guessing when the meaning is ambiguous."""
    raw_vals = s.dropna().unique().tolist()
    if pos_override is not None:
        y = (s.astype(str).str.strip().str.lower() == str(pos_override).strip().lower()).astype(float)
        y[s.isna()] = np.nan
        return y, dict(rule=f"positive label = '{pos_override}' (meta.json)", raw_values=raw_vals)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().sum() == s.notna().sum():  # numeric target
        u = sorted(num.dropna().unique())
        if len(u) == 2:
            return (num == u[1]).astype(float).where(num.notna()), dict(rule=f"{u[1]} -> 1 (positive), {u[0]} -> 0", raw_values=raw_vals)
        if name == "num" and len(u) > 2:
            return (num > 0).astype(float).where(num.notna()), dict(rule="num>0 -> 1 (UCI heart: any angiographic disease)", raw_values=raw_vals)
        raise DatasetError(f"Numeric target '{name}' has {len(u)} classes; binary expected.")
    st = s.astype(str).str.strip().str.lower().where(s.notna())
    u = [v for v in st.dropna().unique()]
    pos = [v for v in u if v in POS_TOKENS]; neg = [v for v in u if v in NEG_TOKENS]
    if len(u) == 2 and len(pos) == 1 and len(neg) == 1:
        y = (st == pos[0]).astype(float).where(st.notna())
        return y, dict(rule=f"'{pos[0]}' -> 1 (positive), '{neg[0]}' -> 0", raw_values=raw_vals)
    raise DatasetError(f"Ambiguous target labels {u}. Add data/raw/<disease>/meta.json with 'positive_label'.")


def _apply_domain_rules(df: pd.DataFrame, target: str, log: list) -> pd.DataFrame:
    for col in df.columns:
        if col == target or not pd.api.types.is_numeric_dtype(df[col]): continue
        med = df[col].median()
        for pat, lo, hi in DOMAIN_RULES:
            if re.search(pat, col) and lo <= med <= hi:
                bad = (df[col] < lo) | (df[col] > hi)
                if bad.sum():
                    df.loc[bad, col] = np.nan
                    log.append(dict(op="invalid_value_to_missing", column=col, rows=int(bad.sum()),
                                    detail=f"outside plausible range [{lo}, {hi}]; set to NaN (imputed later in pipeline), row kept"))
                break
    return df


def clean_dataset(disease: str) -> dict:
    info = discover(disease)
    if info["status"] == "missing":
        raise DatasetError(f"No dataset found in data/raw/{disease}/ - add a CSV/XLSX file.")
    if info["status"] == "image_only":
        raise DatasetError(f"Only image files found in data/raw/{disease}/ ({info['n_images']} images). "
                           "An image (CNN) pipeline is required and is not part of the tabular trainer.")
    meta = _meta(disease)
    raw = load_raw(info["path"])
    log: list[dict] = []
    n0, c0 = raw.shape
    miss_before = raw.isna().sum()
    df = raw.copy()
    df.columns = [norm_col(c) for c in df.columns]
    dup_cols = df.columns[df.columns.duplicated()].tolist()
    if dup_cols:
        df = df.loc[:, ~df.columns.duplicated()]
        log.append(dict(op="drop_duplicate_columns", columns=dup_cols))
    # string tidy-up
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip().replace({"nan": np.nan, "None": np.nan})
    # numeric coercion for object columns that are mostly numeric
    for c in df.select_dtypes(include="object").columns:
        conv = pd.to_numeric(df[c], errors="coerce")
        if conv.notna().sum() >= 0.9 * df[c].notna().sum() and df[c].notna().sum() > 0:
            newly_missing = int(conv.isna().sum() - df[c].isna().sum())
            df[c] = conv
            log.append(dict(op="coerce_numeric", column=c, rows=newly_missing, detail="object -> numeric; unparseable -> NaN"))
    # id-like columns
    removed: list[dict] = []
    for c in list(df.columns):
        if re.match(r"^(id|patient_?id|record_?id|unnamed.*|index|s_?no|sr_?no|pid)$", c) or \
           (pd.api.types.is_integer_dtype(df[c]) and df[c].is_unique and len(df) > 20 and c.endswith("id")):
            df = df.drop(columns=c); removed.append(dict(column=c, reason="identifier column (no predictive meaning)"))
    target = detect_target(list(df.columns), disease, meta.get("target"))
    y, tinfo = encode_target(df[target], target, meta.get("positive_label"))
    # never invent labels
    miss_t = int(y.isna().sum())
    if miss_t:
        log.append(dict(op="drop_rows_missing_target", rows=miss_t, detail="labels are never imputed"))
    keep = y.notna()
    df, y = df[keep].copy(), y[keep].astype(int)
    df = df.drop(columns=[target]); df["_y"] = y.values
    # exact duplicates (features + label)
    nd = int(df.duplicated().sum())
    if nd:
        df = df.drop_duplicates(); log.append(dict(op="drop_exact_duplicates", rows=nd))
    y = df.pop("_y")
    if "age_days_to_years" in meta.get("transforms", []) and "age" in df.columns and df["age"].median() > 1000:
        df["age"] = (df["age"] / 365.25).round(1)
        log.append(dict(op="unit_conversion", column="age", rows=int(len(df)), detail="age in days -> years (/365.25)"))
    df = _apply_domain_rules(df, target, log)
    if {"ap_hi", "ap_lo"} <= set(df.columns):
        bad = df["ap_hi"].notna() & df["ap_lo"].notna() & (df["ap_hi"] < df["ap_lo"])
        if bad.sum():
            df.loc[bad, ["ap_hi", "ap_lo"]] = np.nan
            log.append(dict(op="invalid_value_to_missing", column="ap_hi/ap_lo", rows=int(bad.sum()), detail="systolic < diastolic is physiologically impossible; both set to NaN, row kept"))
    if "derive_bmi" in meta.get("transforms", []) and {"height", "weight"} <= set(df.columns):
        df["bmi"] = (df["weight"] / (df["height"] / 100) ** 2).round(2)
        df = _apply_domain_rules(df, target, [])
        log.append(dict(op="derive_feature", column="bmi", rows=int(len(df)), detail="bmi = weight(kg) / height(m)^2 from input features only (no target information)"))
    # drop constant / mostly-missing columns
    for c in list(df.columns):
        if df[c].isna().mean() > 0.6:
            df = df.drop(columns=c); removed.append(dict(column=c, reason=">60% missing")); continue
        if df[c].nunique(dropna=True) <= 1:
            df = df.drop(columns=c); removed.append(dict(column=c, reason="constant column"))
    for c in meta.get("drop", []):
        c = norm_col(c)
        if c in df.columns: df = df.drop(columns=c); removed.append(dict(column=c, reason="dropped by meta.json"))
    # leakage checks
    leak = []
    for c in list(df.columns):
        if "target" in c or c == target:
            leak.append(dict(column=c, risk="name suggests target-derived", action="removed")); df = df.drop(columns=c); continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().sum() > 20 and y.nunique() == 2:
            try:
                m = df[c].notna()
                auc = roc_auc_score(y[m], df.loc[m, c])
                if max(auc, 1 - auc) >= 0.99:
                    leak.append(dict(column=c, risk=f"single-feature AUC {max(auc,1-auc):.3f} (near-perfect)", action="flagged - review manually"))
            except Exception:
                pass
    # outlier investigation (reported, never deleted)
    outliers = {}
    for c in df.select_dtypes(include=np.number).columns:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        if iqr > 0:
            n = int(((df[c] < q1 - 3 * iqr) | (df[c] > q3 + 3 * iqr)).sum())
            if n: outliers[c] = n
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    # low-cardinality numeric columns treated as categorical codes? keep numeric (documented)
    sens = _sensitive_columns(df)
    clean = df.copy(); clean[target] = y.values
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = PROC_DIR / f"{disease}.csv"
    clean.to_csv(out_csv, index=False)
    cnt = y.value_counts().sort_index()
    imb = float(cnt.min() / cnt.max())
    limitations = []
    if len(df) < 500: limitations.append(f"Small dataset ({len(df)} rows): estimates have wide uncertainty; results may not generalise.")
    if imb < 0.4: limitations.append(f"Class imbalance (minority/majority = {imb:.2f}); recall/PR-AUC are emphasised and class weights are used in training.")
    if not sens: limitations.append("No age/sex columns present - subgroup fairness cannot be audited for this dataset.")
    limitations.append("Single-source public dataset: no external or prospective validation; not clinically validated.")
    corr = None
    if len(num_cols) >= 2:
        cm = df[num_cols[:25]].corr().round(3)
        corr = dict(columns=cm.columns.tolist(), matrix=cm.where(cm.notna(), None).values.tolist())
    dists = {}
    for c in num_cols[:40]:
        v = df[c].dropna()
        if v.nunique() > 1:
            bins = np.histogram_bin_edges(v, bins=20)
            dists[c] = dict(bins=bins.tolist(), counts={str(k): np.histogram(v[y.loc[v.index] == k], bins=bins)[0].tolist() for k in (0, 1)})
    report = dict(
        disease=disease, generated_at=dt.datetime.utcnow().isoformat() + "Z", source_file=Path(info["path"]).name,
        original_rows=n0, cleaned_rows=int(len(clean)), original_columns=c0, cleaned_columns=int(clean.shape[1]),
        target_column=target, target_encoding=tinfo,
        target_distribution={str(k): int(v) for k, v in cnt.items()}, imbalance_ratio=round(imb, 3),
        missing_before={norm_col(k): int(v) for k, v in miss_before.items() if v},
        missing_after={k: int(v) for k, v in df.isna().sum().items() if v},
        missing_note="Remaining missing values are imputed INSIDE the model pipeline (fit on training data only).",
        duplicate_rows_removed=nd, cleaning_log=log, features_retained=df.columns.tolist(),
        numeric_features=num_cols, categorical_features=cat_cols, features_removed=removed,
        leakage_checks=leak or [dict(risk="none detected by automated checks", action="n/a")],
        outliers_iqr3=outliers, outlier_policy="Reported only; valid extreme clinical values are retained.",
        sensitive_columns=sens, limitations=limitations, correlation=corr, distributions=dists,
        cleaned_file=str(out_csv.relative_to(out_csv.parents[2])) if out_csv.is_relative_to(out_csv.parents[2]) else str(out_csv),
        preview=clean.head(50).replace({np.nan: None}).to_dict(orient="records"))
    jdump(report, REPORT_DIR / f"{disease}_quality.json")
    return report


def _sensitive_columns(df: pd.DataFrame) -> dict:
    s = {}
    for c in df.columns:
        if c in ("sex", "gender"): s["sex"] = c
        if c == "age": s["age"] = c
    return s
