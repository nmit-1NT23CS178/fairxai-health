"""Leakage-safe, reproducible multi-model training for one disease."""
from __future__ import annotations
import time, datetime as dt, traceback, warnings
import numpy as np, pandas as pd, joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from .common import PROC_DIR, ARTIFACT_DIR, REPORT_DIR, SEED, jdump, jload
from .data_prep import clean_dataset
from .diseases import DISEASES
from .metrics import compute_metrics, choose_threshold
from . import explain as xai, fairness as fair

warnings.filterwarnings("ignore")


def build_preprocessor(num, cat):
    t = []
    if num: t.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num))
    if cat: t.append(("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                                       ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat))
    return ColumnTransformer(t, verbose_feature_names_out=False)


def candidates(n_rows, pos_weight):
    c = {
        "Logistic Regression (baseline)": (LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
                                           {"clf__C": [0.01, 0.1, 1, 10]}),
        "Decision Tree": (DecisionTreeClassifier(class_weight="balanced", random_state=SEED),
                          {"clf__max_depth": [3, 4, 6, 8], "clf__min_samples_leaf": [1, 5, 10]}),
        "Random Forest": (RandomForestClassifier(class_weight="balanced", random_state=SEED, n_jobs=-1),
                          {"clf__n_estimators": [150, 300], "clf__max_depth": [6, 10, 14], "clf__min_samples_leaf": [1, 3]}),
        "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED, scale_pos_weight=pos_weight, n_jobs=1, verbosity=0),
                    {"clf__n_estimators": [100, 250], "clf__max_depth": [2, 3, 5], "clf__learning_rate": [0.03, 0.1], "clf__subsample": [0.8, 1.0]}),
    }
    if n_rows <= 5000:  # kernel SVM scales ~O(n^2); skipped on large data (documented in metadata)
        c["SVM (RBF)"] = (SVC(probability=True, class_weight="balanced", random_state=SEED),
                          {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01, 0.1]})
    return c


def _calibrator(p_val, y_val):
    """Platt scaling on validation scores (monotone, stable for small n)."""
    lr = LogisticRegression(C=1e6, max_iter=1000)
    z = np.log(np.clip(p_val, 1e-6, 1 - 1e-6) / (1 - np.clip(p_val, 1e-6, 1 - 1e-6))).reshape(-1, 1)
    lr.fit(z, y_val); return lr


def apply_calibrator(cal, p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return cal.predict_proba(np.log(p / (1 - p)).reshape(-1, 1))[:, 1]


def train_disease(disease: str, progress=lambda m: None, search_iter: int = 8) -> dict:
    t_start = time.time()
    progress("cleaning dataset")
    report = clean_dataset(disease)
    df = pd.read_csv(PROC_DIR / f"{disease}.csv")
    target = report["target_column"]
    y = df.pop(target).astype(int)
    X = df
    num = [c for c in report["numeric_features"] if c in X.columns]
    cat = [c for c in report["categorical_features"] if c in X.columns]
    if y.value_counts().min() < 10:
        raise ValueError("Fewer than 10 samples in the minority class; cannot train reliably.")
    # 60/20/20 stratified, fixed seed. Test set is untouched until the very end.
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
    X_va, X_te, y_va, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, stratify=y_tmp, random_state=SEED)
    pos_w = float((y_tr == 0).sum() / max(1, (y_tr == 1).sum()))
    big = len(X_tr) > 20000
    if big: search_iter = min(search_iter, 4)
    cv = StratifiedKFold(n_splits=3 if (big or len(X_tr) < 150) else 5, shuffle=True, random_state=SEED)
    rows, fitted = [], {}
    for name, (est, grid) in candidates(len(X_tr), pos_w).items():
        progress(f"training {name}")
        pipe = Pipeline([("prep", build_preprocessor(num, cat)), ("clf", est)])
        n_comb = int(np.prod([len(v) for v in grid.values()]))
        s = RandomizedSearchCV(pipe, grid, n_iter=min(search_iter, n_comb), scoring="roc_auc", cv=cv,
                               random_state=SEED, n_jobs=1, refit=True)
        t0 = time.time(); s.fit(X_tr, y_tr); fit_s = time.time() - t0
        bi = s.best_index_
        pv = s.best_estimator_.predict_proba(X_va)[:, 1]
        vm = compute_metrics(y_va, pv, 0.5)
        # CV recall at default threshold for the chosen params
        rows.append(dict(model=name, best_params={k.replace("clf__", ""): v for k, v in s.best_params_.items()},
                         cv_roc_auc_mean=float(s.cv_results_["mean_test_score"][bi]), cv_roc_auc_std=float(s.cv_results_["std_test_score"][bi]),
                         val_metrics=vm, train_time_s=round(fit_s, 2)))
        fitted[name] = s.best_estimator_
    # selection on VALIDATION data only (ROC-AUC, tie-break PR-AUC, then recall)
    best = max(rows, key=lambda r: (round(r["val_metrics"]["roc_auc"], 4), round(r["val_metrics"]["pr_auc"], 4), r["val_metrics"]["recall"]))
    name = best["model"]; pipe = fitted[name]
    pv = pipe.predict_proba(X_va)[:, 1]
    cal = _calibrator(pv, y_va.values)
    pv_cal = apply_calibrator(cal, pv)
    thr, thr_rule = choose_threshold(y_va, pv_cal)
    val_final = compute_metrics(y_va, pv_cal, thr, curves=True)
    pt = pipe.predict_proba(X_te)[:, 1]; pt_cal = apply_calibrator(cal, pt)
    test = compute_metrics(y_te, pt_cal, thr, curves=True)               # final, untouched test set
    test["brier_uncalibrated"] = compute_metrics(y_te, pt, 0.5)["brier"]
    from .fairness import wilson
    (tn, fp), (fn, tp) = test["confusion_matrix"]
    test["ci95"] = dict(accuracy=wilson(tn + tp, len(y_te)), recall=wilson(tp, tp + fn), specificity=wilson(tn, tn + fp), precision=wilson(tp, tp + fp))
    extra = []
    if len(y_te) < 200:
        extra.append(f"Test set has only {len(y_te)} records ({int(y_te.sum())} positive); 95% CI for recall is {test['ci95']['recall'][0]:.2f}-{test['ci95']['recall'][1]:.2f}. Treat metrics as indicative only.")
    if test["accuracy"] >= 0.99:
        extra.append("Near-perfect test accuracy on a public benchmark does not imply real-world performance; this dataset may be unusually easy or clean compared with routine clinical data.")
    if test["recall"] < 0.9:
        extra.append(f"Test recall is {test['recall']:.2f}: the model misses a notable share of positive cases and should not be used to rule out disease.")
    for r in rows: r["selected"] = r["model"] == name
    reason = (f"{name} had the highest validation ROC-AUC ({best['val_metrics']['roc_auc']:.3f}; ties broken by PR-AUC then recall). "
              f"Selection never used test data. Decision threshold: {thr_rule}.")
    version = dt.datetime.utcnow().strftime("v%Y%m%d%H%M%S")
    out = ARTIFACT_DIR / disease / version
    schema = build_schema(X_tr, num, cat, disease)
    progress("computing SHAP/LIME background and global explanations")
    background = X_tr.sample(min(len(X_tr), 200), random_state=SEED)
    bundle = dict(pipeline=pipe, calibrator=cal, threshold=thr, num=num, cat=cat, columns=list(X.columns), background_raw=background)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out / "model.joblib")
    global_xai = {}
    try:
        global_xai = xai.global_explanation(bundle, X_te, seed=SEED)
    except Exception as e:  # never invent an explanation
        global_xai = dict(error=f"SHAP global explanation failed: {e}")
        traceback.print_exc()
    progress("fairness audit")
    fairness = fair.audit(bundle, X_va, y_va, X_te, y_te, report["sensitive_columns"], thr)
    meta = dict(
        disease=disease, disease_name=DISEASES[disease]["name"], version=version, algorithm=name,
        trained_at=dt.datetime.utcnow().isoformat() + "Z", seed=SEED, dataset=report["source_file"],
        dataset_rows=report["cleaned_rows"], target_column=target, target_encoding=report["target_encoding"],
        split=dict(train=len(X_tr), validation=len(X_va), test=len(X_te), scheme="60/20/20 stratified", seed=SEED),
        feature_schema=schema, candidates=rows, selection_reason=reason, threshold=thr, threshold_rule=thr_rule,
        validation_metrics=val_final, test_metrics=test, global_explanation=global_xai, fairness=fairness,
        class_balance_method="class_weight / scale_pos_weight in training only (no resampling of validation/test)",
        calibration="Platt scaling fitted on the validation set", training_seconds=round(time.time() - t_start, 1),
        limitations=report["limitations"] + extra, error_costs=dict(false_negative=DISEASES[disease]["fn_cost"], false_positive=DISEASES[disease]["fp_cost"]),
        status="active")
    jdump(meta, out / "metadata.json")
    jdump(dict(latest=version), ARTIFACT_DIR / disease / "LATEST.json")
    progress("done")
    return meta


def build_schema(X: pd.DataFrame, num, cat, disease):
    from .diseases import DOMAIN_RULES
    import re
    fields = []
    for c in X.columns:
        f = dict(name=c, label=c.replace("_", " ").title(), allow_missing=bool(X[c].isna().any()))
        if c in num:
            v = X[c].dropna()
            f.update(type="number", min_seen=float(v.min()), max_seen=float(v.max()), median=float(v.median()),
                     integer=bool((v % 1 == 0).all()), hard_min=None, hard_max=None)
            if v.nunique() <= 2: f["type"] = "binary"; f["options"] = sorted(v.unique().tolist())
            for pat, lo, hi in DOMAIN_RULES:
                if re.search(pat, c) and lo <= v.median() <= hi: f["hard_min"], f["hard_max"] = lo, hi
        else:
            opts = sorted(X[c].dropna().astype(str).unique().tolist())
            f.update(type="category", options=opts, mode=X[c].mode().iat[0] if X[c].notna().any() else None)
        fields.append(f)
    return fields
