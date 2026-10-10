"""Evaluation metrics, curves and calibration."""
from __future__ import annotations
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score, confusion_matrix, brier_score_loss, roc_curve,
                             precision_recall_curve)
from sklearn.calibration import calibration_curve


def _thin(a, n=80):
    a = np.asarray(a); 
    if len(a) <= n: return a.tolist()
    return a[np.linspace(0, len(a) - 1, n).astype(int)].tolist()


def compute_metrics(y, p, thr=0.5, curves=False) -> dict:
    y = np.asarray(y); p = np.asarray(p, dtype=float); pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    m = dict(n=int(len(y)), n_positive=int(y.sum()), threshold=float(thr),
             accuracy=accuracy_score(y, pred), precision=precision_score(y, pred, zero_division=0),
             recall=recall_score(y, pred, zero_division=0), specificity=tn / (tn + fp) if (tn + fp) else None,
             f1=f1_score(y, pred, zero_division=0),
             roc_auc=roc_auc_score(y, p) if len(set(y)) > 1 else None,
             pr_auc=average_precision_score(y, p) if len(set(y)) > 1 else None,
             brier=brier_score_loss(y, np.clip(p, 0, 1)),
             confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
             per_class=dict(
                 negative=dict(precision=float(tn / (tn + fn)) if (tn + fn) else None, recall=float(tn / (tn + fp)) if (tn + fp) else None, support=int(tn + fp)),
                 positive=dict(precision=float(tp / (tp + fp)) if (tp + fp) else None, recall=float(tp / (tp + fn)) if (tp + fn) else None, support=int(tp + fn))))
    if curves and len(set(y)) > 1:
        fpr, tpr, _ = roc_curve(y, p); pr, rc, _ = precision_recall_curve(y, p)
        m["roc_curve"] = dict(fpr=_thin(fpr), tpr=_thin(tpr))
        m["pr_curve"] = dict(precision=_thin(pr), recall=_thin(rc))
        try:
            nb = 5 if len(y) < 200 else 10
            fr, mp = calibration_curve(y, p, n_bins=nb, strategy="quantile")
            m["calibration_curve"] = dict(mean_predicted=mp.tolist(), fraction_positive=fr.tolist())
        except Exception:
            m["calibration_curve"] = None
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()}


TARGET_RECALL = 0.90


def choose_threshold(y, p, target_recall=TARGET_RECALL) -> tuple[float, str]:
    """Validation-derived screening threshold: the HIGHEST threshold that still keeps recall >= target_recall
    on the validation set (i.e. maximum specificity subject to not missing more than ~10% of positives).
    Documented, reproducible, and independent of the test set."""
    y = np.asarray(y); p = np.asarray(p, float)
    P, N = (y == 1).sum(), (y == 0).sum()
    best_t, best_spec = None, -1
    for t in np.unique(np.round(p, 5)):
        pred = p >= t
        rec = (pred & (y == 1)).sum() / max(1, P)
        if rec >= target_recall:
            spec = (~pred & (y == 0)).sum() / max(1, N)
            if spec >= best_spec: best_t, best_spec = float(t), spec
    if best_t is None: best_t, best_spec = float(p.min()), 0.0
    return best_t, (f"highest threshold keeping validation recall >= {target_recall:.0%} (validation specificity at this threshold = {best_spec:.3f}); "
                    "chosen for screening where missed cases are costlier than false alarms")
