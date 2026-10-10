"""Subgroup performance & fairness auditing (Fairlearn + Wilson intervals). Reports metrics - it never
declares a model 'fair'. Fairness analysis is kept separate from ordinary predictive performance."""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from .common import MIN_GROUP_SIZE, SEED


def wilson(k, n, z=1.96):
    if n == 0: return [None, None]
    p = k / n; d = 1 + z * z / n; c = p + z * z / (2 * n); a = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [float(max(0, (c - a) / d)), float(min(1, (c + a) / d))]


def group_table(y, pred, groups):
    y, pred, groups = np.asarray(y), np.asarray(pred), np.asarray(groups).astype(str)
    out = []
    for g in sorted(set(groups)):
        m = groups == g; yy, pp = y[m], pred[m]
        tp = int(((pp == 1) & (yy == 1)).sum()); fp = int(((pp == 1) & (yy == 0)).sum())
        tn = int(((pp == 0) & (yy == 0)).sum()); fn = int(((pp == 0) & (yy == 1)).sum()); n = int(m.sum())
        P, N = tp + fn, tn + fp
        rec = dict(group=g, n=n, n_positive=P, n_negative=N,
                   selection_rate=(tp + fp) / n, tpr=tp / P if P else None, fpr=fp / N if N else None,
                   precision=tp / (tp + fp) if (tp + fp) else None, accuracy=(tp + tn) / n,
                   ci=dict(selection_rate=wilson(tp + fp, n), tpr=wilson(tp, P), fpr=wilson(fp, N), accuracy=wilson(tp + tn, n)),
                   reliable=bool(n >= MIN_GROUP_SIZE and P >= 10 and N >= 10))
        out.append(rec)
    return out


def disparities(table):
    ok = [r for r in table if r["reliable"]]
    if len(ok) < 2: return None
    rng = lambda k: float(max(r[k] for r in ok) - min(r[k] for r in ok))
    sel, tpr, fpr = rng("selection_rate"), rng("tpr"), rng("fpr")
    return dict(demographic_parity_difference=sel, equal_opportunity_difference=tpr, equalized_odds_difference=max(tpr, fpr),
                tpr_difference=tpr, fpr_difference=fpr, accuracy_difference=rng("accuracy"), groups_compared=[r["group"] for r in ok])


def make_groups(X: pd.DataFrame, sensitive: dict):
    res = {}
    if "sex" in sensitive and sensitive["sex"] in X:
        col = X[sensitive["sex"]]
        res["sex"] = (col.astype(str).str.strip(), sensitive["sex"], "Group labels are the dataset's own codes; verify their meaning in the data dictionary.")
    if "age" in sensitive and sensitive["age"] in X and pd.api.types.is_numeric_dtype(X[sensitive["age"]]):
        a = X[sensitive["age"]]
        res["age_group"] = (pd.cut(a, [-1, 39, 59, 200], labels=["<40", "40-59", "60+"]).astype(str).replace("nan", "unknown"), sensitive["age"], "Age bands (<40, 40-59, 60+) chosen for interpretability, not clinical cut-offs.")
    return res


class _ScoreEstimator(BaseEstimator, ClassifierMixin):
    """Wraps the calibrated pipeline so Fairlearn can use it (accepts ndarray or DataFrame)."""
    def __init__(self, bundle=None): self.bundle = bundle
    def fit(self, X, y=None): self.classes_ = np.array([0, 1]); return self
    def _df(self, X):
        d = pd.DataFrame(X, columns=self.bundle["columns"]) if not isinstance(X, pd.DataFrame) else X
        for c in self.bundle["num"]: d[c] = pd.to_numeric(d[c], errors="coerce")
        return d
    def predict_proba(self, X):
        from .train import apply_calibrator
        p = apply_calibrator(self.bundle["calibrator"], self.bundle["pipeline"].predict_proba(self._df(X)[self.bundle["columns"]])[:, 1])
        return np.c_[1 - p, p]
    def predict(self, X): return (self.predict_proba(X)[:, 1] >= self.bundle["threshold"]).astype(int)


def audit(bundle, X_va, y_va, X_te, y_te, sensitive, thr):
    from .train import apply_calibrator
    limits = ["Fairness metrics describe disparities on one held-out test set; small subgroups give wide, unstable estimates.",
              "No single metric establishes fairness; demographic parity, equal opportunity and equalized odds can conflict.",
              "Only attributes present in the data are audited (no ethnicity/socioeconomic data unless supplied); intersectional groups are not evaluated.",
              "Calculating these metrics does not mean bias has been removed."]
    g_te_all, g_va_all = make_groups(X_te, sensitive), make_groups(X_va, sensitive)
    if not g_te_all:
        return dict(status="not_available", reason="Dataset has no recognised age/sex columns, so subgroup fairness cannot be audited.", limitations=limits, attributes=[])
    p = apply_calibrator(bundle["calibrator"], bundle["pipeline"].predict_proba(X_te[bundle["columns"]])[:, 1])
    pred = (p >= thr).astype(int); y = np.asarray(y_te)
    attrs, findings = [], []
    for name, (g, col, note) in g_te_all.items():
        tab = group_table(y, pred, g.values); disp = disparities(tab)
        item = dict(attribute=name, source_column=col, note=note, groups=tab, disparities=disp, mitigation=None)
        if disp is None:
            item["status"] = "insufficient_data"
            findings.append(f"{name}: fewer than two subgroups meet the minimum size (n>={MIN_GROUP_SIZE}, >=10 positives and negatives); parity metrics not computed.")
        else:
            item["status"] = "computed"
            findings.append(f"{name}: TPR gap {disp['tpr_difference']:.3f}, FPR gap {disp['fpr_difference']:.3f}, selection-rate gap {disp['demographic_parity_difference']:.3f} (reliable groups only).")
            item["mitigation"] = _mitigate(bundle, X_va, y_va, X_te, y, g_va_all[name][0], g, pred, disp, tab)
        attrs.append(item)
    return dict(status="available", attributes=attrs, findings=findings, limitations=limits, min_group_size=MIN_GROUP_SIZE)


def _mitigate(bundle, X_va, y_va, X_te, y_te, g_va, g_te, pred_before, disp_before, tab_before):
    """Post-processing comparison (Fairlearn ThresholdOptimizer, equalized odds). Fitted on VALIDATION data,
    evaluated on TEST. Reported for comparison only - the deployed model is NOT silently altered."""
    try:
        from fairlearn.postprocessing import ThresholdOptimizer
        ok = [r["group"] for r in tab_before if r["reliable"]]
        mv = g_va.astype(str).isin(ok).values; mt = g_te.astype(str).isin(ok).values
        if mv.sum() < 60 or mt.sum() < 60: return dict(status="skipped", reason="too few samples in reliable groups")
        est = _ScoreEstimator(bundle).fit(X_va)
        to = ThresholdOptimizer(estimator=est, constraints="equalized_odds", objective="balanced_accuracy_score",
                                prefit=True, predict_method="predict_proba")
        to.fit(X_va[mv], np.asarray(y_va)[mv], sensitive_features=g_va[mv].astype(str).values)
        after = to.predict(X_te[mt], sensitive_features=g_te[mt].astype(str).values, random_state=SEED)
        yt = np.asarray(y_te)[mt]; gt = g_te[mt].astype(str).values
        tab_a = group_table(yt, after, gt); disp_a = disparities(tab_a)
        tab_b = group_table(yt, np.asarray(pred_before)[mt], gt); disp_b = disparities(tab_b)
        acc = lambda pr: float((pr == yt).mean()); rec = lambda pr: float(((pr == 1) & (yt == 1)).sum() / max(1, (yt == 1).sum()))
        return dict(status="computed", method="Fairlearn ThresholdOptimizer (equalized_odds, randomized)",
                    before=dict(accuracy=acc(np.asarray(pred_before)[mt]), recall=rec(np.asarray(pred_before)[mt]), disparities=disp_b),
                    after=dict(accuracy=acc(after), recall=rec(after), disparities=disp_a, groups=tab_a),
                    note="Comparison only; the deployed model is unchanged. Mitigation may trade overall accuracy/recall for smaller gaps and uses randomised decisions, which may be unsuitable clinically.")
    except Exception as e:
        return dict(status="failed", error=str(e)[:300])
