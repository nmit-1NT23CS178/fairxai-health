"""SHAP (global + local) and LIME (local) on the SAME fitted pipeline/feature representation as the model.
SHAP runs on the transformed matrix produced by the fitted preprocessor; LIME perturbs raw features and
calls the full pipeline. Explanations describe model behaviour, not causation."""
from __future__ import annotations
import numpy as np, pandas as pd, shap
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

SPACE = {"tree_prob": "probability", "tree_logit": "log-odds", "linear": "log-odds", "kernel": "probability"}


def _transform(bundle, X_raw: pd.DataFrame) -> pd.DataFrame:
    prep = bundle["pipeline"].named_steps["prep"]
    return pd.DataFrame(prep.transform(X_raw[bundle["columns"]]), columns=prep.get_feature_names_out(), index=X_raw.index)


def make_explainer(bundle):
    clf = bundle["pipeline"].named_steps["clf"]
    bg = _transform(bundle, bundle["background_raw"])
    if isinstance(clf, XGBClassifier):
        return shap.TreeExplainer(clf), "tree_logit"
    if isinstance(clf, (RandomForestClassifier, DecisionTreeClassifier)):
        return shap.TreeExplainer(clf), "tree_prob"
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, bg), "linear"
    km = shap.kmeans(bg, 12)
    return shap.KernelExplainer(lambda z: clf.predict_proba(pd.DataFrame(z, columns=bg.columns))[:, 1], km), "kernel"


def _pos(sv):
    if isinstance(sv, list): sv = sv[1]
    sv = np.asarray(sv)
    return sv[:, :, 1] if sv.ndim == 3 else sv


def _base(ev):
    ev = np.asarray(ev).ravel(); return float(ev[-1]) if ev.size else 0.0


def _orig_map(bundle, names):
    cats = bundle["cat"]; m = {}
    for n in names:
        m[n] = next((c for c in cats if n.startswith(c + "_")), n)
    return m


def _values(expl, kind, Xt):
    return _pos(expl.shap_values(Xt, nsamples=120) if kind == "kernel" else expl.shap_values(Xt))


def global_explanation(bundle, X_raw, seed=42, max_rows=150):
    if len(X_raw) > (60 if isinstance(bundle["pipeline"].named_steps["clf"], __import__("sklearn.svm", fromlist=["SVC"]).SVC) else max_rows):
        n = 60 if "SVC" in type(bundle["pipeline"].named_steps["clf"]).__name__ else max_rows
        X_raw = X_raw.sample(n, random_state=seed)
    Xt = _transform(bundle, X_raw)
    expl, kind = make_explainer(bundle)
    sv = _values(expl, kind, Xt)
    omap = _orig_map(bundle, Xt.columns)
    df_sv = pd.DataFrame(sv, columns=Xt.columns).T.groupby(omap).sum().T          # sum one-hot parts per raw feature
    imp = df_sv.abs().mean().sort_values(ascending=False)
    raw_num = X_raw[bundle["columns"]]
    top = imp.index[:15].tolist()
    swarm = {}
    for f in top:
        if f in bundle["num"]:
            v = raw_num[f].astype(float); r = v.rank(pct=True).fillna(.5).values
        else:
            r = np.full(len(df_sv), .5)
        swarm[f] = dict(shap=df_sv[f].round(5).tolist(), value_rank=np.round(r, 3).tolist(), raw=[None if pd.isna(x) else (float(x) if f in bundle["num"] else str(x)) for x in raw_num[f].values])
    return dict(method="SHAP", explainer=type(expl).__name__, output_space=SPACE[kind], n_samples=int(len(Xt)),
                importance=[dict(feature=k, mean_abs_shap=float(v)) for k, v in imp.items()], beeswarm=swarm,
                note="Mean |SHAP| over a sample of the held-out test set. Importance shows what the model relies on, not causal effect.")


_cache = {}


def local_shap(bundle, row: pd.DataFrame, key=None, top=12):
    k = key or id(bundle)
    if k not in _cache: _cache[k] = make_explainer(bundle)
    expl, kind = _cache[k]
    Xt = _transform(bundle, row)
    sv = _values(expl, kind, Xt)[0]
    omap = _orig_map(bundle, Xt.columns)
    s = pd.Series(sv, index=Xt.columns).groupby(omap).sum()
    out = []
    for f, v in s.items():
        val = row[f].iloc[0]
        out.append(dict(feature=f, value=None if pd.isna(val) else (float(val) if f in bundle["num"] else str(val)),
                        shap=float(v), direction="increases model score" if v > 0 else "decreases model score"))
    out.sort(key=lambda d: -abs(d["shap"]))
    return dict(method="SHAP", output_space=SPACE[kind], base_value=_base(expl.expected_value), contributions=out[:top],
                narrative=_narrative(out[:3]),
                note="Contributions are additive in the model's native output space; they are associations learned from data, not causes.")


def _narrative(items):
    if not items: return ""
    parts = [f"{i['feature'].replace('_', ' ')} ({'raises' if i['shap'] > 0 else 'lowers'} the score)" for i in items]
    return "Most influential inputs for this result: " + ", ".join(parts) + "."


def local_lime(bundle, row: pd.DataFrame, num_features=10, seed=42):
    from lime.lime_tabular import LimeTabularExplainer
    cols, cats = bundle["columns"], bundle["cat"]
    bg = bundle["background_raw"][cols].copy()
    cat_maps = {c: sorted(bg[c].dropna().astype(str).unique().tolist()) for c in cats}
    med = {c: float(bg[c].median()) for c in bundle["num"]}

    def enc(df):
        a = pd.DataFrame(index=df.index)
        for c in cols:
            if c in cats:
                m = {v: i for i, v in enumerate(cat_maps[c])}
                a[c] = df[c].astype(str).map(m).fillna(0).astype(float)
            else:
                a[c] = df[c].astype(float).fillna(med[c])
        return a.values

    def predict_fn(arr):
        d = pd.DataFrame(arr, columns=cols)
        for c in cats: d[c] = [cat_maps[c][min(max(int(round(v)), 0), len(cat_maps[c]) - 1)] for v in d[c]]
        return bundle["pipeline"].predict_proba(d)

    ex = LimeTabularExplainer(enc(bg), feature_names=cols, class_names=["negative", "positive"],
                              categorical_features=[cols.index(c) for c in cats],
                              categorical_names={cols.index(c): cat_maps[c] for c in cats}, mode="classification",
                              discretize_continuous=True, random_state=seed)
    e = ex.explain_instance(enc(row[cols])[0], predict_fn, num_features=num_features, num_samples=1000)
    items = [dict(rule=r, weight=float(w), direction="supports positive class" if w > 0 else "supports negative class") for r, w in e.as_list()]
    return dict(method="LIME", local_prediction=float(e.local_pred[0]) if hasattr(e, "local_pred") else None, r2_of_local_fit=float(e.score),
                items=items, note="LIME fits a local linear surrogate around this record; weights can vary between runs/seeds and are not causal.")
