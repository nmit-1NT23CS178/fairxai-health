"""Honest, auto-generated model report (Markdown) from stored metadata only."""
from . import inference as inf
from .common import REPORT_DIR, DISCLAIMER, jload
from .diseases import DISEASES

pct = lambda x: "n/a" if x is None else f"{x:.3f}"


def build_markdown(disease, version=None):
    m = inf.load_meta(disease, version); t, v = m["test_metrics"], m["validation_metrics"]
    q = jload(REPORT_DIR / f"{disease}_quality.json") if (REPORT_DIR / f"{disease}_quality.json").exists() else None
    L = [f"# FairXAI report - {m['disease_name']}", f"*Model {m['version']} - {m['algorithm']} - trained {m['trained_at']}*", "", f"> {DISCLAIMER}", "",
         "## Dataset", f"- File: `{m['dataset']}`  rows used: {m['dataset_rows']}  target: `{m['target_column']}` ({m['target_encoding']['rule']})",
         f"- Split: {m['split']['scheme']} (seed {m['seed']}): train {m['split']['train']}, validation {m['split']['validation']}, test {m['split']['test']}"]
    if q: L += [f"- Rows {q['original_rows']} -> {q['cleaned_rows']}; duplicates removed {q['duplicate_rows_removed']}; class distribution {q['target_distribution']}"]
    L += ["", "## Model comparison (validation set, 5-fold CV on training data)", "| Model | CV ROC-AUC | Val ROC-AUC | Val PR-AUC | Val recall | Train s | Selected |", "|---|---|---|---|---|---|---|"]
    for c in m["candidates"]:
        vm = c["val_metrics"]; L.append(f"| {c['model']} | {c['cv_roc_auc_mean']:.3f} +/- {c['cv_roc_auc_std']:.3f} | {pct(vm['roc_auc'])} | {pct(vm['pr_auc'])} | {pct(vm['recall'])} | {c['train_time_s']} | {'yes' if c['selected'] else ''} |")
    L += ["", m["selection_reason"], "", "## Final test performance (untouched test set)", "| Metric | Value |", "|---|---|"]
    for k in ("accuracy", "precision", "recall", "specificity", "f1", "roc_auc", "pr_auc", "brier"): L.append(f"| {k} | {pct(t[k])} |")
    (tn, fp), (fn, tp) = t["confusion_matrix"]
    L += ["", f"Confusion matrix (TN, FP / FN, TP): {tn}, {fp} / {fn}, {tp}  (n={t['n']})", f"- False negatives: {m['error_costs']['false_negative']}", f"- False positives: {m['error_costs']['false_positive']}",
          "", "## Explainability (SHAP, global)"]
    ge = m["global_explanation"]
    L += [f"- {i['feature']}: {i['mean_abs_shap']:.4f}" for i in ge.get("importance", [])[:10]] or [f"- {ge.get('error')}"]
    L += ["", "## Fairness"]; f = m["fairness"]
    L += [f"- {f.get('reason')}"] if f["status"] != "available" else [f"- {x}" for x in f["findings"]]
    L += [f"- Limitation: {x}" for x in f.get("limitations", [])]
    L += ["", "## Known limitations"] + [f"- {x}" for x in m["limitations"]]
    return "\n".join(L)
