"""Clean + train every disease that has a tabular dataset in data/raw/<disease>/.
Usage: python scripts/train_all.py [disease ...]"""
import sys, pathlib, traceback
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backend.ml.diseases import DISEASES
from backend.ml.data_prep import discover
from backend.ml.train import train_disease

def main(only):
    summary = []
    for d in (only or DISEASES):
        info = discover(d)
        if info["status"] != "tabular":
            print(f"[{d}] SKIPPED - dataset status: {info['status']}"); summary.append((d, info["status"], None)); continue
        print(f"[{d}] training on {info['path'].name} ...")
        try:
            m = train_disease(d, progress=lambda s, d=d: print(f"   [{d}] {s}"))
            t = m["test_metrics"]
            print(f"[{d}] {m['algorithm']}  test ROC-AUC {t['roc_auc']:.3f}  recall {t['recall']:.3f}  specificity {t['specificity']:.3f}  (n={t['n']})")
            summary.append((d, "trained", m))
        except Exception as e:
            traceback.print_exc(); summary.append((d, f"FAILED: {e}", None))
    print("\nSUMMARY"); [print(" ", d, s) for d, s, _ in summary]

if __name__ == "__main__":
    main(sys.argv[1:])
