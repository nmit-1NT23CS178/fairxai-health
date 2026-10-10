"""Copies the chosen primary dataset for each disease out of the (nested-zip) Kaggle bundle into data/raw/<disease>/.
Usage: python scripts/prepare_datasets.py path/to/DataSets__kaggle_.zip
Raw originals are only copied, never modified. Edit PRIMARY to switch datasets."""
import sys, zipfile, io, shutil, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from backend.ml.common import RAW_DIR

PRIMARY = {  # disease -> (nested zip inside bundle, member file, optional meta.json)
    "cvd": ("Cardio Vascular Diseases CVD/archive.zip", "cardio_train.csv", {"transforms": ["age_days_to_years", "derive_bmi"]}),
    "diabetes": ("Diabetes Mellitus/archive (3).zip", "diabetes.csv", {}),
    "breast_cancer": ("Breast Cancer/archive.zip", "breast-cancer.csv", {}),
    "ckd": ("Chronic Kidney Disease CKD/archive (3).zip", "kidney_disease.csv", {}),
}

def main(bundle):
    z = zipfile.ZipFile(bundle)
    names = {n.split("DataSets (kaggle)/", 1)[-1]: n for n in z.namelist()}
    for disease, (inner, member, meta) in PRIMARY.items():
        zz = zipfile.ZipFile(io.BytesIO(z.read(names[inner])))
        out = RAW_DIR / disease; out.mkdir(parents=True, exist_ok=True)
        (out / member).write_bytes(zz.read(member))
        if meta: (out / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"{disease:14s} <- {inner} :: {member}")
    print("tb: no dataset in bundle -> left empty (status: dataset_missing)")

if __name__ == "__main__":
    main(sys.argv[1])
