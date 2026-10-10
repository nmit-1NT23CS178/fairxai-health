import numpy as np, pandas as pd, pytest
from backend.ml import data_prep as dp
from backend.ml.common import RAW_DIR, PROC_DIR
from tests.conftest import synthetic_frame


def write(disease, df, name="d.csv"):
    f = RAW_DIR / disease; f.mkdir(parents=True, exist_ok=True)
    for old in f.glob("*"): old.unlink()
    df.to_csv(f / name, index=False)


def test_missing_dataset_and_image_only_status():
    assert dp.discover("tb")["status"] == "missing"
    (RAW_DIR / "tb").mkdir(parents=True, exist_ok=True); (RAW_DIR / "tb" / "x.png").write_bytes(b"x")
    assert dp.discover("tb")["status"] == "image_only"
    with pytest.raises(dp.DatasetError): dp.clean_dataset("tb")
    (RAW_DIR / "tb" / "x.png").unlink()
    with pytest.raises(dp.DatasetError): dp.clean_dataset("tb")


def test_missing_target_rows_dropped_never_imputed():
    df = synthetic_frame(100); df.loc[:4, "Outcome"] = np.nan
    write("diabetes", df); r = dp.clean_dataset("diabetes")
    assert r["cleaned_rows"] == 95 and any(l["op"] == "drop_rows_missing_target" and l["rows"] == 5 for l in r["cleaning_log"])
    assert sum(r["target_distribution"].values()) == 95


def test_duplicates_ids_and_invalid_values():
    df = synthetic_frame(100); df = pd.concat([df, df.iloc[:7]]); df.loc[df.index[10], "glucose"] = 0   # impossible glucose
    write("diabetes", df); r = dp.clean_dataset("diabetes")
    assert r["duplicate_rows_removed"] == 7
    assert any(x["column"] == "patient_id" for x in r["features_removed"])
    assert r["missing_after"].get("glucose", 0) >= 1                      # set to NaN, row retained
    assert r["cleaned_rows"] == 100
    assert pd.read_csv(PROC_DIR / "diabetes.csv").shape[0] == 100


def test_raw_file_not_modified():
    df = synthetic_frame(80); write("diabetes", df)
    before = (RAW_DIR / "diabetes" / "d.csv").read_bytes(); dp.clean_dataset("diabetes")
    assert (RAW_DIR / "diabetes" / "d.csv").read_bytes() == before


def test_ambiguous_target_raises():
    df = synthetic_frame(60); df["Outcome"] = np.random.RandomState(0).choice(["a", "b"], 60)
    write("diabetes", df)
    with pytest.raises(dp.DatasetError): dp.clean_dataset("diabetes")


def test_target_derived_column_removed_and_leak_flagged():
    df = synthetic_frame(200); df["target_copy"] = df["Outcome"]; df["perfect"] = df["Outcome"] * 5.0 + np.random.RandomState(1).rand(200) * 0.01
    write("diabetes", df); r = dp.clean_dataset("diabetes")
    assert "target_copy" not in r["features_retained"]
    assert any(x.get("column") == "perfect" for x in r["leakage_checks"])


def test_string_numeric_coercion_and_tab_labels():
    df = synthetic_frame(60); df["classification"] = df.pop("Outcome").map({0: "notckd", 1: "ckd\t"}); df["glucose"] = df["glucose"].round(0).astype(int).astype(str)
    df.loc[3, "glucose"] = "?"; write("ckd", df); r = dp.clean_dataset("ckd")
    assert r["target_encoding"]["rule"].startswith("'ckd' -> 1") and "glucose" in r["numeric_features"]
