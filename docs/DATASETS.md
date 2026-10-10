# Datasets

| Disease | File used | Rows | Target | Fairness attrs | Notes |
|---|---|---|---|---|---|
| CVD | `cardio_train.csv` (Kaggle cardiovascular disease, semicolon separated) | 70,000 | `cardio` | gender (coded 1/2), age | age stored in days -> years; impossible BP/height/weight values set to NaN; BMI derived from height/weight |
| Diabetes | Pima `diabetes.csv` | 768 | `Outcome` | age only (all-female cohort) | zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI are impossible -> NaN |
| Breast cancer | WDBC `breast-cancer.csv` | 569 | `diagnosis` (M=1) | none | tabular tumour measurements, no demographics |
| CKD | UCI `kidney_disease.csv` | 400 | `classification` (ckd=1) | age only | many missing values; tab/whitespace artefacts in strings |
| TB | **not provided** | - | - | - | module reports `dataset_missing` |

Other files in the bundle were inspected but not used (alternatives): additional CVD sets (BRFSS-style `CVD_cleaned.csv`, `CVD Dataset.csv`),
the 1,659-row synthetic-looking CKD set (has sex/ethnicity - useful for fairness), symptom-based diabetes set, BCSC risk text files (no header),
nephrotoxic-drug CKD set, and BUSI ultrasound images. To switch, edit `PRIMARY` in `scripts/prepare_datasets.py`
(optionally add `data/raw/<disease>/meta.json` with `target`, `positive_label`, `drop`, `transforms`).
