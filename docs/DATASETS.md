# Datasets

## Tabular modules
| Disease | File used | Rows | Target | Fairness attrs | Notes |
|---|---|---|---|---|---|
| CVD | `cardio_train.csv` (Kaggle cardiovascular disease, semicolon separated) | 70,000 | `cardio` | gender (coded 1/2), age | age stored in days -> years; impossible BP/height/weight values set to NaN; BMI derived from height/weight |
| Diabetes | Pima `diabetes.csv` | 768 | `Outcome` | age only (all-female cohort) | zeros in Glucose/BloodPressure/SkinThickness/Insulin/BMI are impossible -> NaN |
| Breast cancer | WDBC `breast-cancer.csv` | 569 | `diagnosis` (M=1) | none | tabular tumour measurements, no demographics |
| CKD | UCI `kidney_disease.csv` | 400 | `classification` (ckd=1) | age only | many missing values; tab/whitespace artefacts in strings |
| TB (tabular) | not provided | - | - | - | reports `dataset_missing`; TB is served by the image module |

## Image modules
| Module | Data | Classes | Checkpoint |
|---|---|---|---|
| TB chest X-ray | TBX11K chest X-rays (`data/raw/tb/tb_detection/images/{train,val}`) | health / sick / tb (3-class) | `artifacts/tb_image/best_model.pth` + `metrics.json` |
| Breast ultrasound | BUSI ultrasound images (`Dataset_BUSI_with_GT`) | benign / malignant / normal | `artifacts/breast_ultrasound/best_model.pth` + `metrics.json` |

Class names and image size are read from each `metrics.json` / checkpoint at runtime, so production does
**not** need `data/raw` (which is git-ignored and excluded from the Docker image).
Image training scripts: `backend/ml/train_tb_image.py`, `backend/ml/train_breast_ultrasound.py`.

Other files in the bundle were inspected but not used (alternatives): additional CVD sets (BRFSS-style `CVD_cleaned.csv`, `CVD Dataset.csv`),
the 1,659-row synthetic-looking CKD set (has sex/ethnicity - useful for fairness), symptom-based diabetes set, BCSC risk text files (no header),
and the nephrotoxic-drug CKD set. To switch a tabular disease, edit `PRIMARY` in `scripts/prepare_datasets.py`
(optionally add `data/raw/<disease>/meta.json` with `target`, `positive_label`, `drop`, `transforms`).
