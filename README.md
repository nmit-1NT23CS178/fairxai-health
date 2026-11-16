# FairXAI - Explainable & Fair AI for Healthcare Decision Support

Research prototype (Phase 2) built on the Phase-1 Flask + HTML/JS stack. **Not a medical device; outputs are AI-generated
estimates, not diagnoses.** One Flask service serves the REST API and the dashboard, including two image modules.

## What is implemented
| Area | Implementation |
|---|---|
| Data pipeline | `backend/ml/data_prep.py` - schema inspection, name normalisation, type coercion, duplicate removal, impossible-value -> NaN (rows kept), outlier *reporting*, target validation (never imputed), leakage checks, per-disease quality report (`data/reports/*_quality.json`) |
| Tabular models | Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost, SVM (small data only) tuned by randomized search + stratified CV; 60/20/20 stratified split, seed 42; selection on **validation** only; final metrics on untouched test set |
| Leakage safety | imputation / scaling / encoding live **inside** the sklearn `Pipeline`, fitted on training data only; class imbalance handled by class weights (no resampling of val/test) |
| Threshold & calibration | Platt scaling on validation; threshold = highest value keeping validation recall >= 90% (capped at 0.999 for near-separable data) |
| XAI (tabular) | SHAP (global bar + summary, local) and LIME (local), on the same fitted pipeline |
| Fairness | Fairlearn + Wilson CIs: group TPR/FPR/precision/accuracy/selection rate, demographic-parity / equal-opportunity / equalized-odds differences, subgroup sizes, small-group flags, post-processing mitigation *comparison* |
| Image models | **TB chest X-ray**: ResNet-18, 3 classes (health / sick / tb), Grad-CAM. **Breast ultrasound (BUSI)**: ResNet-18, 3 classes (benign / malignant / normal), Grad-CAM. Both CPU-only. |
| Storage | MongoDB (PyMongo): `datasets`, `model_registry`, `predictions` (TTL retention), `fairness_audits`, `training_runs`. Model binaries + datasets stay on disk; Mongo holds metadata/references |
| Dashboard | 11 pages: Overview, Disease Prediction, TB Image Prediction, Breast Ultrasound, Performance, Explainability, Fairness, Dataset Explorer, History, Reports, Settings |
| Security | server-side validation, restricted CORS, admin token for upload/clean/train/delete, upload type/size limits, no credentials in errors, retention + delete endpoints |

## Quick start (local)
Requires **Python 3.12** (matches the Docker image and the committed model artifacts).
```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Windows: py -3.12 -m venv .venv; .venv\Scripts\activate
pip install -r requirements-dev.txt                        # installs CPU-only torch/torchvision
cp .env.example .env                                        # optional; works without MongoDB (in-memory fallback, dev only)

# 1) put tabular datasets in place (from your Kaggle bundle)
python scripts/prepare_datasets.py path/to/DataSets__kaggle_.zip

# 2) clean + train + evaluate + explain + fairness-audit every disease that has data
python scripts/train_all.py            # or: python scripts/train_all.py ckd diabetes

# 3) run the app  ->  http://127.0.0.1:5000
python -m backend.wsgi

# 4) tests + artifact verification (tabular + image models)
python -m pytest -q
python scripts/verify_artifacts.py
```
Train/clean/upload can also be triggered from **Settings & Status** in the UI.

## MongoDB
* **Local:** install MongoDB Community, run `mongod`, set `MONGO_URI=mongodb://localhost:27017`.
* **Atlas:** create a free M0 cluster -> Database Access (user) -> Network Access (allow your hosts) -> *Connect > Drivers* -> copy the `mongodb+srv://...` string into `MONGO_URI` (env var only, never in git).
* Without `MONGO_URI` the app uses an in-memory mock **only if `ALLOW_MEMORY_DB=1`**; the UI shows "DB: memory" so it never pretends data is persisted. Collections/indexes are created automatically at startup; model/dataset/fairness metadata is re-synced from `artifacts/` on every start.
* Prediction records expire automatically (`PREDICTION_RETENTION_DAYS`, TTL index); `DELETE /api/predictions/<id>` and `DELETE /api/predictions?older_than_days=N` remove them. Set `STORE_PREDICTION_INPUTS=0` to avoid storing patient inputs.

## API (all JSON: `{ok, data}` or `{ok:false, error}`)
`GET /api/health` (includes `database`, tabular `models`, and `image_models`) - `/api/ready` - `/api/system` - `/api/diseases` - `/api/dashboard/summary` - `/api/models[?disease=&status=]` - `/api/models/<d>/metrics|schema[?version=]` -
`POST /api/predict/<d>` `{inputs:{...}, lime:false}` - `/api/explain/<d>/global` - `/api/fairness/<d>[?subgroup=sex|age_group]` -
`POST /api/predict/tb-image` (multipart `file`) - `POST /api/predict/breast-ultrasound` (multipart `file`) -
`/api/datasets` - `/api/datasets/<d>/report` - `POST /api/datasets/<d>/upload|clean` (admin) - `POST /api/train/<d>` (admin) - `/api/train/runs` -
`/api/predictions[?disease=&from=&to=&model_version=&flagged=]` - `DELETE /api/predictions[/<id>]` (admin) - `/api/reports/<d>[?format=json]`.
Admin endpoints need header `X-Admin-Token` when `ADMIN_TOKEN` is set (mandatory in production).

## Deployment (single service, Render + MongoDB Atlas)
The Docker image is built on **Python 3.12** with **exactly pinned** dependencies (including CPU-only
PyTorch from `download.pytorch.org/whl/cpu`). Startup loads model metadata and never retrains.

1. Trained models are committed (`artifacts/`), together with evaluation reports (`data/reports/`); `data/raw` is git-ignored.
   *Privacy note:* each tabular artifact stores a 100-row sample of training records (SHAP/LIME background).
2. Push to GitHub. In Render: **New > Blueprint** and select the repository (uses `render.yaml` + `Dockerfile`).
3. Set the secrets in Render: `MONGO_URI` (Atlas string), `ADMIN_TOKEN` (long random string), optionally `CORS_ORIGINS`.
   `ALLOW_MEMORY_DB=0` and `STORE_PREDICTION_INPUTS=0` are already set by the blueprint.
4. **Sizing:** the full app (4 tabular + 2 image modules) measured **~767 MB peak RSS** on CPU. The Free/Starter
   512 MB tier is only sufficient for tabular-only use; use **Standard (2 GB)** for image uploads.
5. Verify the live URL:
   ```bash
   python scripts/verify_deployment.py --base-url https://<service>.onrender.com --retries 2
   ```
   `/api/health` must report `"database": {"mode": "mongodb", "persistent": true}` and `/api/ready` must return 200.

See `DEPLOYMENT_GUIDE.md` for the exact Atlas + Render steps and `TEST_REPORT.md` for measured results.
Uploads/retraining on a host with an ephemeral disk are lost on restart; retrain locally and redeploy.

## Datasets used (primary, one per disease)
See `docs/DATASETS.md`. The tabular TB module reports `dataset_missing`; TB is served by the separate
image pipeline (`artifacts/tb_image/`), and breast ultrasound by `artifacts/breast_ultrasound/`.

## Known limitations
* Image models are a fast frozen-backbone baseline trained on public data; Grad-CAM is not a segmentation.
* No user accounts/roles (single admin token). `users` collection not created.
* Datasets are public benchmarks: no external validation; none of the models is clinically validated.
* Fairness is limited to attributes present (sex/age) and is only computed for tabular models.
* Render Free/Starter 512 MB cannot serve the image modules reliably; the current deployment detects
  this and returns HTTP 503 `image_capacity` for image uploads (tabular modules are unaffected).
  Upgrade to the 2 GB Standard plan to enable them; see sizing above.
