# FairXAI - Explainable & Fair AI for Healthcare Decision Support

Research prototype (Phase 2) built on the Phase-1 Flask + HTML/JS stack. **Not a medical device; outputs are AI-generated
estimates, not diagnoses.** One Flask service serves the REST API and the dashboard.

## What is implemented
| Area | Implementation |
|---|---|
| Data pipeline | `backend/ml/data_prep.py` - schema inspection, name normalisation, type coercion, duplicate removal, impossible-value -> NaN (rows kept), outlier *reporting*, target validation (never imputed), leakage checks, per-disease quality report (`data/reports/*_quality.json`) |
| Models | Logistic Regression (baseline), Decision Tree, Random Forest, XGBoost, SVM (small data only) tuned by randomized search + stratified CV; 60/20/20 stratified split, seed 42; selection on **validation** only; final metrics on untouched test set |
| Leakage safety | imputation / scaling / encoding live **inside** the sklearn `Pipeline`, fitted on training data only; class imbalance handled by class weights (no resampling of val/test) |
| Threshold & calibration | Platt scaling on validation; threshold = highest value keeping validation recall >= 90% (screening use-case) |
| XAI | SHAP (global bar + summary, local) and LIME (local), on the same fitted pipeline |
| Fairness | Fairlearn + Wilson CIs: group TPR/FPR/precision/accuracy/selection rate, demographic-parity / equal-opportunity / equalized-odds differences, subgroup sizes, small-group flags, post-processing mitigation *comparison* |
| Storage | MongoDB (PyMongo): `datasets`, `model_registry`, `predictions` (TTL retention), `fairness_audits`, `training_runs`. Model binaries + datasets stay on disk; Mongo holds metadata/references |
| Dashboard | 9 pages: Overview, Prediction, Performance, Explainability, Fairness, Dataset Explorer, History, Reports, Settings |
| Security | server-side validation, restricted CORS, admin token for upload/clean/train/delete, upload type/size limits, no credentials in errors, retention + delete endpoints |

## Quick start (local)
```bash
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                        # optional; works without MongoDB (in-memory fallback, dev only)

# 1) put datasets in place (from your Kaggle bundle) - copies the chosen primary file per disease
python scripts/prepare_datasets.py path/to/DataSets__kaggle_.zip

# 2) clean + train + evaluate + explain + fairness-audit every disease that has data (~3-4 min on 1 CPU)
python scripts/train_all.py            # or: python scripts/train_all.py ckd diabetes

# 3) run the app  ->  http://127.0.0.1:5000
python -m backend.wsgi

# 4) tests
python -m pytest -q
```
Train/clean/upload can also be triggered from **Settings & Status** in the UI.

## MongoDB
* **Local:** install MongoDB Community, run `mongod`, set `MONGO_URI=mongodb://localhost:27017`.
* **Atlas:** create a free M0 cluster -> Database Access (user) -> Network Access (allow your host) -> *Connect > Drivers* -> copy the `mongodb+srv://...` string into `MONGO_URI` (env var only, never in git).
* Without `MONGO_URI` the app uses an in-memory mock **only if `ALLOW_MEMORY_DB=1`**; the UI shows "DB: memory" so it never pretends data is persisted. Collections/indexes are created automatically at startup; model/dataset/fairness metadata is re-synced from `artifacts/` on every start.
* Prediction records expire automatically (`PREDICTION_RETENTION_DAYS`, TTL index); `DELETE /api/predictions/<id>` and `DELETE /api/predictions?older_than_days=N` remove them. Set `STORE_PREDICTION_INPUTS=0` to avoid storing patient inputs.

## API (all JSON: `{ok, data}` or `{ok:false, error}`)
`GET /api/health` - `/api/system` - `/api/diseases` - `/api/dashboard/summary` - `/api/models[?disease=&status=]` - `/api/models/<d>/metrics|schema[?version=]` -
`POST /api/predict/<d>` `{inputs:{...}, lime:false}` - `/api/explain/<d>/global` - `/api/fairness/<d>[?subgroup=sex|age_group]` -
`/api/datasets` - `/api/datasets/<d>/report` - `POST /api/datasets/<d>/upload|clean` (admin) - `POST /api/train/<d>` (admin) - `/api/train/runs` -
`/api/predictions[?disease=&from=&to=&model_version=&flagged=]` - `DELETE /api/predictions[/<id>]` (admin) - `/api/reports/<d>[?format=json]`.
Admin endpoints need header `X-Admin-Token` when `ADMIN_TOKEN` is set (mandatory in production).

## Deployment (single service, e.g. Render + MongoDB Atlas)
1. Train locally (`scripts/train_all.py`) and **commit `artifacts/` and `data/reports/`** (trained models are small; free hosts have ephemeral disks, so models must ship with the image). Do **not** commit `data/raw` (git-ignored).
   *Privacy note:* each artifact stores a 200-row sample of training records (needed for SHAP/LIME background). Only publish artifacts built from public datasets.
2. Push to GitHub. In Render: *New > Blueprint* -> select repo (uses `render.yaml` / `Dockerfile`).
3. Set env vars: `MONGO_URI` (Atlas string), `ADMIN_TOKEN` (long random string), `CORS_ORIGINS=https://<your-app>.onrender.com`.
4. Deploy; verify `https://<your-app>.onrender.com/api/health` shows `"database": {"mode": "mongodb"}`.
Equivalent: `docker build -t fairxai . && docker run -p 5000:5000 -e MONGO_URI=... -e ADMIN_TOKEN=... fairxai`.
Uploads/retraining on a free host write to an ephemeral disk and are lost on restart; retrain locally and redeploy.

## Datasets used (primary, one per disease)
See `docs/DATASETS.md`. **TB: no dataset was provided -> module reports `dataset_missing`; no model is claimed.**
Breast-cancer ultrasound images (BUSI) were provided but need a CNN pipeline (not implemented; no GPU/DL stack).

## Known limitations
* Only tabular pipelines; image models (BUSI, TB X-ray) and Grad-CAM are not implemented.
* No user accounts/roles (single admin token). `users` collection not created.
* Datasets are public benchmarks: no external validation; none of the models is clinically validated.
* Fairness is limited to attributes present (sex/age); several datasets have none or too few samples per group - the UI says so.
* Deployment configs are prepared but **not deployed/verified** by the author of this package.
