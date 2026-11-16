# FairXAI-Health - Deployment Guide

**Current deployment (verified 2026-11-16):**
<https://fairxai-health.onrender.com> — Render service `fairxai-health`, commit `36b6d5e`,
MongoDB Atlas connected (`persistent=true`), 4 tabular models live; image modules return a clear
503 on the 512 MB tier (upgrade to Standard 2 GB to enable them).

Single Flask service (REST API + dashboard + both image modules) on **Render**, persistent storage on
**MongoDB Atlas**. No separate frontend is required: the dashboard is served at `/`.

| Component | Where |
|---|---|
| Dashboard + API | Render Web Service `fairxai-health` (Docker, `render.yaml` / `Dockerfile`) |
| Tabular models | `artifacts/<disease>/<version>/model.joblib` (4 diseases, committed) |
| Image models | `artifacts/tb_image/best_model.pth`, `artifacts/breast_ultrasound/best_model.pth` (committed) |
| Evaluation reports | `data/reports/*_quality.json`, `artifacts/*/metrics.json` (committed) |
| Metadata + prediction history | MongoDB Atlas, database `fairxai` |

Runtime: Docker base `python:3.12-slim`, dependencies pinned in `requirements.txt`, CPU-only PyTorch
(`torch==2.15.0+cpu`, `torchvision==0.29.2+cpu` from `download.pytorch.org/whl/cpu`).
**Do not upgrade these versions without retraining the tabular models** - the `.joblib` artifacts were
built in this environment.

---

## 1. MongoDB Atlas (existing cluster)

Your cluster (from the account details):

| Setting | Value |
|---|---|
| Cluster host | `cluster0.gjl4o0o.mongodb.net` |
| Database user | `fairxai_admin` |
| Database | `fairxai` |
| URI | `mongodb+srv://fairxai_admin:<PASSWORD>@cluster0.gjl4o0o.mongodb.net/fairxai?retryWrites=true&w=majority` |

Steps:

1. <https://cloud.mongodb.com> -> **Project 0** -> **Cluster0** -> **Database Access**.
   Confirm `fairxai_admin` has **Read and write to any database** (or `readWrite` on `fairxai`).
   That is the minimum the app needs: it creates `datasets`, `model_registry`, `predictions`,
   `fairness_audits`, and `training_runs`, including indexes and a TTL index.
2. **Network Access**: Render's free/standard instances use dynamic egress, so add
   `0.0.0.0/0` (allow from anywhere), or Render's static outbound IPs if your plan provides them.
   Access is controlled by the strong database password.
3. **Connect -> Drivers -> Python** and copy the SRV string. URL-encode the password if it contains
   `@ : / ? # [ ] %`.
4. Do **not** put the URI in git. It belongs only in Render's environment variables (and optionally a
   local `.env`, which is git-ignored).

The application reads exactly these variables (`backend/app/config.py`):
`MONGO_URI`, `MONGO_DB`, `ALLOW_MEMORY_DB`, `ADMIN_TOKEN`, `CORS_ORIGINS`, `MAX_CONTENT_LENGTH`,
`PREDICTION_RETENTION_DAYS`, `STORE_PREDICTION_INPUTS`, `RANDOM_SEED`, `APP_ENV`, `LOG_LEVEL`,
`TORCH_NUM_THREADS`, `WEB_CONCURRENCY`, `WEB_THREADS`.

---

## 2. Render (existing service `fairxai-health`)

Target repository: **https://github.com/nmit-1NT23CS178/fairxai-health**, branch `main`.
The dashboard is already connected to that repository. Settings to confirm:

| Setting | Value |
|---|---|
| Service name | `fairxai-health` |
| Runtime | Docker |
| Region | Singapore |
| Root directory | *(blank / repository root)* |
| Health check path | `/api/health` |
| Plan | **Standard 2 GB recommended** (see sizing) |

Environment variables (Render dashboard -> Environment):

| Key | Value |
|---|---|
| `APP_ENV` | `production` |
| `MONGO_URI` | Atlas SRV string |
| `MONGO_DB` | `fairxai` |
| `ALLOW_MEMORY_DB` | `0` |
| `ADMIN_TOKEN` | long random string (`python -c "import secrets;print(secrets.token_urlsafe(32))"`) |
| `STORE_PREDICTION_INPUTS` | `0` (public demo: no patient inputs stored) |
| `PREDICTION_RETENTION_DAYS` | `90` |
| `CORS_ORIGINS` | `https://fairxai-health.onrender.com` (optional; same-origin works) |
| `WEB_CONCURRENCY` | `1` |
| `WEB_THREADS` | `4` |
| `TORCH_NUM_THREADS` | `1` |
| `LOG_LEVEL` | `INFO` |

`render.yaml` in the repository declares the same values; secrets remain `sync: false`.

First build installs CPU PyTorch plus the scientific stack; allow several minutes. Render's health
check is `GET /api/health`.

### Sizing (measured and confirmed on Render)

| Workload | Peak RSS |
|---|---|
| Tabular only (4 models + SHAP/LIME) | ~330 MB |
| Tabular + TB X-ray + breast ultrasound | **~767 MB** (measured locally, CPU) |

The deployed Free instance is **512 MB**. `/api/health` confirms `image_capacity_mb=512`, so the image
endpoints detect the limit and return:

```json
HTTP 503 {"error":{"code":"image_capacity","limit_mb":512,"required_mb":900,
          "message":"Image inference needs about 900 MB of RAM; ... upgrade ... to Standard (2 GB) ..."}}
```

This guard prevents the OOM that killed the worker during the first deployment attempt. Tabular
predictions, SHAP/LIME, fairness, Atlas history and the dashboard are unaffected and verified.

To enable the image modules, upgrade `fairxai-health` to **Standard (2 GB)** (paid, owner approval
required). No code change is needed: `/api/health` will report `image_models_available=true` and both
image endpoints will serve predictions with Grad-CAM. `IMAGE_MODELS_MIN_MB` defaults to `900`.

Do not upgrade the plan without the account owner's approval.

---

## 3. Verify the deployment

```bash
python scripts/verify_deployment.py --base-url https://fairxai-health.onrender.com --retries 2
```

Checks against the live URL: health (`status`, `database.mode == "mongodb"`, `persistent == true`,
4 tabular + 2 image models), readiness, registry, dashboard, all four tabular modules
(schema -> prediction -> SHAP -> LIME -> global explanation -> fairness -> metrics), image modules
(TB X-ray + ultrasound with Grad-CAM), history persistence in Atlas, invalid input (422),
unknown disease (404), untrained tabular TB (404), and admin authorization.

Manual:

```bash
curl https://fairxai-health.onrender.com/api/health
curl https://fairxai-health.onrender.com/api/ready          # 200
curl https://fairxai-health.onrender.com/api/diseases
```

Expected health excerpt:
```json
{"ok":true,"data":{"status":"ok","database":{"mode":"mongodb","connected":true,"persistent":true,"required":true},
 "models_loaded":4,"image_models":{"tb_image":true,"breast_ultrasound":true},"image_models_loaded":2}}
```

Dashboard flow: Prediction (tabular) -> probability + SHAP + LIME; TB Image / Breast Ultrasound ->
upload -> class + confidence + Grad-CAM; History -> record survives a restart because it is in Atlas.

---

## 4. Redeploy after GitHub updates

1. Push/merge to `main` on `nmit-1NT23CS178/fairxai-health`.
2. Render auto-deploys (`autoDeploy: true`) or click **Manual Deploy -> Deploy latest commit**.
3. Re-run `scripts/verify_deployment.py`.

Retraining models:

```bash
py -3.12 -m venv .venv && .venv\Scripts\activate
pip install -r requirements-dev.txt
python scripts/prepare_datasets.py path/to/DataSets__kaggle_.zip
python scripts/train_all.py
python scripts/verify_artifacts.py
git add artifacts data/reports && git commit -m "Retrain models" && git push
```

Image models are retrained with `backend/ml/train_tb_image.py` and
`backend/ml/train_breast_ultrasound.py` (PyTorch, CPU).

---

## 5. Local production rehearsal

```bash
docker build -t fairxai .
docker run --rm -p 5000:5000 -e MONGO_URI="mongodb+srv://..." -e ADMIN_TOKEN="..." \
  -e APP_ENV=production -e ALLOW_MEMORY_DB=0 fairxai
python scripts/verify_deployment.py --base-url http://127.0.0.1:5000
```

Without Docker, a real MongoDB was used during validation:

```bash
mongod --dbpath ./mongo-data --port 27017
set MONGO_URI=mongodb://127.0.0.1:27017 && set APP_ENV=production && set ALLOW_MEMORY_DB=0
python -m backend.wsgi
```

---

## 6. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `status == "degraded"`, `database.mode == "unavailable"` | `MONGO_URI` missing/invalid or Atlas IP allowlist blocks Render. |
| `/api/ready` returns 503 | Production requires a persistent database. |
| Build fails installing `torch==2.15.0+cpu` | The CPU wheel index must be reachable; `requirements.txt` sets `--extra-index-url https://download.pytorch.org/whl/cpu`. |
| `lime==0.2.0.1` build step | It is a pure-Python source distribution; no compiler required. |
| 502 / restarts during image upload | 512 MB instance out of memory. Use Standard 2 GB (see sizing). |
| `model_unavailable` for a disease | Artifact missing/incompatible; run `python scripts/verify_artifacts.py`. |
| `XGBoostError: input stream corrupted` | Library drift; restore the pinned versions and retrain. |
| 401 on train/upload/delete | Send `X-Admin-Token` (Settings page keeps it in browser session storage only). |
| First request slow | Free-plan cold start (if still on Free); use `--retries 2`. |
| Patient inputs in history | `STORE_PREDICTION_INPUTS=0` (already in the blueprint). |
