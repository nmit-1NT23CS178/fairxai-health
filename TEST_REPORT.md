# FairXAI-Health - Test Report

Date: 2026-11-16
Environment: Windows, **Python 3.12.10**, dependencies pinned in `requirements.txt`
(`scikit-learn 1.9.1`, `xgboost 3.4.1`, `shap 0.53.0`, `numpy 2.5.3`, `torch 2.15.0+cpu`,
`torchvision 0.29.2+cpu`).

Command run from the release tree:

```bash
python -m pytest -q
python scripts/verify_artifacts.py
```

## 1. Automated test suite

| Result | Count |
|---|---|
| Passed | **36** |
| Failed | **0** |
| Errors | **0** |
| Skipped | **0** |
| Duration | 33.09 s |

Coverage includes the original 25 tests plus:
* 6 production-configuration tests (`tests/test_production.py`): degraded DB reporting, `/api/ready`
  503 without a persistent DB, admin endpoints disabled without a token, security headers and
  `Cache-Control: no-store`, CORS allowlist, SPA fallback vs. 404 for missing assets.
* 5 image-model tests (`tests/test_image_api.py`): health reports both image models; TB X-ray and
  breast-ultrasound predictions with Grad-CAM; upload validation (415/400/422); and a 503
  `image_capacity` guard so a 512 MB instance cannot be OOM-killed. These exercise the real committed
  checkpoints and confirm TB classes no longer come from `data/raw`.

The pre-existing suite still covers: data cleaning/leakage rules, model training structure and
serialization, health/disease registry, per-disease schema isolation, prediction + SHAP + LIME,
invalid/incomplete inputs, model-unavailable and unknown-disease handling, metrics/explainability/
fairness/report/dashboard endpoints, DB failure degradation, admin protection, empty states, and the
full upload -> clean -> train -> register -> predict flow.

## 2. Shipped-model verification

`scripts/verify_artifacts.py` (no retraining; loads every committed artifact):

```
[PASS] cvd                 XGBoost                 p=0.3644 thr=0.2839 shap=12 lime=10
[PASS] diabetes            Logistic Regression     p=0.2054 thr=0.2590 shap=8  lime=8
[PASS] breast_cancer       Logistic Regression     p=0.0000 thr=1.0000 shap=12 lime=10
[PASS] ckd                 Random Forest           p=0.9999 thr=1.0000 shap=12 lime=10
[SKIP] tb (tabular): dataset_missing by design
[PASS] tb_image            class=sick       conf=0.5845 gradcam=yes
[PASS] breast_ultrasound   class=malignant  conf=0.6606 gradcam=yes

6/6 shipped models verified.
```

## 3. Tabular model evaluation (untouched test sets)

| Disease | Version | Model | Threshold | Accuracy | Precision | Recall | Specificity | F1 | ROC-AUC | PR-AUC | n |
|---|---|---|---|---|---|---|---|---|---|---|---|
| CVD | v20261009124218 | XGBoost | 0.2839 | 0.6762 | 0.6212 | 0.9025 | 0.4501 | 0.7359 | 0.8048 | 0.7888 | 13,996 |
| Diabetes | v20261009124339 | Logistic Regression | 0.2590 | 0.6558 | 0.5000 | 0.7736 | 0.5941 | 0.6074 | 0.7876 | 0.6726 | 154 |
| Breast cancer (tabular) | v20261009124423 | Logistic Regression | 1.0000 | 0.9737 | 0.9756 | 0.9524 | 0.9861 | 0.9639 | 0.9868 | 0.9679 | 114 |
| CKD | v20261009175112 | Random Forest | 0.99996 | 0.9375 | 1.0000 | 0.9000 | 1.0000 | 0.9474 | 1.0000 | 1.0000 | 80 |

Confusion matrices (TN, FP / FN, TP):
* CVD: `[[3151, 3850], [682, 6313]]`
* Diabetes: `[[60, 41], [12, 41]]`
* Breast (tabular): `[[71, 1], [2, 40]]`
* CKD: `[[30, 0], [5, 45]]`

Notes: precision/PR-AUC for CVD are shown in `metadata.json`; the threshold is chosen on validation
only and capped at 0.999 for decision-making when a stored threshold is ~1.0 (CKD, breast cancer), while
the stored threshold and its test metrics remain unchanged for reproducibility.

## 4. Image model evaluation

**TB chest X-ray** (`artifacts/tb_image/metrics.json`, 3 classes, ResNet-18 frozen backbone, CPU):

| Metric | Value |
|---|---|
| Validation accuracy | 0.8811 (1,800 images) |
| Validation ROC-AUC (macro) | 0.9721 |
| TB precision / recall / F1 | 0.519 / 0.830 / 0.638 |
| Health precision / recall / F1 | 0.964 / 0.933 / 0.948 |
| Sick precision / recall / F1 | 0.955 / 0.843 / 0.895 |
| Confusion matrix (health/sick/tb) | `[[746,9,45],[17,674,109],[11,23,166]]` |

**Breast ultrasound** (`artifacts/breast_ultrasound/metrics.json`, 3 classes, 156 test images):

| Metric | Value |
|---|---|
| Test accuracy | 0.8462 |
| Balanced accuracy | 0.8578 |
| Macro F1 | 0.8395 |
| Benign / malignant / normal F1 | 0.871 / 0.791 / 0.857 |
| Confusion matrix (benign/malignant/normal) | `[[74,10,4],[6,34,2],[2,0,24]]` |

These are public-dataset research results, not clinical validation.

## 5. End-to-end API checks (local, same code as deployment)

| Check | Result |
|---|---|
| App startup (production config) | Pass |
| Frontend served at `/` and SPA routes | Pass |
| `/api/health`, `/api/ready` | Pass |
| `/api/diseases`, `/api/dashboard/summary` | Pass |
| Tabular predictions (CVD, diabetes, breast, CKD) | Pass |
| Probability outputs in [0,1] + threshold flag | Pass |
| SHAP local + global contributions | Pass |
| LIME local explanations | Pass |
| Fairlearn fairness metrics (CVD sex/age, diabetes age, CKD age) | Pass |
| Breast tabular fairness | `not_available` (dataset has no age/sex) — reported honestly |
| TB X-ray + breast ultrasound + Grad-CAM | Pass |
| Invalid input -> 422; bad upload -> 415/422; unknown disease -> 404 | Pass |
| Admin authorization -> 401 without token | Pass |

## 6. MongoDB verification (real server)

A local MongoDB 7.0.14 server was used (same PyMongo code path as Atlas), production config
(`APP_ENV=production`, `ALLOW_MEMORY_DB=0`, `MONGO_URI=mongodb://127.0.0.1:27018`):

* `GET /api/health` -> `database.mode=mongodb`, `connected=true`, `persistent=true`, `required=true`.
* Collections auto-created: `datasets` (4 docs), `model_registry` (9 versions),
  `fairness_audits` (9), `predictions`, `training_runs`.
* Indexes on `predictions`: `created_at_-1`, `disease_1_model_version_1`, `prediction_id_1` (unique),
  `expires_at_1` (TTL).
* Prediction inserted via `POST /api/predict/cvd`; retrieved via `/api/predictions?disease=cvd`.
* **Restart test:** a second, fresh process re-read the same `prediction_id` from MongoDB and the
  dashboard summary returned `predictions.total=1`, proving persistence across application restart.

**Atlas status: VERIFIED** on the live deployment (see section 9). The same integration was also
reproduced locally against MongoDB 7.0.14.

## 7. Memory measurement (deployment sizing)

| Scenario | Peak RSS |
|---|---|
| App created (tabular stack imported) | 211 MB |
| + CVD prediction with SHAP/LIME | 326 MB |
| + TB X-ray inference | 648 MB |
| + Breast ultrasound inference | **767 MB** |

Conclusion: Render Free/Starter **512 MB is insufficient** for the image modules. Standard 2 GB is
recommended, pending owner approval. Tabular-only usage fits in ~330 MB.

## 8. Deployment status

| Item | Status |
|---|---|
| GitHub original repo (`nmit-1NT23CS178/fairxai-health`) | **UPDATED** — `main` at code release `36b6d5e` / docs commit |
| Render service | **DEPLOYED & VERIFIED** — `https://fairxai-health.onrender.com` |
| MongoDB Atlas | **VERIFIED** — mode `mongodb`, persistent, history survived deployments |
| Public URL | **https://fairxai-health.onrender.com** |
| Instance | Free, 512 MB (image modules blocked by design on this tier) |

## 9. Live deployment verification (2026-11-16)

`python scripts/verify_deployment.py --base-url https://fairxai-health.onrender.com`

| Result | Count |
|---|---|
| Passed | **35** |
| Failed | **0** |
| Skipped | **1** (image models on the 512 MB instance) |

Verified against the public URL:
* `/api/health`: `status=ok`, `database.mode=mongodb`, `persistent=true`, commit `36b6d5e`,
  4 tabular models loaded, both image checkpoints present.
* `/api/ready`: HTTP 200.
* Disease registry and dashboard: 5 supported, 4 trained.
* CVD, diabetes, breast cancer, CKD: schema -> prediction -> SHAP (local + global) -> LIME ->
  fairness -> metrics -> **history read back from Atlas** -> invalid input rejected with 422.
* Atlas persistence across deployments: prediction counts increased across the redeploy
  (CVD 2 -> 3, diabetes 3 -> 5, CKD 2 -> 3; dashboard total 16), proving records survive
  Render restarts.
* Unknown disease -> 404, untrained tabular TB -> 404.
* Frontend `/` and `/js/app.js` served; TB Image and Breast Ultrasound pages present.
* Image endpoints on 512 MB: `POST /api/predict/tb-image` and `/api/predict/breast-ultrasound`
  return **HTTP 503** `{"code":"image_capacity","limit_mb":512,"required_mb":900}` instead of
  crashing the service (verified directly with `curl`).
* No fabricated results, no fake predictions: every number above came from the live API and the
  committed model metadata.

## 10. Remaining limitation

The image modules are deployed but intentionally refused on the 512 MB instance because measured peak
RSS is ~767 MB. To enable them, upgrade the Render service to **Standard (2 GB)** (paid, owner
approval required) and set `IMAGE_MODELS_MIN_MB=900` (already the default). No code change is needed;
`/api/health` will then report `image_models_available=true` and the image endpoints will serve
predictions with Grad-CAM.
