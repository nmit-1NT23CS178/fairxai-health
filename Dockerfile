FROM python:3.12-slim

# APP_ENV=production disables Flask debug and the in-memory database fallback.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    APP_ENV=production \
    PORT=5000 \
    TORCH_NUM_THREADS=1 \
    MALLOC_ARENA_MAX=2

# libgomp1 is required by XGBoost / scikit-learn wheels on slim images.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system fairxai \
    && useradd --system --gid fairxai --create-home fairxai

WORKDIR /app

# Dependencies first for layer caching. Versions are pinned in requirements.txt; the trained
# artifacts in artifacts/ require exactly these versions.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY frontend ./frontend
COPY artifacts ./artifacts
COPY data/reports ./data/reports

# Writable directories for the optional upload/clean/train endpoints (ephemeral on free hosts).
RUN mkdir -p data/raw data/processed \
    && chown -R fairxai:fairxai /app

USER fairxai
EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:' + os.environ.get('PORT','5000') + '/api/health', timeout=8).status == 200 else 1)"

# One worker keeps memory predictable on the free plan; threads handle concurrent requests.
CMD ["sh", "-c", "exec gunicorn backend.wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers ${WEB_CONCURRENCY:-1} --threads ${WEB_THREADS:-4} --timeout 120 --graceful-timeout 30 --access-logfile - --error-logfile -"]
