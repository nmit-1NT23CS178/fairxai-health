FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 APP_ENV=production
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY backend ./backend
COPY frontend ./frontend
COPY artifacts ./artifacts
COPY data/reports ./data/reports
RUN mkdir -p data/raw data/processed
EXPOSE 5000
CMD gunicorn backend.wsgi:app --bind 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120
