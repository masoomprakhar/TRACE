# TRACE — API + dashboard + ML stack (CPU). Tuned for Render Web Services.
# Set env vars in Render (see render.yaml / docs/DEPLOY_RENDER.md).
FROM python:3.11-slim

# OpenCV + common numeric runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Core + ML deps so /api/analyze works (YOLO, EasyOCR, Roboflow SDK).
COPY requirements.txt requirements-ml.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt

COPY . .
RUN pip install --no-cache-dir -e . \
    && chmod +x scripts/render_start.sh \
    && mkdir -p data/output

# Render injects PORT at runtime; local/docker-compose default to 8000.
ENV PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:' + __import__('os').environ.get('PORT','8000') + '/api/health')" || exit 1

CMD ["scripts/render_start.sh"]
