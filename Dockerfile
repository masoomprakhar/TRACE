# TRACE — core API + dashboard image (CPU).
# For real detection/OCR, install the ML extras (see requirements-ml.txt)
# in a derived image or at runtime.
FROM python:3.11-slim

# OpenCV runtime libs.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 8000
CMD ["uvicorn", "trace_cv.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
