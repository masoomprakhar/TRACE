#!/bin/sh
# Render sets PORT (usually 10000); local/docker-compose default to 8000.
set -e
PORT="${PORT:-8000}"
echo "TRACE starting on 0.0.0.0:${PORT}"
exec uvicorn trace_cv.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT}" \
  --proxy-headers \
  --forwarded-allow-ips='*'
