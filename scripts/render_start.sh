#!/bin/sh
# Render sets PORT; docker-compose and local runs default to 8000.
set -e
PORT="${PORT:-8000}"
exec uvicorn trace_cv.api.main:app --host 0.0.0.0 --port "$PORT"
