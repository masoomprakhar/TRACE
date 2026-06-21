#!/usr/bin/env bash
# Judge demo — seed DB, serve API + dashboard (run from repo root).
set -euo pipefail
cd "$(dirname "$0")/.."

export TRACE_CONFIG="${TRACE_CONFIG:-config/roboflow-fast.yaml}"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

echo "==> Seeding demo violations (40 records)…"
python3 -m trace_cv.cli seed-demo -n 40

if [[ ! -f data/eval/eval-summary.json ]] && [[ -f data/eval/results.json ]]; then
  echo "==> Writing eval-summary for Settings page…"
  python3 -c "
import json
from pathlib import Path
from trace_cv.evaluation.summary_export import write_eval_summary
write_eval_summary(json.loads(Path('data/eval/results.json').read_text()), Path('data/eval/eval-summary.json'))
"
fi

echo "==> Starting TRACE at http://127.0.0.1:8000"
echo "    Dashboard: http://127.0.0.1:8000/#overview"
echo "    Settings → Performance shows offline mAP/F1 metrics"
python3 -m trace_cv.cli serve --host 127.0.0.1 --port 8000
