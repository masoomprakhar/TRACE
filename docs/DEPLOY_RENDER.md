# Deploy TRACE on Render

## One-click (Blueprint)

1. Push this repo to GitHub.
2. In [Render Dashboard](https://dashboard.render.com/) → **New** → **Blueprint**.
3. Connect `masoomprakhar/TRACE`.
4. When prompted, set **`ROBOFLOW_API_KEY`** (private key from [Roboflow Settings](https://app.roboflow.com/settings/api)).
5. Deploy. First build may take **10–15 minutes** (PyTorch + EasyOCR).

## Manual Web Service

| Setting | Value |
|---------|--------|
| Runtime | Docker |
| Health check | `/api/health` |
| Start command | *(default from Dockerfile — uses `$PORT`)* |

### Environment variables

```env
ROBOFLOW_API_KEY=<your private key>
ROBOFLOW_WORKSPACE=prakhar-parkar
ROBOFLOW_WORKFLOW_ID=general-segmentation-api-4
ROBOFLOW_HELMET_MODEL_ID=helmet-gj8do/2
ROBOFLOW_OCR_MODEL_ID=ocr-character-cgtzm/4
TRACE_CONFIG=config/roboflow-fast.yaml
TRACE_LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
WEB_CONCURRENCY=1
```

## After first deploy

Open **Shell** on the service and seed demo data (optional):

```bash
python -m trace_cv.cli seed-demo -n 40
```

Visit `https://<your-service>.onrender.com/#overview`.

## Troubleshooting

### "No open ports detected, continuing to scan…"

This is a **transient Render message** while the container boots. If you then see:

```
Uvicorn running on http://0.0.0.0:10000
```

the deploy succeeded. Render uses `PORT=10000` by default; `scripts/render_start.sh` binds to it automatically.

If deploy **fails** after 15 minutes:

1. Set **Health Check Path** to `/api/health` (fast liveness — does not load ML models).
2. Use **Starter** plan or higher (512 MB+ RAM).
3. **Clear build cache & deploy** after pulling the latest `main`.

### Site loads but analyze is slow / OOM

First image analysis loads YOLO + EasyOCR into memory. Upgrade to a plan with more RAM if the service restarts on upload.

## Notes

- **SQLite** (`data/trace.db`) is ephemeral on free/starter unless you add a [Render Disk](https://render.com/docs/disks) mounted at `/app/data`.
- **Model weights** (`*.pt`) are not in git. `config/roboflow-fast.yaml` uses Roboflow for helmet + YOLO COCO auto-download for detection when VioVision weights are absent.
- Use **Starter** plan or higher (≥512 MB RAM recommended for ML stack).
- Rotate your Roboflow API key if it was ever exposed in chat or logs.
