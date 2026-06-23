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

## Notes

- **SQLite** (`data/trace.db`) is ephemeral on free/starter unless you add a [Render Disk](https://render.com/docs/disks) mounted at `/app/data`.
- **Model weights** (`*.pt`) are not in git. `config/roboflow-fast.yaml` uses Roboflow for helmet + YOLO COCO auto-download for detection when VioVision weights are absent.
- Use **Starter** plan or higher (≥512 MB RAM recommended for ML stack).
- Rotate your Roboflow API key if it was ever exposed in chat or logs.
