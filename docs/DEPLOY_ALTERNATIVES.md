# TRACE — Where to deploy (full demo prototype)

Render free/starter tiers often struggle with TRACE because the stack loads **PyTorch + YOLO + EasyOCR** (~1–2 GB RAM on first analyze). These options work better for a **full demo**.

## Quick pick

| Goal | Best option |
|------|-------------|
| Easiest cloud (like Render) | **Railway** |
| More RAM, Mumbai region | **Fly.io** (2 GB machine) |
| Most reliable for judges | **VPS + Docker** (Hetzner / DigitalOcean) |
| Zero cost, presentation day | **Local + Cloudflare Tunnel** |

---

## 1. Railway (recommended)

**Why:** Docker deploy like Render, but easier RAM upgrades and fewer port-detection quirks.

1. [railway.app](https://railway.app) → New Project → Deploy from GitHub → `masoomprakhar/TRACE`
2. Settings → **Memory: 2 GB** (Settings → Resources)
3. Variables → paste env block from `docs/DEPLOY_RENDER.md`
4. Deploy uses root `Dockerfile` + `railway.toml` automatically
5. Generate domain → open `https://xxx.up.railway.app/#overview`
6. Shell: `python -m trace_cv.cli seed-demo -n 40`

---

## 2. Fly.io (good for India latency)

**Why:** Pick **Mumbai (`bom`)** region; explicit **2 GB** VM in `fly.toml`.

```bash
# One-time (install flyctl: https://fly.io/docs/hands-on/install-flyctl/)
fly auth login
fly launch --no-deploy    # use existing fly.toml
fly secrets set ROBOFLOW_API_KEY=your_key
fly deploy
fly open
```

After deploy: `fly ssh console -C "python -m trace_cv.cli seed-demo -n 40"`

---

## 3. VPS — most reliable full demo (~$5–6/mo)

**Why:** Full CPU/RAM, no serverless limits, best for Flipkart live demo.

**Hetzner CX22**, **DigitalOcean Droplet** (2 GB RAM), or **Oracle Cloud** free ARM VM.

```bash
# On the VPS (Ubuntu 22.04+)
sudo apt update && sudo apt install -y docker.io docker-compose-plugin git
git clone https://github.com/masoomprakhar/TRACE.git && cd TRACE

cat > .env <<'EOF'
ROBOFLOW_API_KEY=your_key
ROBOFLOW_WORKSPACE=prakhar-parkar
ROBOFLOW_WORKFLOW_ID=general-segmentation-api-4
ROBOFLOW_HELMET_MODEL_ID=helmet-gj8do/2
ROBOFLOW_OCR_MODEL_ID=ocr-character-cgtzm/4
TRACE_CONFIG=config/roboflow-fast.yaml
TRACE_LOG_LEVEL=INFO
EOF

docker compose -f docker-compose.prod.yml up -d --build
docker compose -f docker-compose.prod.yml exec api python -m trace_cv.cli seed-demo -n 40
```

Open `http://YOUR_VPS_IP:8000/#overview` (or put Nginx/Caddy in front for HTTPS).

---

## 4. Local demo + public URL (free, best for presentation)

**Why:** Zero cloud RAM limits; your laptop runs the full pipeline; tunnel gives judges a URL.

```bash
cd TRACE
source .venv/bin/activate   # or: pip install -r requirements.txt -r requirements-ml.txt && pip install -e .
export TRACE_CONFIG=config/roboflow-fast.yaml
export ROBOFLOW_API_KEY=your_key
./scripts/judge_demo.sh
```

In another terminal:

```bash
# Cloudflare Tunnel (free, stable URL)
cloudflared tunnel --url http://127.0.0.1:8000
```

Or ngrok: `ngrok http 8000`

Share the `https://....trycloudflare.com` URL with judges.

---

## 5. Google Cloud Run (optional)

Works with the same `Dockerfile` but **cold starts are slow** (30–60s) when ML loads. Set **2 GiB memory**, min instances = 1 (costs more). Better for API than live dashboard demo.

---

## Environment variables (all platforms)

```env
ROBOFLOW_API_KEY=your_private_key
ROBOFLOW_WORKSPACE=prakhar-parkar
ROBOFLOW_WORKFLOW_ID=general-segmentation-api-4
ROBOFLOW_HELMET_MODEL_ID=helmet-gj8do/2
ROBOFLOW_OCR_MODEL_ID=ocr-character-cgtzm/4
TRACE_CONFIG=config/roboflow-fast.yaml
TRACE_LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
```

---

## Lighter config if RAM is tight

Use zero-key local mode (no Roboflow bill, smaller footprint):

```env
TRACE_CONFIG=config/default.yaml
```

YOLO COCO weights auto-download; EasyOCR runs locally. Helmet uses local path only if weights exist.

---

## Flipkart Grid presentation tip

For the live pitch, prefer **local `judge_demo.sh` + Cloudflare Tunnel** or a **2 GB VPS** — most reliable. Keep Render/Railway as backup links only.
