# TRACE on VPS + Docker (full demo)

Stable 24/7 deployment with persistent data. **Minimum: 2 GB RAM, 2 vCPU, 20 GB disk.**

Recommended providers (pick one):

| Provider | Plan | Cost | Region tip |
|----------|------|------|------------|
| [Hetzner CX23](https://www.hetzner.com/cloud) | 2 vCPU / 4 GB | ~€4/mo | Falkenstein or Singapore |
| [DigitalOcean](https://www.digitalocean.com) | Basic 2 GB | ~$12/mo | Bangalore `blr1` |
| [Oracle Cloud](https://www.oracle.com/cloud/free/) | Ampere A1 (free) | $0 | Mumbai if available |

---

## Step 1 — Create the VPS

1. Create an **Ubuntu 22.04** or **24.04** server.
2. Note the **public IP** (e.g. `165.22.x.x`).
3. Add your SSH key at create time (recommended).

---

## Step 2 — SSH into the server

```bash
ssh root@YOUR_VPS_IP
```

---

## Step 3 — Install Docker

```bash
apt-get update && apt-get install -y ca-certificates curl git
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
chmod a+r /etc/apt/keyrings/docker.asc

echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
  https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | tee /etc/apt/sources.list.d/docker.list > /dev/null

apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

---

## Step 4 — Clone TRACE and configure

```bash
cd /opt
git clone https://github.com/masoomprakhar/TRACE.git
cd TRACE

cat > .env <<'EOF'
ROBOFLOW_API_KEY=YOUR_ROBOFLOW_PRIVATE_KEY
ROBOFLOW_WORKSPACE=prakhar-parkar
ROBOFLOW_WORKFLOW_ID=general-segmentation-api-4
ROBOFLOW_HELMET_MODEL_ID=helmet-gj8do/2
ROBOFLOW_OCR_MODEL_ID=ocr-character-cgtzm/4
TRACE_CONFIG=config/roboflow-fast.yaml
TRACE_LOG_LEVEL=INFO
PYTHONUNBUFFERED=1
EOF

nano .env   # paste your real ROBOFLOW_API_KEY, save Ctrl+O Enter Ctrl+X
```

---

## Step 5 — Build and start (first run ~15–20 min)

```bash
docker compose -f docker-compose.prod.yml up -d --build
```

Watch logs:

```bash
docker compose -f docker-compose.prod.yml logs -f api
```

Wait for: `Uvicorn running on http://0.0.0.0:8000`

---

## Step 6 — Seed demo data

```bash
docker compose -f docker-compose.prod.yml exec api python -m trace_cv.cli seed-demo -n 40
```

---

## Step 7 — Open firewall port 8000

```bash
ufw allow OpenSSH
ufw allow 8000/tcp
ufw enable
```

In your cloud provider panel, also allow **inbound TCP 8000** in the security group / firewall.

**Test:** open `http://YOUR_VPS_IP:8000/#overview`

---

## Step 8 — HTTPS with Caddy (optional, recommended)

```bash
apt-get install -y debian-keyring debian-archive-keyring apt-transport-https curl
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list
apt-get update && apt-get install -y caddy
```

Point a domain's **A record** to your VPS IP, then:

```bash
cat > /etc/caddy/Caddyfile <<'EOF'
trace.yourdomain.com {
    reverse_proxy localhost:8000
}
EOF

systemctl reload caddy
```

Visit: `https://trace.yourdomain.com/#overview`

---

## Useful commands

```bash
# Status
docker compose -f docker-compose.prod.yml ps

# Logs
docker compose -f docker-compose.prod.yml logs -f api

# Restart after git pull
cd /opt/TRACE && git pull
docker compose -f docker-compose.prod.yml up -d --build

# Stop
docker compose -f docker-compose.prod.yml down

# Stop but keep data volume
docker compose -f docker-compose.prod.yml down
# (volume trace_data persists violations + SQLite)
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Build runs out of memory | Use 4 GB RAM plan or add 2 GB swap |
| `Cannot connect` on :8000 | Open port in `ufw` + cloud firewall |
| Analyze very slow | Normal on 2 GB CPU; first request loads models |
| OOM kill | Upgrade to 4 GB RAM |
| Empty dashboard | Run `seed-demo` again |

### Add swap (if only 2 GB RAM)

```bash
fallocate -l 2G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
```

---

## One-command bootstrap (Ubuntu)

From a fresh root SSH session:

```bash
curl -fsSL https://raw.githubusercontent.com/masoomprakhar/TRACE/main/scripts/vps_bootstrap.sh | bash
```

Then edit `/opt/TRACE/.env` with your API key and re-run compose.
