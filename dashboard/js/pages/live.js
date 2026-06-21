import { api } from "../api.js";
import { $, $$ } from "../formatters.js";
import { MOCK, addLiveAlert } from "../mock-data.js";
import { toast, violationCard, emptyState, animateConfBars } from "../components.js";
import { invalidateCache } from "../data-bridge.js";
import { updateBadges } from "../nav-badges.js";

export const liveState = {
  running: false,
  source: "device",
  stream: null,
  timer: null,
  analyzing: false,
  frames: 0,
  selectedCam: MOCK.cameras.find((c) => c.active) || MOCK.cameras[3],
};

let liveBound = false;

export function initLive() {
  if (liveBound) return;
  liveBound = true;

  $$(".live-src-btn").forEach((btn) =>
    btn.addEventListener("click", () => setLiveSource(btn.dataset.liveSrc)));
  $("#live-start-btn")?.addEventListener("click", () => startLive());
  $("#live-stop-btn")?.addEventListener("click", stopLive);
  $("#live-demo-start")?.addEventListener("click", () => {
    setLiveSource("device");
    startLive();
  });
  $("#live-cctv-start")?.addEventListener("click", () => {
    setLiveSource("cctv");
    $("#live-cctv-wrap")?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    toast("Enter a CCTV URL below, then press Start live.", "info", 5000);
  });

  document.addEventListener("click", (e) => {
    const thumb = e.target.closest?.(".cam-thumb-live");
    if (!thumb) return;
    const cam = MOCK.cameras.find((c) => String(c.id) === thumb.dataset.cam);
    if (cam) {
      liveState.selectedCam = cam;
      $$(".cam-thumb-live").forEach((b) => b.classList.toggle("active", b === thumb));
      updateCamLabel();
    }
  });
}

function updateCamLabel() {
  const label = $("#live-cam-label");
  if (!label) return;
  if (liveState.source === "device") {
    label.textContent = "Device camera (demo)";
  } else if (liveState.running) {
    label.textContent = liveState.selectedCam?.location || "CCTV stream";
  } else {
    label.textContent = liveState.selectedCam?.location || "CCTV — not connected";
  }
}

function setLiveSource(src) {
  if (liveState.running) stopLive();
  liveState.source = src;
  $$(".live-src-btn").forEach((b) => b.classList.toggle("active", b.dataset.liveSrc === src));
  $("#live-cctv-wrap")?.classList.toggle("hidden", src !== "cctv");
  $("#live-device-hint")?.classList.toggle("hidden", src !== "device");
  updateCamLabel();
}

function updateResolution(w, h) {
  const tag = $("#live-res-tag");
  if (tag && w && h) tag.textContent = `Res: ${w}×${h}`;
}

export async function startLive() {
  if (liveState.running) return;
  try { await api.post("/api/live/reset"); } catch (_) {}

  liveState.running = true;
  liveState.frames = 0;
  $("#live-start-btn")?.classList.add("hidden");
  $("#live-stop-btn")?.classList.remove("hidden");
  const dot = $("#live-dot");
  dot?.classList.remove("hidden");
  dot?.classList.add("on");
  $("#live-status-text").textContent = "Live";
  $("#live-idle")?.classList.add("hidden");

  const video = $("#live-video");
  const cctvImg = $("#live-cctv-img");

  if (liveState.source === "device") {
    cctvImg?.classList.add("hidden");
    cctvImg?.removeAttribute("src");
    try {
      liveState.stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: { ideal: "environment" },
          width: { ideal: 1280 },
          height: { ideal: 720 },
        },
        audio: false,
      });
      video.srcObject = liveState.stream;
      video.classList.remove("hidden");
      await video.play();
      video.onloadedmetadata = () => updateResolution(video.videoWidth, video.videoHeight);
      updateCamLabel();
    } catch (err) {
      stopLive();
      toast(
        err.name === "NotAllowedError"
          ? "Camera permission denied. Allow camera access in your browser, or use CCTV mode."
          : `Camera error: ${err.message}`,
        "error",
        6000
      );
      return;
    }
  } else {
    video?.classList.add("hidden");
    liveState.stream?.getTracks().forEach((t) => t.stop());
    liveState.stream = null;
    const url = ($("#live-cctv-url")?.value || "").trim();
    const qs = url ? `url=${encodeURIComponent(url)}` : "source=0";
    cctvImg.src = `/api/live/stream?${qs}&t=${Date.now()}`;
    cctvImg.classList.remove("hidden");
    cctvImg.onload = () => updateResolution(cctvImg.naturalWidth, cctvImg.naturalHeight);
    cctvImg.onerror = () => {
      if (liveState.running) {
        toast("CCTV stream failed — check the URL or switch to Device camera demo.", "error", 6000);
      }
    };
    updateCamLabel();
  }

  scheduleLiveAnalyze();
  toast(
    liveState.source === "device"
      ? "Device camera live — AI analysis running."
      : "CCTV stream connected — AI analysis running.",
    "success",
    3000
  );
}

export function stopLive() {
  liveState.running = false;
  if (liveState.timer) { clearTimeout(liveState.timer); liveState.timer = null; }
  liveState.stream?.getTracks().forEach((t) => t.stop());
  liveState.stream = null;
  const video = $("#live-video");
  video?.pause();
  if (video) video.srcObject = null;
  video?.classList.add("hidden");
  const cctvImg = $("#live-cctv-img");
  cctvImg?.removeAttribute("src");
  cctvImg?.classList.add("hidden");
  $("#live-idle")?.classList.remove("hidden");
  $("#live-start-btn")?.classList.remove("hidden");
  $("#live-stop-btn")?.classList.add("hidden");
  const dot = $("#live-dot");
  dot?.classList.add("hidden");
  dot?.classList.remove("on");
  $("#live-status-text").textContent = "Stopped";
  $("#live-fps-tag").textContent = "— fps";
  $("#live-res-tag").textContent = "—";
  $("#live-scan")?.classList.add("hidden");
  updateCamLabel();
}

function scheduleLiveAnalyze() {
  if (!liveState.running) return;
  const ms = parseInt($("#live-interval")?.value, 10) || 2000;
  liveState.timer = setTimeout(analyzeLiveFrame, ms);
}

function captureLiveBlob() {
  return new Promise((resolve, reject) => {
    const canvas = $("#live-canvas");
    const video = $("#live-video");
    const cctvImg = $("#live-cctv-img");
    let w = 0, h = 0, draw = null;
    if (liveState.source === "device" && video && !video.classList.contains("hidden") && video.videoWidth) {
      w = video.videoWidth;
      h = video.videoHeight;
      draw = (ctx) => ctx.drawImage(video, 0, 0, w, h);
    } else if (cctvImg && !cctvImg.classList.contains("hidden") && cctvImg.naturalWidth) {
      w = cctvImg.naturalWidth;
      h = cctvImg.naturalHeight;
      draw = (ctx) => ctx.drawImage(cctvImg, 0, 0, w, h);
    }
    if (!w || !h || !draw) return reject(new Error("no frame"));
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext("2d");
    draw(ctx);
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("capture failed"))), "image/jpeg", 0.85);
  });
}

async function analyzeLiveFrame() {
  if (!liveState.running || liveState.analyzing) { scheduleLiveAnalyze(); return; }
  liveState.analyzing = true;
  $("#live-scan")?.classList.remove("hidden");
  const t0 = performance.now();
  try {
    const blob = await captureLiveBlob();
    const persist = $("#live-persist")?.checked;
    const loc = liveState.source === "device"
      ? "Device-Camera"
      : (liveState.selectedCam?.location || "CCTV-Live");
    const r = await api.postFile("/api/live/frame", blob, {
      location: loc,
      tracking: "true",
      persist: persist ? "true" : "false",
      preview: "true",
    });
    renderLiveResult(r);
    liveState.frames += 1;
    const elapsed = (performance.now() - t0) / 1000;
    if (elapsed > 0) $("#live-fps-tag").textContent = `${(1 / elapsed).toFixed(1)} fps`;
    if (persist) { invalidateCache(); updateBadges(); }
    for (const v of r.violations || []) {
      if ((v.confidence || 0) >= 0.9) {
        addLiveAlert({ label: v.label || v.type, confidence: Math.round((v.confidence || 0) * 100) });
      }
    }
  } catch (err) {
    if (liveState.running && liveState.frames === 0) {
      toast(`Waiting for video frames… ${err.message}`, "info", 3000);
    } else if (liveState.running) {
      toast(`Live analyze: ${err.message}`, "error", 4000);
    }
  } finally {
    liveState.analyzing = false;
    $("#live-scan")?.classList.add("hidden");
    scheduleLiveAnalyze();
  }
}

function renderLiveResult(r) {
  const viol = (r.violations || []).length;
  const dets = (r.detections || []).length;
  $("#live-results-tag").innerHTML = viol
    ? `<span class="text-danger">${viol} flagged</span>` : "Clear";

  const sum = $("#live-summary");
  if (sum) {
    sum.classList.remove("hidden");
    sum.innerHTML = [
      { k: "Detections", v: dets },
      { k: "Violations", v: viol, danger: viol > 0 },
      { k: "Latency", v: r.processing_ms != null ? `${Math.round(r.processing_ms)} ms` : "—" },
      { k: "Frames", v: liveState.frames },
    ].map((c) => `<div class="live-metric"><div class="live-metric-k">${c.k}</div><div class="live-metric-v ${c.danger ? "danger" : ""}">${c.v}</div></div>`).join("");
  }

  const img = $("#live-annotated-img");
  const ann = $("#live-annotated-wrap");
  if (r.annotated_preview) { img.src = r.annotated_preview; ann?.classList.remove("hidden"); }
  else if (r.annotated_url) { img.src = r.annotated_url; ann?.classList.remove("hidden"); }

  const body = $("#live-results-body");
  if (!body) return;
  if (!viol) {
    body.innerHTML = emptyState("No violations this frame", `${dets} object${dets === 1 ? "" : "s"} tracked.`);
    return;
  }
  body.innerHTML = `<div class="violation-list">${r.violations.map(violationCard).join("")}</div>`;
  animateConfBars(body);
}

export function initLivePage() {
  initLive();
  const strip = $("#live-camera-strip");
  if (strip && !strip.children.length) {
    strip.innerHTML = MOCK.cameras.map((c) =>
      `<button type="button" class="cam-thumb-live ${c.active ? "active" : ""}" data-cam="${c.id}">${c.name}</button>`
    ).join("");
  }
  setLiveSource(liveState.source);
  if (!liveState.running) {
    $("#live-results-body").innerHTML = emptyState(
      "No live analysis yet",
      "Start device camera or CCTV — violations appear here in real time."
    );
  }
}

export function onLeaveLive() {
  if (liveState.running) stopLive();
}
