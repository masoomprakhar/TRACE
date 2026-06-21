import { api } from "../api.js";
import { $, prettify, REVIEW_THRESHOLD, pct, num } from "../formatters.js";
import { VIOLATION_ORDER, vinfo } from "../formatters.js";
import { toast } from "../components.js";
import { getEvalSummary } from "../data-bridge.js";

const MODEL_LABELS = { detector: "Vehicle Detector", ocr: "Plate OCR", helmet: "Helmet Model", seatbelt: "Seatbelt Model", plate: "Plate Detector" };
const MODEL_ORDER = ["detector", "ocr", "helmet", "seatbelt", "plate"];

export async function initSettingsPage() {
  try {
    const h = await api.get("/api/health");
    $("#settings-status").textContent = h.status === "ok" ? "Operational" : "Degraded";
    $("#settings-status").className = `status-pill ${h.status === "ok" ? "ok" : "warn"}`;
    $("#settings-version").textContent = h.version ? `v${h.version}` : "—";
    const grid = $("#settings-models");
    if (grid) {
      grid.innerHTML = MODEL_ORDER.map((key) => {
        const on = !!(h.models && h.models[key]);
        return `<div class="model-card ${on ? "on" : "off"}">
          <span class="model-led"></span>
          <span class="model-name">${MODEL_LABELS[key]}</span>
          <span class="model-state">${on ? "Loaded" : "Not loaded"}</span>
        </div>`;
      }).join("");
    }
  } catch (_) {
    $("#settings-status").textContent = "Offline";
  }

  const tax = $("#settings-taxonomy");
  if (tax) {
    tax.innerHTML = VIOLATION_ORDER.map((t) => {
      const { label, color } = vinfo(t);
      return `<span class="vbadge" style="--vbadge-color:${color}">${label}</span>`;
    }).join("");
  }

  $("#settings-threshold").textContent = `${Math.round(REVIEW_THRESHOLD * 100)}%`;

  const perf = $("#settings-performance");
  if (perf) {
    const ev = await getEvalSummary();
    const m = ev?.metrics || {};
    const rows = [
      ["Detection mAP@0.5", m.detection_map50],
      ["Motorcycle AP@0.5", m.motorcycle_ap50],
      ["Violation micro-F1", m.violation_micro_f1],
      ["No-helmet F1", m.no_helmet_f1],
      ["OCR exact match", m.ocr_exact_match],
      ["Latency (ms/frame)", m.latency_ms],
    ];
    perf.innerHTML = rows.map(([label, val]) =>
      `<div class="meta-row"><span>${label}</span><span>${val != null ? (typeof val === "number" ? (label.includes("F1") || label.includes("mAP") || label.includes("match") ? val.toFixed(4) : num(val)) : val) : "—"}</span></div>`
    ).join("") + (ev?.note ? `<p class="text-sm text-muted mt-2">${ev.note}</p>` : "");
  }

  const interval = localStorage.getItem("trace_live_interval") || "2000";
  const loc = localStorage.getItem("trace_default_location") || "Camera-01";
  const intervalInput = $("#settings-interval");
  const locInput = $("#settings-location");
  if (intervalInput) intervalInput.value = interval;
  if (locInput) locInput.value = loc;

  $("#settings-save")?.addEventListener("click", () => {
    localStorage.setItem("trace_live_interval", intervalInput?.value || "2000");
    localStorage.setItem("trace_default_location", locInput?.value || "Camera-01");
    const liveInterval = document.getElementById("live-interval");
    if (liveInterval) liveInterval.value = intervalInput?.value || "2000";
    toast("Settings saved.", "success");
  });

  $("#settings-clear-storage")?.addEventListener("click", () => {
    localStorage.clear();
    toast("Local storage cleared.", "info");
  });

  $("#settings-seed-hint")?.addEventListener("click", () => {
    toast("Run: python -m trace_cv.cli seed-demo -n 40", "info", 8000);
  });
}
