import { $, $$, esc, fmtTime, deriveSeverity } from "../formatters.js";
import { getRecentViolations } from "../data-bridge.js";
import { navigate } from "../router.js";
import { emptyState } from "../components.js";

let filter = "all";
let _items = [];

export function initAlerts() {
  $("#alerts-mark-read")?.addEventListener("click", () => {
    filter = "read";
    $$(".alerts-tab").forEach((t) => t.classList.toggle("active", t.dataset.filter === "read"));
    renderAlerts();
  });
  $$(".alerts-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      $$(".alerts-tab").forEach((t) => t.classList.remove("active"));
      tab.classList.add("active");
      filter = tab.dataset.filter;
      renderAlerts();
    });
  });
}

function renderAlerts() {
  const list = $("#alerts-list");
  if (!list) return;
  let items = [..._items];
  if (filter === "high") {
    items = items.filter((a) => a.severity === "critical" || a.severity === "high");
  }

  list.innerHTML = items.length ? items.map((a) => `
    <div class="alert-item" data-id="${esc(a.id)}">
      <div class="alert-sev sev-${a.severity}"></div>
      <div class="alert-body">
        <div class="alert-title">${esc(a.title)}</div>
        <div class="alert-msg">${esc(a.message)}</div>
        <div class="alert-foot">
          <span class="alert-time">${fmtTime(a.time)}</span>
          <button class="btn btn-sm btn-ghost alert-action" data-id="${esc(a.id)}">View →</button>
        </div>
      </div>
    </div>`).join("") : emptyState("No violation alerts", "Recent detections appear here after analysis.");

  list.querySelectorAll(".alert-action").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      window.openEvidence?.(btn.dataset.id);
    });
  });
}

export async function initAlertsPage() {
  initAlerts();
  const recent = await getRecentViolations(30);
  _items = recent.map((r) => {
    const types = r.violation_types || [];
    const sev = deriveSeverity(types);
    return {
      id: r.id,
      severity: sev,
      title: r.violation_label || types.join(", ") || "Violation detected",
      message: `${r.location || "Camera"} · ${r.vehicle_type || "vehicle"}`,
      time: r.timestamp,
    };
  });
  renderAlerts();
}
