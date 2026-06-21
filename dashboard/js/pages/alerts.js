import { $, $$, esc, fmtTime } from "../formatters.js";
import { MOCK, getUnreadAlertCount, markAllAlertsRead } from "../mock-data.js";
import { navigate } from "../router.js";
import { updateBadges } from "../nav-badges.js";

let filter = "all";

export function initAlerts() {
  $("#alerts-mark-read")?.addEventListener("click", () => {
    markAllAlertsRead();
    renderAlerts();
    updateBadges();
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
  let items = [...MOCK.alerts];
  if (filter === "unread") items = items.filter((a) => !a.read);
  else if (filter === "read") items = items.filter((a) => a.read);

  list.innerHTML = items.length ? items.map((a) => `
    <div class="alert-item ${a.read ? "read" : ""}" data-id="${esc(a.id)}">
      <div class="alert-sev sev-${a.severity}"></div>
      <div class="alert-body">
        <div class="alert-title">${esc(a.title)}</div>
        <div class="alert-msg">${esc(a.message)}</div>
        <div class="alert-foot">
          <span class="alert-time">${fmtTime(a.time)}</span>
          ${a.action ? `<button class="btn btn-sm btn-ghost alert-action" data-view="${a.action.view}">View →</button>` : ""}
        </div>
      </div>
    </div>`).join("") : '<div class="empty-state"><p>No alerts in this tab.</p></div>';

  list.querySelectorAll(".alert-item").forEach((item) => {
    item.addEventListener("click", () => {
      const alert = MOCK.alerts.find((a) => a.id === item.dataset.id);
      if (alert) alert.read = true;
      item.classList.add("read");
      updateBadges();
    });
  });
  list.querySelectorAll(".alert-action").forEach((btn) => {
    btn.addEventListener("click", (e) => {
      e.stopPropagation();
      navigate(btn.dataset.view);
    });
  });
}

export function initAlertsPage() {
  initAlerts();
  renderAlerts();
}
