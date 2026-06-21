import { api } from "./api.js";
import { $, $$, esc, fmtTime, pct } from "./formatters.js";
import { toast, imageFallback, badges, plateChip } from "./components.js";
import { MOCK } from "./mock-data.js";
import { parseHash, updatePageHeader, setSwitchHandler, VALID } from "./router.js";
import { updateBadges } from "./nav-badges.js";
import { initOverview } from "./pages/overview.js";
import { initLive, initLivePage, onLeaveLive } from "./pages/live.js";
import { initViolationsPage } from "./pages/violations.js";
import { initAnprPage } from "./pages/anpr.js";
import { initEvidencePage } from "./pages/evidence.js";
import { initReportsPage } from "./pages/reports.js";
import { initHotspotsPage } from "./pages/hotspots.js";
import { initOffendersPage } from "./pages/offenders.js";
import { initAlertsPage } from "./pages/alerts.js";
import { initSettingsPage } from "./pages/settings.js";

const loaded = {};

async function pollHealth() {
  try {
    const h = await api.get("/api/health");
    const dot = $("#header-status-dot");
    if (dot) dot.className = "status-dot-inline " + (h.status === "ok" ? "online" : "warn");
  } catch (_) {
    $("#header-status-dot")?.classList.remove("online");
  }
}

function updateClock() {
  const el = $("#header-clock");
  if (!el) return;
  const d = new Date();
  el.textContent = d.toLocaleString(undefined, {
    day: "numeric", month: "short", year: "numeric",
    hour: "2-digit", minute: "2-digit",
  });
}

function initShell() {
  // Sidebar nav
  $$(".nav-item").forEach((n) => {
    n.addEventListener("click", () => {
      const view = n.dataset.view;
      location.hash = `#${view}`;
    });
  });

  // Mobile nav
  $$(".mobile-nav-item").forEach((n) => {
    n.addEventListener("click", () => {
      location.hash = `#${n.dataset.view}`;
    });
  });

  $("#sidebar-toggle")?.addEventListener("click", () => {
    document.body.classList.toggle("sidebar-open");
  });

  // Notifications dropdown
  $("#notif-btn")?.addEventListener("click", (e) => {
    e.stopPropagation();
    $("#notif-dropdown")?.classList.toggle("open");
  });
  document.addEventListener("click", () => $("#notif-dropdown")?.classList.remove("open"));

  renderNotifDropdown();
  initModal();
  initLive();

  // Apply saved settings to live interval
  const savedInterval = localStorage.getItem("trace_live_interval");
  if (savedInterval) {
    const li = $("#live-interval");
    if (li) li.value = savedInterval;
  }
}

function renderNotifDropdown() {
  const dd = $("#notif-dropdown");
  if (!dd) return;
  const unread = MOCK.alerts.filter((a) => !a.read).slice(0, 5);
  dd.innerHTML = unread.length
    ? unread.map((a) => `<div class="notif-item"><strong>${esc(a.title)}</strong><span>${esc(a.message)}</span></div>`).join("")
    : '<div class="notif-item muted">No new notifications</div>';
}

async function switchView(view, params = {}) {
  if (!VALID.includes(view)) view = "overview";
  if (view !== "live") onLeaveLive();

  $$(".view").forEach((v) => v.classList.toggle("active", v.id === `view-${view}`));
  $$(".nav-item, .mobile-nav-item").forEach((n) =>
    n.classList.toggle("active", n.dataset.view === view));
  updatePageHeader(view);
  window.scrollTo({ top: 0, behavior: "smooth" });

  if (view === "overview") await initOverview();
  if (view === "live") initLivePage();
  if (view === "violations" && !loaded.violations) { loaded.violations = true; initViolationsPage(); }
  else if (view === "violations") { const { loadViolations } = await import("./pages/violations.js"); loadViolations(); }
  if (view === "anpr") initAnprPage(params);
  if (view === "evidence" && !loaded.evidence) { loaded.evidence = true; initEvidencePage(); }
  else if (view === "evidence") { const { loadGallery } = await import("./pages/evidence.js"); loadGallery(); }
  if (view === "reports") initReportsPage();
  if (view === "hotspots") await initHotspotsPage();
  if (view === "offenders") await initOffendersPage();
  if (view === "alerts") initAlertsPage();
  if (view === "settings") await initSettingsPage();

  updateBadges();
}

function initModal() {
  const modal = $("#drawer");
  $("#drawer-close")?.addEventListener("click", closeDrawer);
  modal?.addEventListener("click", (e) => { if (e.target === modal) closeDrawer(); });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
}

function closeDrawer() {
  $("#drawer")?.classList.remove("open");
}

window.openEvidence = async function openEvidence(id) {
  const modal = $("#drawer");
  const img = $("#drawer-img");
  const meta = $("#drawer-meta");
  if (!id || !modal) return;

  if (img) {
    img.src = `/api/violations/${encodeURIComponent(id)}/evidence`;
    img.onerror = () => { img.onerror = null; img.replaceWith(imageFallback()); };
  }
  if (meta) meta.innerHTML = '<div class="skel skel-line"></div>'.repeat(4);
  modal.classList.add("open");

  try {
    const r = await api.get(`/api/violations/${encodeURIComponent(id)}`);
    $("#drawer-title").textContent = r.violation_label || "Violation record";
    if (meta) {
      meta.innerHTML = `
        <div class="mb-3">${badges(r.violation_types)}</div>
        <div class="mb-3">${plateChip(r.plate_number, true)}</div>
        <div class="drawer-rows">
          <div class="meta-row"><span>Time</span><span>${fmtTime(r.timestamp)}</span></div>
          <div class="meta-row"><span>Location</span><span>${esc(r.location || "—")}</span></div>
          <div class="meta-row"><span>Vehicle</span><span>${esc(r.vehicle_type || "—")}</span></div>
          <div class="meta-row"><span>Confidence</span><span>${pct(r.confidence)}</span></div>
        </div>`;
    }
  } catch (err) {
    if (meta) meta.innerHTML = `<p class="text-danger">${esc(err.message)}</p>`;
  }
};

function boot() {
  initShell();
  setSwitchHandler(switchView);

  window.addEventListener("hashchange", () => {
    const { view, params } = parseHash();
    switchView(view, params);
  });

  pollHealth();
  setInterval(pollHealth, 10000);
  updateClock();
  setInterval(updateClock, 30000);
  updateBadges();

  const { view, params } = parseHash();
  switchView(view, params);
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", boot);
} else {
  boot();
}
