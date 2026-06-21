import { $, num, VIOLATION_ORDER } from "../formatters.js";
import { getOverviewKpis, getRecentViolations, getMonthlyTrend, getSeverityBreakdown, violationDistribution } from "../data-bridge.js";
import { MOCK } from "../mock-data.js";
import { kpiCard, recentViolationItem, emptyState, animateConfBars } from "../components.js";
import { configureChartDefaults, doughnutChart, lineChart, severityDoughnut, renderSparklines, renderDistributionLegend } from "../charts.js";
import { navigate } from "../router.js";

const KPI_ICONS = {
  "Total Violations": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 9v4M12 17h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"/></svg>',
  "Vehicles Scanned": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17m-2 0a2 2 0 100 4 2 2 0 000-4zM17 17m-2 0a2 2 0 100 4 2 2 0 000-4z"/><path d="M5 17H3v-6l2-5h9l4 5h3v6h-2M9 17h6"/></svg>',
  "High Severity": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>',
  "Active Cameras": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M23 7l-7 5 7 5V7z"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>',
  "Challans Issued": '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 00-2 2v16a2 2 0 002 2h12a2 2 0 002-2V8z"/><path d="M14 2v6h6M16 13H8M16 17H8M10 9H8"/></svg>',
};

export async function initOverview() {
  configureChartDefaults();
  const kpiGrid = $("#overview-kpis");
  if (!kpiGrid) return;
  kpiGrid.innerHTML = '<div class="skeleton-card col-span-5"><div class="skel skel-line"></div></div>';

  const kpis = await getOverviewKpis();
  const summary = { by_type: kpis.byType, total: kpis.totalViolations };
  const distribution = violationDistribution(summary);
  const total = distribution.reduce((s, d) => s + d.count, 0);
  const severity = getSeverityBreakdown(summary);
  const trend = getMonthlyTrend(summary);
  const recent = await getRecentViolations(5);

  const cards = [
    { label: "Total Violations", value: num(kpis.totalViolations), trend: kpis.trends.totalViolations, spark: kpis.sparklines.totalViolations, color: "#0D6EFD" },
    { label: "Vehicles Scanned", value: num(kpis.vehiclesScanned), trend: kpis.trends.vehiclesScanned, spark: kpis.sparklines.vehiclesScanned, color: "#8B5CF6" },
    { label: "High Severity", value: num(kpis.highSeverity), trend: kpis.trends.highSeverity, spark: kpis.sparklines.highSeverity, color: "#EF4444" },
    { label: "Active Cameras", value: num(kpis.activeCameras), status: "Online", spark: null, color: "#10B981" },
    { label: "Challans Issued", value: num(kpis.challansIssued), trend: kpis.trends.challansIssued, spark: kpis.sparklines.challansIssued, color: "#F59E0B" },
  ];

  kpiGrid.innerHTML = cards.map((c) => kpiCard({
    label: c.label,
    value: c.value,
    trend: c.trend,
    sparkData: c.spark,
    icon: KPI_ICONS[c.label],
    color: c.color,
    status: c.status,
  })).join("");
  renderSparklines(kpiGrid);

  // Live feed placeholder with overlays
  const overlays = MOCK.liveOverlays.map((o) =>
    `<div class="live-overlay-box" style="${o.style};border-color:${o.color}">
      <span class="live-overlay-label" style="background:${o.color}">${o.label} ${o.pct}%</span>
    </div>`
  ).join("");
  const liveWrap = $("#overview-live-feed");
  if (liveWrap) {
    liveWrap.innerHTML = `
      <div class="live-feed-header">
        <span>Camera: Ring Road — AI Cam 47</span>
        <span class="live-badge">LIVE</span>
      </div>
      <div class="live-feed-body" id="overview-live-body">
        <img src="assets/live-placeholder.png" alt="Live traffic feed" class="live-feed-img" />
        <div class="live-overlays">${overlays}</div>
      </div>
      <div class="live-feed-footer">
        <span>FPS: 24</span><span>Res: 1080p</span>
        <button class="btn btn-sm btn-primary" id="overview-go-live">Go Live →</button>
      </div>
      <div class="camera-strip">${MOCK.cameras.map((c) =>
        `<button class="cam-thumb ${c.active ? "active" : ""}" data-cam="${c.id}">${c.name}</button>`
      ).join("")}</div>`;
    $("#overview-go-live")?.addEventListener("click", () => navigate("live"));
  }

  // Distribution chart — chart only; legend is custom HTML (no duplicate Chart.js legend)
  const typeEntries = VIOLATION_ORDER.map((k) => [k, kpis.byType?.[k] || 0]);
  doughnutChart("chart-overview-type", typeEntries, { legend: false, cutout: "72%" });
  const totalEl = $("#overview-dist-total");
  const footTotal = $("#overview-dist-foot-total");
  if (totalEl) totalEl.textContent = num(total);
  if (footTotal) footTotal.textContent = num(total);
  renderDistributionLegend("overview-dist-legend", distribution);

  // Recent violations
  const recentEl = $("#overview-recent");
  if (recentEl) {
    recentEl.innerHTML = recent.length
      ? recent.map(recentViolationItem).join("")
      : emptyState("No recent violations", "Analyze a frame or seed demo data.");
    recentEl.querySelectorAll(".recent-item").forEach((item) => {
      item.addEventListener("click", () => window.openEvidence?.(item.dataset.id));
    });
  }

  // Trend
  lineChart("chart-overview-trend", trend.map((t) => t.day), trend.map((t) => t.count));

  // Severity
  severityDoughnut("chart-overview-severity", severity);
  const sevCenter = $("#overview-severity-center");
  if (sevCenter) sevCenter.textContent = num(kpis.highSeverity);

  // Hotspots mini
  const hotspotsEl = $("#overview-hotspots");
  if (hotspotsEl) {
    hotspotsEl.innerHTML = MOCK.hotspots.map((h, i) =>
      `<div class="hotspot-rank"><span class="hotspot-num">${i + 1}</span><span class="hotspot-name">${h.name}</span><span class="hotspot-count">${h.count}</span></div>`
    ).join("");
  }

  // ANPR widget
  const anprInput = $("#overview-anpr-input");
  const anprBtn = $("#overview-anpr-btn");
  const runAnpr = () => {
    const q = anprInput?.value?.trim();
    if (q) navigate("anpr", { q });
    else navigate("anpr");
  };
  anprBtn?.addEventListener("click", runAnpr);
  anprInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") runAnpr(); });
  const tags = $("#overview-anpr-tags");
  if (tags) {
    tags.innerHTML = MOCK.recentSearches.map((p) =>
      `<button class="tag-chip" data-plate="${p}">${p}</button>`
    ).join("");
    tags.querySelectorAll(".tag-chip").forEach((btn) => {
      btn.addEventListener("click", () => navigate("anpr", { q: btn.dataset.plate }));
    });
  }

  animateConfBars();
}
