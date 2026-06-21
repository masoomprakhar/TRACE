import { $, num, VIOLATION_ORDER } from "../formatters.js";
import { getOverviewKpis, getRecentViolations, getHourlyTrend, getSeverityBreakdown, violationDistribution, getViolationsByLocation, getTopPlates } from "../data-bridge.js";
import { kpiCard, recentViolationItem, emptyState, animateConfBars } from "../components.js";
import { configureChartDefaults, doughnutChart, lineChart, severityDoughnut, renderDistributionLegend, renderSparklines } from "../charts.js";
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
  const summary = { by_type: kpis.byType, total: kpis.totalViolations, by_hour: kpis.byHour };
  const distribution = violationDistribution(summary);
  const total = distribution.reduce((s, d) => s + d.count, 0);
  const severity = getSeverityBreakdown(summary);
  const trend = getHourlyTrend(summary);
  const recent = await getRecentViolations(5);
  const byLocation = await getViolationsByLocation();

  const cards = [
    {
      label: "Total Violations",
      value: num(kpis.totalViolations),
      color: "#0D6EFD",
      trend: kpis.trends?.totalViolations,
      sparkData: kpis.sparklines?.totalViolations,
    },
    {
      label: "Vehicles Scanned",
      value: num(kpis.vehiclesScanned),
      color: "#8B5CF6",
      trend: kpis.trends?.vehiclesScanned,
      sparkData: kpis.sparklines?.vehiclesScanned,
    },
    {
      label: "High Severity",
      value: num(kpis.highSeverity),
      color: "#EF4444",
      trend: kpis.trends?.highSeverity,
      sparkData: kpis.sparklines?.highSeverity,
    },
    {
      label: "Active Cameras",
      value: num(kpis.activeCameras),
      color: "#10B981",
      status: kpis.activeCameras ? "Online" : "—",
    },
    {
      label: "Challans Issued",
      value: num(kpis.challansIssued),
      color: "#F59E0B",
      trend: kpis.trends?.challansIssued,
      sparkData: kpis.sparklines?.challansIssued,
    },
  ];

  kpiGrid.innerHTML = cards.map((c) => kpiCard({
    label: c.label,
    value: c.value,
    icon: KPI_ICONS[c.label] || "",
    color: c.color,
    status: c.status,
    trend: c.trend,
    sparkData: c.sparkData,
  })).join("");
  renderSparklines(kpiGrid);

  const liveWrap = $("#overview-live-feed");
  if (liveWrap) {
    liveWrap.innerHTML = `
      <div class="live-feed-header">
        <span>Camera: Ring Road — Junction Cam 01</span>
        <span class="live-badge">LIVE</span>
      </div>
      <div class="live-feed-body" id="overview-live-body">
        <img src="assets/live-placeholder.png" alt="Live traffic feed" class="live-feed-img" />
      </div>
      <div class="live-feed-footer">
        <span>${kpis.processingFps ? `~${Number(kpis.processingFps).toFixed(1)} FPS` : "1080p · Demo feed"}</span>
        <button class="btn btn-sm btn-primary" id="overview-go-live">Open Live Monitor →</button>
      </div>`;
    $("#overview-go-live")?.addEventListener("click", () => navigate("live"));
  }

  const typeEntries = VIOLATION_ORDER.map((k) => [k, kpis.byType?.[k] || 0]);
  doughnutChart("chart-overview-type", typeEntries, { legend: false, cutout: "72%" });
  const totalEl = $("#overview-dist-total");
  const footTotal = $("#overview-dist-foot-total");
  if (totalEl) totalEl.textContent = num(total);
  if (footTotal) footTotal.textContent = num(total);
  renderDistributionLegend("overview-dist-legend", distribution);

  const recentEl = $("#overview-recent");
  if (recentEl) {
    recentEl.innerHTML = recent.length
      ? recent.map(recentViolationItem).join("")
      : emptyState("No recent violations", "Run: python -m trace_cv.cli seed-demo -n 40");
    recentEl.querySelectorAll(".recent-item").forEach((item) => {
      item.addEventListener("click", () => window.openEvidence?.(item.dataset.id));
    });
  }

  if (trend.length) {
    lineChart("chart-overview-trend", trend.map((t) => t.day), trend.map((t) => t.count));
  }

  severityDoughnut("chart-overview-severity", severity);
  const sevCenter = $("#overview-severity-center");
  if (sevCenter) sevCenter.textContent = num(kpis.highSeverity);

  const hotspotsEl = $("#overview-hotspots");
  if (hotspotsEl) {
    const rows = Object.entries(byLocation).sort((a, b) => b[1] - a[1]).slice(0, 5);
    hotspotsEl.innerHTML = rows.length
      ? rows.map(([name, count], i) =>
        `<div class="hotspot-rank"><span class="hotspot-num">${i + 1}</span><span class="hotspot-name">${name}</span><span class="hotspot-count">${count}</span></div>`
      ).join("")
      : emptyState("No location data yet", "Violations will group by camera location.");
  }

  const anprInput = $("#overview-anpr-input");
  const anprBtn = $("#overview-anpr-btn");
  const runAnpr = () => {
    const q = anprInput?.value?.trim();
    navigate("anpr", q ? { q } : {});
  };
  anprBtn?.addEventListener("click", runAnpr);
  anprInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") runAnpr(); });

  const tags = $("#overview-anpr-tags");
  if (tags) {
    const plates = await getTopPlates();
    const top = plates.slice(0, 6).map((p) => p.plate).filter(Boolean);
    tags.innerHTML = top.length
      ? top.map((p) => `<button class="tag-chip" data-plate="${p}">${p}</button>`).join("")
      : '<span class="text-muted text-sm">No plates in database yet</span>';
    tags.querySelectorAll(".tag-chip").forEach((btn) => {
      btn.addEventListener("click", () => navigate("anpr", { q: btn.dataset.plate }));
    });
  }

  animateConfBars();
}
