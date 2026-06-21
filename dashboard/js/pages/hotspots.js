import { $ } from "../formatters.js";
import { MOCK } from "../mock-data.js";
import { getViolationsByLocation, getAnalyticsSummary } from "../data-bridge.js";
import { emptyState } from "../components.js";
import { configureChartDefaults, hourChart } from "../charts.js";

export async function initHotspotsPage() {
  configureChartDefaults();
  const summary = await getAnalyticsSummary();
  const byLocation = await getViolationsByLocation();

  // Map markers
  const map = $("#hotspots-map");
  if (map) {
    map.innerHTML = `
      <svg viewBox="0 0 400 280" class="hotspots-svg">
        <rect width="400" height="280" fill="#F0F4F8" rx="12"/>
        <path d="M50 140 Q120 80 200 100 T350 130" stroke="#CBD5E1" stroke-width="8" fill="none"/>
        <path d="M80 200 Q200 160 320 190" stroke="#CBD5E1" stroke-width="6" fill="none"/>
        ${MOCK.hotspots.map((h, i) => {
          const x = 80 + i * 75;
          const y = 90 + (i % 2) * 60;
          return `<g class="hotspot-marker" transform="translate(${x},${y})">
            <circle r="18" fill="${i === 0 ? "#EF4444" : "#F59E0B"}" opacity="0.9"/>
            <text text-anchor="middle" dy="5" fill="#fff" font-size="12" font-weight="600">${i + 1}</text>
          </g>`;
        }).join("")}
      </svg>`;
  }

  const list = $("#hotspots-list");
  if (list) {
    list.innerHTML = MOCK.hotspots.map((h, i) => `
      <div class="hotspot-card">
        <span class="hotspot-rank-num">${i + 1}</span>
        <div class="hotspot-card-body">
          <div class="hotspot-card-name">${h.name}</div>
          <div class="hotspot-card-meta">${h.count} violations · Peak ${h.peakHour}</div>
        </div>
        <span class="hotspot-trend trend-${h.trend}">${h.trend === "up" ? "↑" : h.trend === "down" ? "↓" : "→"}</span>
      </div>`).join("");
  }

  const table = $("#hotspots-cameras");
  if (table) {
    const rows = Object.entries(byLocation).sort((a, b) => b[1] - a[1]);
    table.innerHTML = rows.length
      ? rows.map(([loc, count]) => `
        <tr><td>${loc}</td><td>${count}</td><td><span class="status-dot-inline online"></span> Online</td></tr>`).join("")
      : `<tr><td colspan="3">${emptyState("No camera data")}</td></tr>`;
  }

  hourChart("chart-hotspots-hour", summary?.by_hour || {});
}
