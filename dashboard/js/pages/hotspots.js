import { $ } from "../formatters.js";
import { getViolationsByLocation, getAnalyticsSummary } from "../data-bridge.js";
import { emptyState } from "../components.js";
import { configureChartDefaults, hourChart } from "../charts.js";

export async function initHotspotsPage() {
  configureChartDefaults();
  const summary = await getAnalyticsSummary();
  const byLocation = await getViolationsByLocation();
  const rows = Object.entries(byLocation).sort((a, b) => b[1] - a[1]);

  const map = $("#hotspots-map");
  if (map) {
    map.innerHTML = rows.length
      ? `<svg viewBox="0 0 400 280" class="hotspots-svg">
        <rect width="400" height="280" fill="#F0F4F8" rx="12"/>
        ${rows.slice(0, 6).map(([loc, count], i) => {
          const x = 60 + (i % 3) * 120;
          const y = 70 + Math.floor(i / 3) * 90;
          const r = Math.min(28, 12 + count * 2);
          return `<g transform="translate(${x},${y})">
            <circle r="${r}" fill="${i === 0 ? "#EF4444" : "#F59E0B"}" opacity="0.85"/>
            <text text-anchor="middle" dy="4" fill="#fff" font-size="11" font-weight="600">${count}</text>
            <title>${loc}</title>
          </g>`;
        }).join("")}
      </svg>`
      : emptyState("No hotspot data", "Violations grouped by camera location appear here.");
  }

  const list = $("#hotspots-list");
  if (list) {
    list.innerHTML = rows.length
      ? rows.map(([name, count], i) => `
        <div class="hotspot-card">
          <span class="hotspot-rank-num">${i + 1}</span>
          <div class="hotspot-card-body">
            <div class="hotspot-card-name">${name}</div>
            <div class="hotspot-card-meta">${count} violation${count === 1 ? "" : "s"}</div>
          </div>
        </div>`).join("")
      : emptyState("No locations in database");
  }

  const table = $("#hotspots-cameras");
  if (table) {
    table.innerHTML = rows.length
      ? rows.map(([loc, count]) => `
        <tr><td>${loc}</td><td>${count}</td><td><span class="status-dot-inline online"></span> Online</td></tr>`).join("")
      : `<tr><td colspan="3">${emptyState("No camera data")}</td></tr>`;
  }

  hourChart("chart-hotspots-hour", summary?.by_hour || {});
}
