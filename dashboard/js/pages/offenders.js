import { $, esc, fmtTime, num } from "../formatters.js";
import { MOCK } from "../mock-data.js";
import { getTopPlates } from "../data-bridge.js";
import { badges, emptyState, plateChip } from "../components.js";
import { configureChartDefaults, barChart } from "../charts.js";
import { api } from "../api.js";

export async function initOffendersPage() {
  configureChartDefaults();
  const topPlates = await getTopPlates();
  const merged = topPlates.map((p, i) => ({
    plate: p.plate,
    count: p.count,
    lastSeen: p.last_seen,
    types: p.violations || MOCK.offenders[i]?.types || [],
    risk: Math.min(99, 40 + (p.count || 0) * 5),
  }));
  while (merged.length < 10) {
    const m = MOCK.offenders[merged.length];
    if (!m || merged.find((x) => x.plate === m.plate)) break;
    merged.push(m);
  }

  const tbody = $("#offenders-tbody");
  if (tbody) {
    tbody.innerHTML = merged.slice(0, 10).map((o, i) => `
      <tr class="clickable-row" data-plate="${esc(o.plate)}">
        <td><span class="rank-badge">${i + 1}</span></td>
        <td>${plateChip(o.plate, true)}</td>
        <td><strong>${num(o.count)}</strong></td>
        <td>${fmtTime(o.lastSeen)}</td>
        <td>${badges(o.types)}</td>
        <td><div class="risk-bar"><div class="risk-fill" style="width:${o.risk}%"></div></div><span class="text-xs">${o.risk}</span></td>
      </tr>`).join("");
    tbody.querySelectorAll(".clickable-row").forEach((row) => {
      row.addEventListener("click", () => showDetail(row.dataset.plate));
    });
  }

  barChart(
    "chart-offenders",
    merged.slice(0, 10).map((o) => o.plate),
    merged.slice(0, 10).map((o) => o.count),
    true,
    ["#EF4444", "#F87171", "#FB923C", "#FBBF24", "#FCD34D", "#A3E635", "#4ADE80", "#2DD4BF", "#38BDF8", "#818CF8"]
  );
}

async function showDetail(plate) {
  const panel = $("#offender-detail");
  if (!panel) return;
  panel.classList.remove("hidden");
  panel.innerHTML = '<div class="skeleton-card"></div>';
  try {
    const d = await api.get(`/api/plates/search?q=${encodeURIComponent(plate)}`);
    const item = d.items?.[0];
    panel.innerHTML = `
      <div class="card p-4">
        <div class="flex justify-between items-start mb-4">
          ${plateChip(plate, true)}
          <button class="btn btn-ghost btn-sm" id="close-offender-detail">✕</button>
        </div>
        ${item ? `
          <p class="text-sm text-muted mb-2">${item.count} incidents · Last seen ${fmtTime(item.last_seen)}</p>
          ${badges(item.violations || [])}
        ` : `<p class="text-muted">No detailed history in database.</p>`}
      </div>`;
    $("#close-offender-detail")?.addEventListener("click", () => panel.classList.add("hidden"));
  } catch (err) {
    panel.innerHTML = emptyState("Couldn't load plate history", err.message);
  }
}
