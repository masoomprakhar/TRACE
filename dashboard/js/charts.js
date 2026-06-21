import { $, vinfo, VIOLATION_ORDER, prettify, esc } from "./formatters.js";

const registry = {};

export function configureChartDefaults() {
  if (!window.Chart) return;
  Chart.defaults.color = "#6B7280";
  Chart.defaults.font.family = "'Inter', system-ui, sans-serif";
  Chart.defaults.font.size = 11;
  Chart.defaults.borderColor = "#E8ECF1";
  Chart.defaults.plugins.legend.display = false;
  Chart.defaults.plugins.tooltip = {
    backgroundColor: "#111827",
    borderColor: "#374151",
    borderWidth: 1,
    titleColor: "#F9FAFB",
    bodyColor: "#F9FAFB",
    padding: 10,
    cornerRadius: 8,
  };
}

const GRID = { color: "#F3F4F6", drawBorder: false };
const AXIS_NONE = { display: false };

export function destroyChart(key) {
  if (registry[key]) {
    registry[key].destroy();
    registry[key] = null;
  }
}

export function chartEmpty(parent, msg) {
  if (!parent) return;
  parent.querySelector(".chart-empty")?.remove();
  const note = document.createElement("div");
  note.className = "chart-empty";
  note.textContent = msg;
  parent.appendChild(note);
}

export function clearChartEmpty(parent) {
  parent?.querySelector(".chart-empty")?.remove();
}

export function doughnutChart(canvasId, entries, opts = {}) {
  const canvas = $(`#${canvasId}`);
  if (!canvas) return;
  const parent = canvas.parentElement;
  clearChartEmpty(parent);
  const filtered = entries.filter(([, v]) => v > 0);
  if (!filtered.length) {
    chartEmpty(parent, opts.emptyMsg || "No data yet");
    return;
  }
  const key = canvasId;
  destroyChart(key);
  const labels = filtered.map(([k]) => (opts.labelFn ? opts.labelFn(k) : vinfo(k).label));
  const values = filtered.map(([, v]) => v);
  const colors = filtered.map(([k]) => (opts.colorFn ? opts.colorFn(k) : vinfo(k).color));
  const total = values.reduce((a, b) => a + b, 0);
  registry[key] = new Chart(canvas, {
    type: "doughnut",
    data: {
      labels,
      datasets: [{
        data: values,
        backgroundColor: colors,
        borderColor: "#fff",
        borderWidth: 2,
        hoverOffset: 4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      cutout: opts.cutout || "65%",
      plugins: {
        legend: opts.legend ? {
          display: true,
          position: opts.legendPosition || "right",
          labels: { boxWidth: 8, padding: 10, font: { size: 11 } },
        } : { display: false },
        tooltip: {
          callbacks: {
            label: (c) => ` ${c.label}: ${c.parsed} (${Math.round((c.parsed / total) * 100)}%)`,
          },
        },
      },
    },
  });
  return { total, values, labels, colors };
}

export function lineChart(canvasId, labels, values, color = "#0D6EFD") {
  const canvas = $(`#${canvasId}`);
  if (!canvas) return;
  destroyChart(canvasId);
  registry[canvasId] = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: color,
        backgroundColor: color + "18",
        fill: true,
        tension: 0.35,
        pointRadius: 3,
        pointBackgroundColor: color,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: AXIS_NONE, ticks: { maxRotation: 0, autoSkip: true, maxTicksLimit: 8 } },
        y: { grid: GRID, beginAtZero: true, ticks: { precision: 0 } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

export function barChart(canvasId, labels, values, horizontal = false, colors) {
  const canvas = $(`#${canvasId}`);
  if (!canvas) return;
  destroyChart(canvasId);
  const ramp = colors || ["#0D6EFD", "#3B82F6", "#60A5FA", "#93C5FD"];
  const bg = values.map((_, i) => ramp[Math.min(i, ramp.length - 1)]);
  registry[canvasId] = new Chart(canvas, {
    type: "bar",
    data: { labels, datasets: [{ data: values, backgroundColor: bg, borderRadius: 4, borderSkipped: false }] },
    options: {
      indexAxis: horizontal ? "y" : "x",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { grid: horizontal ? GRID : AXIS_NONE, beginAtZero: true },
        y: { grid: horizontal ? AXIS_NONE : GRID, beginAtZero: true },
      },
    },
  });
}

export function hourChart(canvasId, byHour) {
  const values = Array.from({ length: 24 }, (_, h) => Number(byHour?.[h] || byHour?.[String(h)] || 0));
  if (!values.some((v) => v > 0)) {
    chartEmpty($(`#${canvasId}`)?.parentElement, "No hourly data");
    return;
  }
  const labels = values.map((_, h) => `${String(h).padStart(2, "0")}:00`);
  barChart(canvasId, labels, values, false, ["#0D6EFD55", "#0D6EFD88", "#0D6EFD"]);
}

export function vehicleChart(canvasId, byVehicle) {
  const entries = Object.entries(byVehicle || {}).filter(([, v]) => v > 0).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    chartEmpty($(`#${canvasId}`)?.parentElement, "No vehicle data");
    return;
  }
  barChart(canvasId, entries.map(([k]) => prettify(k)), entries.map(([, v]) => v), true);
}

export function typeDoughnut(canvasId, byType) {
  const entries = VIOLATION_ORDER.map((k) => [k, byType?.[k] || 0]);
  return doughnutChart(canvasId, entries, { legend: true, legendPosition: "right" });
}

export function severityDoughnut(canvasId, severity) {
  const entries = [
    ["critical", severity.critical?.count || 0],
    ["high", severity.high?.count || 0],
    ["medium", severity.medium?.count || 0],
    ["low", severity.low?.count || 0],
  ];
  const colors = { critical: "#EF4444", high: "#F59E0B", medium: "#0D6EFD", low: "#10B981" };
  return doughnutChart(canvasId, entries, {
    cutout: "60%",
    labelFn: (k) => prettify(k),
    colorFn: (k) => colors[k],
    legend: true,
    legendPosition: "bottom",
  });
}

export function sparkline(canvas, data, color = "#0D6EFD") {
  if (!canvas || !data?.length || !window.Chart) return;
  const id = canvas.id || `spark-${Math.random()}`;
  destroyChart(id);
  registry[id] = new Chart(canvas, {
    type: "line",
    data: {
      labels: data.map((_, i) => i),
      datasets: [{
        data,
        borderColor: color,
        borderWidth: 1.5,
        pointRadius: 0,
        fill: false,
        tension: 0.4,
      }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false }, tooltip: { enabled: false } },
      scales: { x: { display: false }, y: { display: false } },
    },
  });
}

export function renderSparklines(root = document) {
  configureChartDefaults();
  root.querySelectorAll(".kpi-sparkline[data-spark]").forEach((canvas) => {
    try {
      const data = JSON.parse(canvas.dataset.spark);
      const card = canvas.closest(".kpi-card");
      const icon = card?.querySelector(".kpi-icon");
      const color = icon ? getComputedStyle(icon).color : "#0D6EFD";
      sparkline(canvas, data, color);
    } catch (_) {}
  });
}

export function renderDistributionLegend(containerId, distribution) {
  const el = $(`#${containerId}`);
  if (!el) return;
  el.innerHTML = distribution.map((d) => `
    <div class="dist-legend-row">
      <span class="dist-col-type">
        <span class="dist-dot" style="background:${d.color}"></span>
        <span class="dist-label">${esc(d.label)}</span>
      </span>
      <span class="dist-col-count">${d.count}</span>
      <span class="dist-col-pct">${d.pct}%</span>
    </div>`).join("");
}
