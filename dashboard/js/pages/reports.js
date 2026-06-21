import { api } from "../api.js";
import { $, num, pct } from "../formatters.js";
import { getAnalyticsSummary } from "../data-bridge.js";
import { toast, emptyState, kpiCard } from "../components.js";
import { configureChartDefaults, typeDoughnut, hourChart, vehicleChart } from "../charts.js";

export function initReports() {
  $("#reports-csv")?.addEventListener("click", () => window.open("/api/violations.csv", "_blank"));
  $("#reports-copy")?.addEventListener("click", copyReport);
  $("#reports-print")?.addEventListener("click", () => window.print());
  $("#reports-refresh")?.addEventListener("click", loadReports);
}

async function copyReport() {
  try {
    const d = await api.get("/api/report/summary");
    await navigator.clipboard.writeText(d.report || "");
    toast("Report copied to clipboard.", "success");
  } catch (err) {
    toast(`Copy failed: ${err.message}`, "error");
  }
}

export async function loadReports() {
  configureChartDefaults();
  const kpiGrid = $("#reports-kpis");
  const preview = $("#reports-preview");
  if (kpiGrid) kpiGrid.innerHTML = '<div class="skeleton-card"></div>'.repeat(4);

  try {
    const [summary, reportData] = await Promise.all([
      getAnalyticsSummary(true),
      api.get("/api/report/summary").catch(() => null),
    ]);
    const s = summary || { total: 0, avg_confidence: 0, processing_fps: 0, top_plates: [] };
    if (kpiGrid) {
      kpiGrid.innerHTML = [
        kpiCard({ label: "Total Violations", value: num(s.total), icon: "📊", color: "#0D6EFD" }),
        kpiCard({ label: "Avg Confidence", value: pct(s.avg_confidence), icon: "🎯", color: "#10B981" }),
        kpiCard({ label: "Processing FPS", value: s.processing_fps?.toFixed(1) || "—", icon: "⚡", color: "#F59E0B" }),
        kpiCard({ label: "Distinct Plates", value: num((s.top_plates || []).length), icon: "🔢", color: "#8B5CF6" }),
      ].join("");
    }
    typeDoughnut("chart-reports-type", s.by_type || {});
    hourChart("chart-reports-hour", s.by_hour || {});
    vehicleChart("chart-reports-vehicle", s.by_vehicle || {});
    if (preview) {
      preview.textContent = reportData?.report || "No report data available. Seed demo data to generate a report.";
    }
  } catch (err) {
    if (kpiGrid) kpiGrid.innerHTML = emptyState("Couldn't load reports", err.message);
  }
}

export function initReportsPage() {
  initReports();
  loadReports();
}
