import { api } from "./api.js";
import { MOCK } from "./mock-data.js";
import { deriveSeverity, VIOLATION_ORDER, vinfo } from "./formatters.js";

let _summaryCache = null;

export async function getAnalyticsSummary(force = false) {
  if (_summaryCache && !force) return _summaryCache;
  try {
    _summaryCache = await api.get("/api/analytics/summary");
    return _summaryCache;
  } catch (_) {
    return null;
  }
}

export async function getEvalSummary() {
  try {
    return await api.get("/api/eval/summary");
  } catch (_) {
    return null;
  }
}

export async function getViolationsTotal() {
  try {
    const d = await api.get("/api/violations?limit=1&offset=0");
    return d.total ?? 0;
  } catch (_) {
    return 0;
  }
}

export async function getRecentViolations(limit = 5) {
  try {
    const d = await api.get(`/api/violations?limit=${limit}&offset=0`);
    if (d.items?.length) return d.items;
  } catch (_) {}
  return [];
}

export function deriveHighSeverity(byType) {
  if (!byType) return 0;
  const highTypes = ["red_light", "triple_riding", "wrong_side", "stop_line"];
  let sum = 0;
  for (const t of highTypes) sum += byType[t] || 0;
  return sum;
}

function hourlyValues(byHour) {
  if (!byHour) return [];
  return Object.entries(byHour)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([, v]) => Number(v) || 0);
}

function cumulativeSpark(vals) {
  if (!vals.length) return null;
  let run = 0;
  const out = vals.map((v) => {
    run += v;
    return run;
  });
  return out.length > 1 ? out : null;
}

function periodTrend(vals) {
  if (!vals || vals.length < 2) return null;
  const half = Math.max(1, Math.floor(vals.length / 2));
  const first = vals.slice(0, half).reduce((a, b) => a + b, 0);
  const second = vals.slice(half).reduce((a, b) => a + b, 0);
  if (first === 0 && second === 0) return null;
  if (first === 0) return { delta: 100, up: true };
  const pct = Math.round(((second - first) / first) * 1000) / 10;
  return { delta: Math.abs(pct), up: second >= first };
}

export async function getOverviewKpis() {
  const summary = await getAnalyticsSummary();
  const total = summary?.total ?? 0;
  const hasReal = total > 0;
  const byLocation = hasReal ? await getViolationsByLocation() : {};
  const cameraCount = Object.keys(byLocation).length;
  const hourVals = hourlyValues(summary?.by_hour);
  const trend = periodTrend(hourVals);
  const cumSpark = cumulativeSpark(hourVals);
  const challans = summary?.challans_issued ?? 0;
  const vehicles = summary?.vehicles_scanned ?? total;

  const sparklines = cumSpark
    ? {
        totalViolations: cumSpark,
        vehiclesScanned: cumSpark.map((v, i) => Math.round(v * (vehicles / Math.max(total, 1)))),
        highSeverity: cumSpark.map((v, i) => Math.round(v * (deriveHighSeverity(summary?.by_type) / Math.max(total, 1)))),
        challansIssued: cumSpark.map((v, i) => Math.round(v * (challans / Math.max(total, 1)))),
      }
    : {};

  return {
    totalViolations: total,
    vehiclesScanned: vehicles,
    highSeverity: hasReal ? deriveHighSeverity(summary.by_type) : 0,
    activeCameras: cameraCount,
    challansIssued: challans,
    avgConfidence: hasReal ? summary.avg_confidence : null,
    processingFps: hasReal ? summary.processing_fps : null,
    byType: hasReal ? summary.by_type : {},
    byHour: hasReal ? summary.by_hour : null,
    trends: trend
      ? {
          totalViolations: trend,
          vehiclesScanned: trend,
          highSeverity: trend,
          challansIssued: trend,
        }
      : {},
    sparklines,
    isReal: hasReal,
  };
}

export async function getViolationsByLocation() {
  try {
    const d = await api.get("/api/violations?limit=200&offset=0");
    const map = {};
    for (const it of d.items || []) {
      const loc = it.location || "Unknown";
      map[loc] = (map[loc] || 0) + 1;
    }
    if (Object.keys(map).length) return map;
  } catch (_) {}
  return {};
}

export async function getTopPlates() {
  const summary = await getAnalyticsSummary();
  return summary?.top_plates || [];
}

export function getSeverityBreakdown(summary) {
  if (!summary?.by_type) {
    return { critical: { count: 0, pct: 0 }, high: { count: 0, pct: 0 }, medium: { count: 0, pct: 0 }, low: { count: 0, pct: 0 } };
  }
  const buckets = { critical: 0, high: 0, medium: 0, low: 0 };
  for (const [type, count] of Object.entries(summary.by_type)) {
    const sev = deriveSeverity([type]);
    buckets[sev] += count;
  }
  const total = Object.values(buckets).reduce((a, b) => a + b, 0) || 1;
  return {
    critical: { count: buckets.critical, pct: Math.round((buckets.critical / total) * 100) },
    high: { count: buckets.high, pct: Math.round((buckets.high / total) * 100) },
    medium: { count: buckets.medium, pct: Math.round((buckets.medium / total) * 100) },
    low: { count: buckets.low, pct: Math.round((buckets.low / total) * 100) },
  };
}

export function getHourlyTrend(summary) {
  if (!summary?.by_hour) return [];
  return Object.entries(summary.by_hour)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([hour, count]) => ({ day: `${hour}:00`, count: Number(count) || 0 }));
}

export function violationDistribution(summary) {
  const byType = summary?.by_type || {};
  const entries = VIOLATION_ORDER
    .map((k) => [k, byType[k] || 0])
    .filter(([, v]) => v > 0);
  const total = entries.reduce((s, [, v]) => s + v, 0) || 1;
  return entries.map(([k, v]) => ({
    type: k, label: vinfo(k).label, color: vinfo(k).color, count: v,
    pct: Math.round((v / total) * 100),
  }));
}

export function invalidateCache() {
  _summaryCache = null;
}
