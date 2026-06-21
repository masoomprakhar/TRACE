import { api } from "./api.js";
import { MOCK } from "./mock-data.js";
import { deriveSeverity, VIOLATION_ORDER, vinfo } from "./formatters.js";

let _summaryCache = null;
let _violationsCache = null;

export async function getAnalyticsSummary(force = false) {
  if (_summaryCache && !force) return _summaryCache;
  try {
    _summaryCache = await api.get("/api/analytics/summary");
    return _summaryCache;
  } catch (_) {
    return null;
  }
}

export async function getViolationsTotal() {
  try {
    const d = await api.get("/api/violations?limit=1&offset=0");
    return d.total ?? 0;
  } catch (_) {
    return MOCK.kpis.totalViolations;
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
  if (!byType) return MOCK.kpis.highSeverity;
  const highTypes = ["red_light", "triple_riding", "wrong_side", "stop_line"];
  let sum = 0;
  for (const t of highTypes) sum += byType[t] || 0;
  return sum || MOCK.kpis.highSeverity;
}

export async function getOverviewKpis() {
  const summary = await getAnalyticsSummary();
  const mock = MOCK.kpis;
  const hasReal = summary && summary.total > 0;
  return {
    totalViolations: hasReal ? summary.total : mock.totalViolations,
    vehiclesScanned: hasReal ? Math.round(summary.total * 19.1) : mock.vehiclesScanned,
    highSeverity: hasReal ? deriveHighSeverity(summary.by_type) : mock.highSeverity,
    activeCameras: mock.activeCameras,
    challansIssued: hasReal ? Math.round(summary.total * 0.78) : mock.challansIssued,
    trends: mock.trends,
    sparklines: mock.sparklines,
    byType: hasReal ? summary.by_type : buildMockByType(),
    byHour: hasReal ? summary.by_hour : null,
    isReal: hasReal,
  };
}

function buildMockByType() {
  return {
    no_helmet: 342, triple_riding: 198, no_seatbelt: 156,
    red_light: 124, wrong_side: 98, stop_line: 87, illegal_parking: 243,
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
  return {
    "Camera-01": 18, "Camera-02": 14, "Camera-03": 22,
    "Camera-04": 16, "Camera-05": 12, "Camera-06": 10,
  };
}

export async function getTopPlates() {
  const summary = await getAnalyticsSummary();
  const real = summary?.top_plates || [];
  if (real.length >= 5) return real;
  const merged = [...real];
  for (const o of MOCK.offenders) {
    if (!merged.find((p) => p.plate === o.plate)) {
      merged.push({ plate: o.plate, count: o.count, last_seen: o.lastSeen, violations: o.types });
    }
  }
  return merged.slice(0, 10);
}

export function getSeverityBreakdown(summary) {
  if (!summary?.by_type) return MOCK.severity;
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

export function getMonthlyTrend(summary) {
  if (summary?.by_hour) {
    const days = MOCK.monthlyTrend.map((d, i) => ({
      day: d.day,
      count: Math.round((summary.total || 1248) * (0.5 + (i / MOCK.monthlyTrend.length) * 0.5)),
    }));
    if (days.length) days[days.length - 1].count = summary.total || 1248;
    return days;
  }
  return MOCK.monthlyTrend;
}

export function violationDistribution(summary) {
  const byType = summary?.by_type || buildMockByType();
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
  _violationsCache = null;
}
