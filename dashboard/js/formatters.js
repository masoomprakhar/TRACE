/** Violation taxonomy — single source of truth for labels and colors. */
export const VIOLATION = {
  no_helmet:       { label: "No Helmet",           color: "#F59E0B" },
  no_seatbelt:     { label: "No Seatbelt",         color: "#0D6EFD" },
  triple_riding:   { label: "Triple Riding",       color: "#EF4444" },
  wrong_side:      { label: "Wrong-Side Driving",  color: "#8B5CF6" },
  stop_line:       { label: "Stop-Line Violation", color: "#06B6D4" },
  red_light:       { label: "Red-Light Violation", color: "#DC2626" },
  illegal_parking: { label: "Illegal Parking",     color: "#3B82F6" },
};
export const VIOLATION_ORDER = Object.keys(VIOLATION);
export const REVIEW_THRESHOLD = 0.35;

const SEVERITY_MAP = {
  red_light: "critical",
  triple_riding: "critical",
  wrong_side: "high",
  stop_line: "high",
  no_helmet: "medium",
  no_seatbelt: "medium",
  illegal_parking: "low",
};

export function vinfo(type) {
  return VIOLATION[type] || { label: prettify(type), color: "#6B7280" };
}

export function deriveSeverity(types, confidence = 0.5) {
  if (!types?.length) return "low";
  const ranks = { critical: 4, high: 3, medium: 2, low: 1 };
  let max = "low";
  for (const t of types) {
    const s = SEVERITY_MAP[t] || "medium";
    if (ranks[s] > ranks[max]) max = s;
  }
  if (confidence >= 0.9 && max === "medium") return "high";
  return max;
}

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
export const el = (tag, cls, html) => {
  const n = document.createElement(tag);
  if (cls) n.className = cls;
  if (html != null) n.innerHTML = html;
  return n;
};

export function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

export function prettify(s) {
  return String(s ?? "").replace(/_/g, " ").replace(/\b\w/g, (c) => c.toUpperCase());
}

export function fmtTime(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return esc(iso);
  return d.toLocaleString(undefined, {
    month: "short", day: "2-digit", hour: "2-digit", minute: "2-digit",
  });
}

export function fmtDate(iso) {
  if (!iso) return "—";
  const d = new Date(iso);
  if (isNaN(d)) return esc(iso);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" });
}

export function pct(x) {
  if (x == null || isNaN(x)) return "—";
  const v = x <= 1 ? x * 100 : x;
  return Math.round(v) + "%";
}

export function confColor(x) {
  const v = x == null ? 0 : (x <= 1 ? x : x / 100);
  if (v < REVIEW_THRESHOLD) return "#9CA3AF";
  if (v < 0.6) return "#F59E0B";
  if (v < 0.8) return "#EAB308";
  return "#10B981";
}

export function num(n) {
  if (n == null || isNaN(n)) return "—";
  return Number(n).toLocaleString();
}
