import { $, esc, fmtTime, pct, confColor, prettify, vinfo, REVIEW_THRESHOLD, deriveSeverity } from "./formatters.js";

const ICONS = {
  error: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg>',
  success: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M20 6L9 17l-5-5"/></svg>',
  info: '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="9"/><path d="M12 16v-4M12 8h.01"/></svg>',
};

export function toast(message, kind = "error", ttl = 5200) {
  const stack = $("#toast-stack");
  if (!stack) return;
  const t = document.createElement("div");
  t.className = `toast toast-${kind}`;
  t.innerHTML = `<span class="toast-ico">${ICONS[kind] || ICONS.info}</span>
    <div class="toast-body">${esc(message)}</div>`;
  stack.appendChild(t);
  const remove = () => { t.classList.add("leaving"); setTimeout(() => t.remove(), 240); };
  t.addEventListener("click", remove);
  if (ttl) setTimeout(remove, ttl);
}

export function badge(type) {
  const { label, color } = vinfo(type);
  return `<span class="vbadge" style="--vbadge-color:${color}">${esc(label)}</span>`;
}

export function badges(types) {
  if (!types?.length) return `<span class="text-muted">—</span>`;
  return `<div class="badge-row">${types.map(badge).join("")}</div>`;
}

export function confBar(conf) {
  const v = conf == null ? 0 : (conf <= 1 ? conf : conf / 100);
  const w = Math.max(0, Math.min(100, v * 100));
  const c = confColor(v);
  const tick = Math.round(REVIEW_THRESHOLD * 100);
  return `<div class="confbar-wrap">
    <div class="confbar" title="Review threshold ${tick}%">
      <div class="confbar-fill" data-w="${w}%" style="width:0;background:${c}"></div>
      <span class="confbar-tick" style="left:${tick}%"></span>
    </div>
    <span class="confbar-pct" style="color:${c}">${pct(v)}</span>
  </div>`;
}

export function plateChip(text, big = false) {
  const cls = big ? "plate plate-lg" : "plate";
  if (!text) return `<span class="${cls} plate-empty">no read</span>`;
  return `<span class="${cls}">${esc(text)}</span>`;
}

const VEHICLE_ICON = {
  motorcycle: "🏍", car: "🚗", truck: "🚛", bus: "🚌", bicycle: "🚲",
};

export function vehicleCell(v) {
  const key = String(v || "").toLowerCase();
  const ico = VEHICLE_ICON[key] || "•";
  return `<span class="vehicle-cell"><span class="vehicle-ico">${ico}</span>${v ? esc(prettify(v)) : "—"}</span>`;
}

export function severityBadge(types, confidence) {
  const sev = deriveSeverity(types, confidence);
  const colors = { critical: "#EF4444", high: "#F59E0B", medium: "#0D6EFD", low: "#10B981" };
  return `<span class="sev-badge" style="--sev-color:${colors[sev]}">${prettify(sev)}</span>`;
}

const KPI_META_ICONS = {
  target: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/><path d="M12 2v3M12 19v3M2 12h3M19 12h3"/></svg>',
  clock: '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>',
};

function kpiTrendMeta(trend, status) {
  if (trend) {
    const dir = trend.up ? "Up" : "Down";
    return `${dir} ${trend.delta}% this period`;
  }
  if (status) return status;
  return "Last 24 hours";
}

export function kpiCard({
  label,
  value,
  trend,
  sparkData,
  icon,
  color = "#0D6EFD",
  status,
  meta1 = "TRACE Enforcement",
  meta2,
  footerLeftLabel = "Trend",
  footerRightLabel = "Total",
}) {
  const sparkId = `spark-${label.replace(/\s/g, "-").toLowerCase()}`;
  const hasSpark = Array.isArray(sparkData) && sparkData.length > 1;
  const meta2Text = meta2 ?? kpiTrendMeta(trend, status);
  const showProgress = Boolean(trend);
  const footerRightCaption = showProgress ? "Progress" : footerRightLabel;
  const footerRightValue = showProgress ? `${trend.delta}%` : String(value);

  const sparkHtml = hasSpark
    ? `<canvas class="kpi-sparkline" id="${sparkId}"
        data-spark='${JSON.stringify(sparkData)}'
        data-color="${esc(color)}"></canvas>`
    : `<div class="kpi-trend-dots" aria-hidden="true">
        <span style="background:${color}"></span>
        <span style="background:${color}99"></span>
        <span style="background:${color}55"></span>
        <span class="kpi-trend-dots-plus">+</span>
      </div>`;

  return `<article class="kpi-card" style="--kpi-accent:${esc(color)}">
    <div class="kpi-float-icon" style="background:${esc(color)};box-shadow:0 10px 28px ${esc(color)}55">
      ${icon}
    </div>
    <div class="kpi-card-body">
      <h3 class="kpi-title">${esc(label)}</h3>
      ${showProgress ? `<div class="kpi-stat">${esc(String(value))}</div>` : ""}
      <div class="kpi-meta">
        <div class="kpi-meta-row">${KPI_META_ICONS.target}<span>${esc(meta1)}</span></div>
        <div class="kpi-meta-row">${KPI_META_ICONS.clock}<span>${esc(meta2Text)}</span></div>
      </div>
      <div class="kpi-divider" aria-hidden="true"></div>
      <div class="kpi-footer">
        <div class="kpi-footer-col">
          <span class="kpi-footer-label">${esc(footerLeftLabel)}</span>
          <div class="kpi-trend-viz">${sparkHtml}</div>
        </div>
        <div class="kpi-footer-col kpi-footer-col-right">
          <span class="kpi-footer-label">${esc(footerRightCaption)}</span>
          <span class="kpi-footer-value">${esc(footerRightValue)}</span>
        </div>
      </div>
    </div>
  </article>`;
}

export function emptyState(msg, sub = "") {
  return `<div class="empty-state">
    <div class="empty-icon"><svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="9"/><path d="M12 8v4M12 16h.01"/></svg></div>
    <p class="empty-title">${esc(msg)}</p>
    ${sub ? `<p class="empty-sub">${sub}</p>` : ""}
  </div>`;
}

export function skeleton(n = 3) {
  return Array.from({ length: n }, () =>
    `<div class="skeleton-card"><div class="skel skel-line w-1/3"></div><div class="skel skel-line w-2/3"></div></div>`
  ).join("");
}

export function violationCard(v) {
  const { color } = vinfo(v.type);
  const plateText = v.plate?.text || null;
  const w = Math.max(0, Math.min(100, (v.confidence || 0) * 100));
  const c = confColor(v.confidence);
  const valid = v.plate?.valid_format;
  return `<div class="violation-card" style="border-left-color:${color}">
    <div class="violation-card-head">
      ${badge(v.type)}
      <span class="confbar-pct" style="color:${c}">${pct(v.confidence)}</span>
    </div>
    ${confBar(v.confidence)}
    <div class="violation-card-foot">
      ${vehicleCell(v.vehicle_class)}
      <div class="flex gap-2 items-center">
        ${plateChip(plateText)}
        ${plateText ? `<span class="tag-sm ${valid ? "valid" : ""}">${valid ? "valid" : "unverified"}</span>` : ""}
      </div>
    </div>
  </div>`;
}

export function violationRow(it, onClick) {
  const conf = it.confidence;
  const w = Math.max(0, Math.min(100, (conf || 0) * 100));
  const c = confColor(conf);
  const tick = Math.round(REVIEW_THRESHOLD * 100);
  const thumb = it.evidence_url
    ? `<img class="thumb" src="${esc(it.evidence_url)}" alt="" loading="lazy" />`
    : `<div class="thumb-fallback">📷</div>`;
  return `<tr class="clickable-row" data-id="${esc(it.id)}">
    <td><div class="thumb-wrap">${thumb}</div></td>
    <td class="mono text-sm text-muted">${fmtTime(it.timestamp)}</td>
    <td>${badges(it.violation_types)}</td>
    <td>${severityBadge(it.violation_types, conf)}</td>
    <td>${vehicleCell(it.vehicle_type)}</td>
    <td>${plateChip(it.plate_number)}</td>
    <td><span class="text-sm text-muted">${esc(it.location || "—")}</span></td>
    <td>
      <div class="confbar" title="Review threshold ${tick}%">
        <div class="confbar-fill" data-w="${w}%" style="width:0;background:${c}"></div>
      </div>
      <span class="confbar-pct text-xs" style="color:${c}">${pct(conf)}</span>
    </td>
  </tr>`;
}

export function recentViolationItem(it) {
  const type = it.violation_types?.[0] || "unknown";
  const { label, color } = vinfo(type);
  const thumb = it.evidence_url
    ? `<img src="${esc(it.evidence_url)}" alt="" class="recent-thumb" />`
    : `<div class="recent-thumb recent-thumb-empty"></div>`;
  return `<div class="recent-item" data-id="${esc(it.id)}">
    ${thumb}
    <div class="recent-body">
      <div class="recent-type" style="color:${color}">${esc(label)}</div>
      <div class="recent-meta">${fmtTime(it.timestamp)}</div>
      <div class="recent-foot">
        ${plateChip(it.plate_number)}
        <span class="recent-conf">${pct(it.confidence)}</span>
      </div>
    </div>
  </div>`;
}

export function plateResultCard(p) {
  return `<div class="plate-result-card">
    <div class="plate-result-head">
      ${plateChip(p.plate, true)}
      <div class="plate-count">${esc(p.count ?? 0)}<span class="plate-count-label">incidents</span></div>
    </div>
    <div class="meta-row"><span>Last seen</span><span>${fmtTime(p.last_seen)}</span></div>
    <div class="mt-2">${badges(p.violations || [])}</div>
  </div>`;
}

export function animateConfBars(root = document) {
  requestAnimationFrame(() =>
    Array.from(root.querySelectorAll(".confbar-fill[data-w]")).forEach((f) => {
      f.style.width = f.dataset.w;
    })
  );
}

export function imageFallback() {
  const d = document.createElement("div");
  d.className = "evidence-fallback";
  d.innerHTML = emptyState("Evidence image unavailable");
  return d;
}
