/** Hash router — avoids circular imports between main and pages. */
const PAGE_META = {
  overview: { title: "Overview", subtitle: "Real-time traffic monitoring & violation analytics" },
  live: { title: "Live Monitor", subtitle: "CCTV streams for production · device camera for instant demos" },
  violations: { title: "Violations", subtitle: "Review queue with filters, evidence, and export" },
  anpr: { title: "ANPR Search", subtitle: "License plate lookup and incident history" },
  evidence: { title: "Evidence Center", subtitle: "Visual gallery and frame analysis upload" },
  reports: { title: "Reports", subtitle: "Analytics summaries and data exports" },
  hotspots: { title: "Hotspots", subtitle: "Geographic violation concentration and peak hours" },
  offenders: { title: "Repeat Offenders", subtitle: "Top plates by incident count and risk score" },
  alerts: { title: "Alerts", subtitle: "System notifications and anomaly detection" },
  settings: { title: "Settings", subtitle: "System health, models, and configuration" },
};

const VALID = Object.keys(PAGE_META);
let _onSwitch = null;

export function setSwitchHandler(fn) {
  _onSwitch = fn;
}

export function navigate(view, params = {}) {
  let hash = `#${view}`;
  if (params.q) hash += `?q=${encodeURIComponent(params.q)}`;
  if (location.hash !== hash) location.hash = hash;
  else _onSwitch?.(view, params);
}

export function parseHash() {
  const raw = (location.hash || "#overview").slice(1);
  const [view, query] = raw.split("?");
  const params = {};
  if (query) new URLSearchParams(query).forEach((v, k) => { params[k] = v; });
  return { view: VALID.includes(view) ? view : "overview", params };
}

export function updatePageHeader(view) {
  const meta = PAGE_META[view] || PAGE_META.overview;
  const title = document.getElementById("page-title");
  const subtitle = document.getElementById("page-subtitle");
  if (title) title.textContent = meta.title;
  if (subtitle) subtitle.textContent = meta.subtitle;
}

export { PAGE_META, VALID };
