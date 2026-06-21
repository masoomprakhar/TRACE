import { api } from "../api.js";
import { $, VIOLATION_ORDER, vinfo, deriveSeverity } from "../formatters.js";
import { toast, violationRow, emptyState, animateConfBars } from "../components.js";
import { updateBadges } from "../nav-badges.js";

const vstate = { limit: 20, offset: 0, total: 0, type: "", plate: "", severity: "" };

export function initViolations() {
  const sel = $("#filter-type");
  if (sel && sel.options.length <= 1) {
    VIOLATION_ORDER.forEach((t) => {
      const o = document.createElement("option");
      o.value = t; o.textContent = vinfo(t).label;
      sel.appendChild(o);
    });
  }
  $("#filter-refresh")?.addEventListener("click", () => { vstate.offset = 0; loadViolations(); });
  sel?.addEventListener("change", () => { vstate.type = sel.value; vstate.offset = 0; loadViolations(); });
  $("#filter-severity")?.addEventListener("change", (e) => { vstate.severity = e.target.value; vstate.offset = 0; loadViolations(); });
  let debounce;
  $("#filter-plate")?.addEventListener("input", (e) => {
    clearTimeout(debounce);
    debounce = setTimeout(() => { vstate.plate = e.target.value.trim(); vstate.offset = 0; loadViolations(); }, 350);
  });
  $("#page-prev")?.addEventListener("click", () => {
    if (vstate.offset <= 0) return;
    vstate.offset = Math.max(0, vstate.offset - vstate.limit);
    loadViolations();
  });
  $("#page-next")?.addEventListener("click", () => {
    if (vstate.offset + vstate.limit >= vstate.total) return;
    vstate.offset += vstate.limit;
    loadViolations();
  });
  $("#export-csv")?.addEventListener("click", exportCsv);
}

function violationsQuery() {
  const p = new URLSearchParams({ limit: vstate.limit, offset: vstate.offset });
  if (vstate.type) p.set("type", vstate.type);
  if (vstate.plate) p.set("plate", vstate.plate);
  return `/api/violations?${p.toString()}`;
}

function exportCsv() {
  const p = new URLSearchParams();
  if (vstate.type) p.set("type", vstate.type);
  if (vstate.plate) p.set("plate", vstate.plate);
  const qs = p.toString();
  window.open(`/api/violations.csv${qs ? "?" + qs : ""}`, "_blank");
}

export async function loadViolations() {
  const tbody = $("#violations-tbody");
  if (!tbody) return;
  tbody.innerHTML = `<tr><td colspan="8"><div class="skeleton-card"><div class="skel skel-line"></div></div></td></tr>`;
  try {
    const data = await api.get(violationsQuery());
    vstate.total = data.total || 0;
    updateBadges();
    let items = data.items || [];
    if (vstate.severity) {
      items = items.filter((it) =>
        deriveSeverity(it.violation_types, it.confidence) === vstate.severity
      );
    }
    if (!items.length) {
      tbody.innerHTML = `<tr><td colspan="8">${emptyState(vstate.type || vstate.plate ? "No matching violations" : "No violations recorded yet")}</td></tr>`;
    } else {
      tbody.innerHTML = items.map((it) => violationRow(it)).join("");
      tbody.querySelectorAll(".clickable-row").forEach((row) => {
        row.addEventListener("click", () => window.openEvidence?.(row.dataset.id));
      });
      animateConfBars(tbody);
    }
    updatePager();
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="8">${emptyState("Couldn't load violations", err.message)}</td></tr>`;
    updatePager();
  }
}

function updatePager() {
  const { offset, limit, total } = vstate;
  const start = total === 0 ? 0 : offset + 1;
  const end = Math.min(offset + limit, total);
  $("#violations-range").textContent = total === 0 ? "0 records" : `${start}–${end} of ${total}`;
  const prev = $("#page-prev");
  const next = $("#page-next");
  if (prev) prev.disabled = offset <= 0;
  if (next) next.disabled = offset + limit >= total;
}

export function initViolationsPage() {
  initViolations();
  loadViolations();
}
