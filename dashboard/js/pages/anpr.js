import { api } from "../api.js";
import { $, esc, fmtTime, pct } from "../formatters.js";
import { MOCK } from "../mock-data.js";
import { toast, plateResultCard, emptyState, badges, plateChip } from "../components.js";
import { parseHash } from "../router.js";

const RECENT_KEY = "trace_recent_plates";

export function getRecentSearches() {
  try {
    const stored = JSON.parse(localStorage.getItem(RECENT_KEY) || "[]");
    return [...new Set([...stored, ...MOCK.recentSearches])].slice(0, 8);
  } catch (_) {
    return MOCK.recentSearches;
  }
}

function saveRecent(q) {
  const list = getRecentSearches().filter((p) => p !== q);
  list.unshift(q);
  localStorage.setItem(RECENT_KEY, JSON.stringify(list.slice(0, 8)));
}

export function initAnpr() {
  $("#anpr-search-btn")?.addEventListener("click", () => searchPlates($("#anpr-query")?.value?.trim()));
  $("#anpr-query")?.addEventListener("keydown", (e) => {
    if (e.key === "Enter") searchPlates($("#anpr-query")?.value?.trim());
  });
  renderRecentTags();
}

function renderRecentTags() {
  const wrap = $("#anpr-recent-tags");
  if (!wrap) return;
  wrap.innerHTML = getRecentSearches().map((p) =>
    `<button class="tag-chip" data-plate="${esc(p)}">${esc(p)}</button>`
  ).join("");
  wrap.querySelectorAll(".tag-chip").forEach((btn) => {
    btn.addEventListener("click", () => {
      const q = btn.dataset.plate;
      $("#anpr-query").value = q;
      searchPlates(q);
    });
  });
}

export async function searchPlates(q) {
  const wrap = $("#anpr-results");
  if (!wrap) return;
  if (!q) {
    wrap.innerHTML = emptyState("Type a plate to search", 'Even a partial like <span class="mono">MH</span> works.');
    return;
  }
  saveRecent(q);
  renderRecentTags();
  wrap.innerHTML = '<div class="skeleton-card"></div>'.repeat(3);
  try {
    const d = await api.get(`/api/plates/search?q=${encodeURIComponent(q)}`);
    const items = d.items || [];
    if (!items.length) {
      wrap.innerHTML = emptyState(`No plates match "${q}"`, "Try a shorter or different fragment.");
      return;
    }
    wrap.innerHTML = `<div class="anpr-results-grid">${items.map(plateResultCard).join("")}</div>`;
    renderTimeline(items[0]);
  } catch (err) {
    wrap.innerHTML = emptyState("Search failed", err.message);
  }
}

function renderTimeline(item) {
  const tl = $("#anpr-timeline");
  if (!tl || !item) return;
  const entries = [
    { time: item.last_seen, event: "Last detected", types: item.violations },
    { time: item.last_seen, event: `${item.count} total incident${item.count === 1 ? "" : "s"}`, types: [] },
  ];
  tl.innerHTML = entries.map((e) => `
    <div class="timeline-item">
      <div class="timeline-dot"></div>
      <div class="timeline-body">
        <div class="timeline-event">${esc(e.event)}</div>
        <div class="timeline-time">${fmtTime(e.time)}</div>
        ${e.types?.length ? badges(e.types) : ""}
      </div>
    </div>`).join("");
}

export function initAnprPage(params = {}) {
  initAnpr();
  if (params.q) {
    const input = $("#anpr-query");
    if (input) input.value = params.q;
    searchPlates(params.q);
  }
}
