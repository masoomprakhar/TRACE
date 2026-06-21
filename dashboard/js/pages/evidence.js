import { api } from "../api.js";
import { $, esc, fmtTime, pct } from "../formatters.js";
import { toast, violationCard, emptyState, animateConfBars, imageFallback, badges, plateChip } from "../components.js";
import { invalidateCache } from "../data-bridge.js";
import { updateBadges } from "../nav-badges.js";

export function initEvidence() {
  const dz = $("#evidence-dropzone");
  const input = $("#evidence-file-input");
  $("#evidence-choose")?.addEventListener("click", (e) => { e.stopPropagation(); input?.click(); });
  dz?.addEventListener("click", () => input?.click());
  input?.addEventListener("change", () => {
    if (input.files?.[0]) analyzeFile(input.files[0]);
    input.value = "";
  });
  ["dragenter", "dragover"].forEach((evt) =>
    dz?.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.add("dragover"); }));
  ["dragleave", "drop"].forEach((evt) =>
    dz?.addEventListener(evt, (e) => { e.preventDefault(); dz.classList.remove("dragover"); }));
  dz?.addEventListener("drop", (e) => {
    const f = e.dataTransfer?.files?.[0];
    if (f) analyzeFile(f);
  });
  $("#evidence-filter-type")?.addEventListener("change", loadGallery);
  $("#evidence-refresh")?.addEventListener("click", loadGallery);
}

async function analyzeFile(file) {
  if (!file.type.startsWith("image/")) {
    toast("Please upload a JPG or PNG image.", "error");
    return;
  }
  const status = $("#evidence-upload-status");
  if (status) status.textContent = "Analyzing…";
  try {
    const r = await api.postFile("/api/analyze", file);
    const n = (r.violations || []).length;
    toast(n ? `${n} violation(s) detected.` : "Frame clear — no violations.", n ? "info" : "success");
    invalidateCache();
    updateBadges();
    loadGallery();
    if (status) status.textContent = "";
    if (r.annotated_url) showUploadResult(r);
  } catch (err) {
    toast(`Analysis failed: ${err.message}`, "error");
    if (status) status.textContent = "";
  }
}

function showUploadResult(r) {
  const panel = $("#evidence-upload-result");
  if (!panel) return;
  panel.classList.remove("hidden");
  panel.innerHTML = `
    <div class="card">
      <img src="${esc(r.annotated_url)}" alt="Annotated" class="evidence-upload-img" />
      <div class="p-4">
        <div class="text-sm text-muted mb-2">${(r.violations || []).length} violation(s) · ${Math.round(r.processing_ms || 0)} ms</div>
        <div class="violation-list">${(r.violations || []).map(violationCard).join("") || emptyState("No violations")}</div>
      </div>
    </div>`;
  animateConfBars(panel);
}

export async function loadGallery() {
  const grid = $("#evidence-grid");
  if (!grid) return;
  grid.innerHTML = '<div class="skeleton-card"></div>'.repeat(6);
  const type = $("#evidence-filter-type")?.value || "";
  try {
    let url = "/api/violations?limit=50&offset=0";
    if (type) url += `&type=${encodeURIComponent(type)}`;
    const data = await api.get(url);
    const items = data.items || [];
    if (!items.length) {
      grid.innerHTML = emptyState("No evidence yet", "Upload a frame or seed demo data.");
      return;
    }
    grid.innerHTML = items.map((it) => `
      <div class="evidence-card" data-id="${esc(it.id)}">
        <div class="evidence-card-img-wrap">
          <img src="${esc(it.evidence_url)}" alt="" loading="lazy" class="evidence-card-img" />
          <div class="evidence-card-overlay">${badges(it.violation_types)}</div>
        </div>
        <div class="evidence-card-body">
          ${plateChip(it.plate_number)}
          <div class="evidence-card-meta">${fmtTime(it.timestamp)} · ${esc(it.location || "—")}</div>
          <div class="text-xs text-muted">${pct(it.confidence)} confidence</div>
        </div>
      </div>`).join("");
    grid.querySelectorAll(".evidence-card").forEach((card) => {
      card.addEventListener("click", () => window.openEvidence?.(card.dataset.id));
    });
  } catch (err) {
    grid.innerHTML = emptyState("Couldn't load evidence", err.message);
  }
}

export function initEvidencePage() {
  initEvidence();
  loadGallery();
}
