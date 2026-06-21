import { getViolationsTotal, getAnalyticsSummary, deriveHighSeverity } from "./data-bridge.js";
import { $ } from "./formatters.js";

export async function updateBadges() {
  try {
    const total = await getViolationsTotal();
    const vc = $("#nav-violations-count");
    if (vc) {
      vc.textContent = total > 99 ? "99+" : String(total);
      vc.classList.toggle("hidden", total === 0);
    }
  } catch (_) {}

  const ac = $("#nav-alerts-count");
  if (ac) {
    try {
      const summary = await getAnalyticsSummary();
      const high = deriveHighSeverity(summary?.by_type);
      ac.textContent = String(high);
      ac.classList.toggle("hidden", !high);
    } catch (_) {
      ac.classList.add("hidden");
    }
  }
}
