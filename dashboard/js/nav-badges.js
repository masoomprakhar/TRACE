import { getViolationsTotal } from "./data-bridge.js";
import { getUnreadAlertCount } from "./mock-data.js";
import { $ } from "./formatters.js";

export async function updateBadges() {
  try {
    const total = await getViolationsTotal();
    const vc = $("#nav-violations-count");
    if (vc) vc.textContent = total > 99 ? "99+" : String(total);
  } catch (_) {}
  const ac = $("#nav-alerts-count");
  if (ac) {
    const n = getUnreadAlertCount();
    ac.textContent = String(n);
    ac.classList.toggle("hidden", n === 0);
  }
}
