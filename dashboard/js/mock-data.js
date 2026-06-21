/** Mock data for dashboard widgets not backed by API fields. */
export const MOCK = {
  user: { name: "Prakhar Singh", role: "Administrator", initials: "PS" },
  weather: { temp: 32, city: "New Delhi", condition: "sunny" },
  kpis: {
    totalViolations: 1248,
    vehiclesScanned: 23842,
    highSeverity: 312,
    activeCameras: 156,
    challansIssued: 974,
    trends: {
      totalViolations: { delta: 18.6, up: true },
      vehiclesScanned: { delta: 23.4, up: true },
      highSeverity: { delta: 11.2, up: true },
      challansIssued: { delta: 15.7, up: true },
    },
    sparklines: {
      totalViolations: [820, 890, 940, 1010, 1080, 1150, 1200, 1248],
      vehiclesScanned: [18000, 19200, 20100, 21000, 21800, 22500, 23200, 23842],
      highSeverity: [240, 255, 268, 280, 290, 300, 308, 312],
      challansIssued: [720, 780, 820, 860, 900, 930, 960, 974],
    },
  },
  severity: {
    critical: { count: 87, pct: 28 },
    high: { count: 156, pct: 50 },
    medium: { count: 53, pct: 17 },
    low: { count: 16, pct: 5 },
  },
  monthlyTrend: [
    { day: "Jun 1", count: 680 }, { day: "Jun 3", count: 720 }, { day: "Jun 5", count: 810 },
    { day: "Jun 7", count: 890 }, { day: "Jun 9", count: 920 }, { day: "Jun 11", count: 980 },
    { day: "Jun 13", count: 1050 }, { day: "Jun 15", count: 1120 }, { day: "Jun 17", count: 1190 },
    { day: "Jun 18", count: 1248 },
  ],
  hotspots: [
    { id: 1, name: "Ring Road", count: 312, peakHour: "08:00", lat: 28.65, lng: 77.23, trend: "up" },
    { id: 2, name: "ITO Cross", count: 248, peakHour: "18:00", lat: 28.63, lng: 77.24, trend: "up" },
    { id: 3, name: "AIIMS Flyover", count: 196, peakHour: "09:00", lat: 28.57, lng: 77.21, trend: "flat" },
    { id: 4, name: "Moolchand", count: 164, peakHour: "17:00", lat: 28.56, lng: 77.24, trend: "down" },
  ],
  cameras: [
    { id: 44, name: "Cam 44", location: "Ring Road - Sector A", status: "online" },
    { id: 45, name: "Cam 45", location: "Ring Road - Sector B", status: "online" },
    { id: 46, name: "Cam 46", location: "ITO Cross", status: "online" },
    { id: 47, name: "Cam 47", location: "Ring Road - AI Cam 47", status: "online", active: true },
    { id: 48, name: "Cam 48", location: "AIIMS Flyover", status: "online" },
    { id: 49, name: "Cam 49", location: "Moolchand", status: "online" },
    { id: 50, name: "Cam 50", location: "Saket", status: "offline" },
  ],
  recentSearches: ["DL01AB1234", "MH12DE5678", "KA03FG9012", "HR26BK3456"],
  challans: [
    { id: "CH-2026-8841", plate: "DL01AB1234", amount: 1000, status: "issued", types: ["no_helmet"] },
    { id: "CH-2026-8840", plate: "MH12DE5678", amount: 2000, status: "paid", types: ["red_light"] },
    { id: "CH-2026-8839", plate: "KA03FG9012", amount: 1500, status: "pending", types: ["triple_riding"] },
  ],
  offenders: [
    { plate: "DL01AB1234", count: 12, lastSeen: "2026-06-18T08:30:00Z", types: ["no_helmet", "triple_riding"], risk: 92 },
    { plate: "MH12DE5678", count: 9, lastSeen: "2026-06-17T14:20:00Z", types: ["red_light"], risk: 85 },
    { plate: "KA03FG9012", count: 7, lastSeen: "2026-06-16T11:45:00Z", types: ["no_seatbelt"], risk: 78 },
    { plate: "HR26BK3456", count: 6, lastSeen: "2026-06-15T19:10:00Z", types: ["wrong_side"], risk: 74 },
    { plate: "UP14CD7890", count: 5, lastSeen: "2026-06-14T07:55:00Z", types: ["illegal_parking"], risk: 68 },
    { plate: "RJ14EF2345", count: 5, lastSeen: "2026-06-13T16:30:00Z", types: ["stop_line"], risk: 65 },
    { plate: "TN09GH6789", count: 4, lastSeen: "2026-06-12T10:15:00Z", types: ["no_helmet"], risk: 58 },
    { plate: "GJ01IJ0123", count: 4, lastSeen: "2026-06-11T13:40:00Z", types: ["red_light"], risk: 55 },
    { plate: "WB02KL4567", count: 3, lastSeen: "2026-06-10T09:25:00Z", types: ["triple_riding"], risk: 48 },
    { plate: "PB10MN8901", count: 3, lastSeen: "2026-06-09T18:50:00Z", types: ["no_seatbelt"], risk: 42 },
  ],
  alerts: [
    { id: "a1", type: "spike", severity: "high", title: "Violation spike on Ring Road", message: "47 violations in the last hour — 3× above average.", read: false, time: "2026-06-18T10:30:00Z", action: { view: "violations" } },
    { id: "a2", type: "camera", severity: "medium", title: "Camera Cam 50 offline", message: "Saket junction feed lost connection.", read: false, time: "2026-06-18T09:15:00Z", action: { view: "live" } },
    { id: "a3", type: "cluster", severity: "critical", title: "High-severity cluster detected", message: "12 red-light violations at ITO Cross in 30 min.", read: false, time: "2026-06-18T08:45:00Z", action: { view: "hotspots" } },
    { id: "a4", type: "model", severity: "low", title: "OCR model warming up", message: "Plate read latency elevated — check Settings.", read: true, time: "2026-06-17T22:00:00Z", action: { view: "settings" } },
    { id: "a5", type: "spike", severity: "medium", title: "Triple riding surge", message: "18 triple-riding detections near AIIMS Flyover.", read: true, time: "2026-06-17T18:20:00Z", action: { view: "violations", filter: "triple_riding" } },
  ],
  liveOverlays: [
    { label: "Triple Riding", pct: 95, color: "#EF4444", style: "top:28%;left:12%;width:22%;height:18%" },
    { label: "No Seatbelt", pct: 92, color: "#0D6EFD", style: "top:42%;left:38%;width:18%;height:16%" },
    { label: "No Helmet", pct: 97, color: "#F59E0B", style: "top:35%;left:62%;width:16%;height:20%" },
  ],
};

export function getUnreadAlertCount() {
  return MOCK.alerts.filter((a) => !a.read).length;
}

export function markAllAlertsRead() {
  MOCK.alerts.forEach((a) => { a.read = true; });
}

export function addLiveAlert(violation) {
  MOCK.alerts.unshift({
    id: `live-${Date.now()}`,
    type: "detection",
    severity: "high",
    title: `High-confidence ${violation.label}`,
    message: `Detected at ${violation.confidence}% confidence on live feed.`,
    read: false,
    time: new Date().toISOString(),
    action: { view: "live" },
  });
}
