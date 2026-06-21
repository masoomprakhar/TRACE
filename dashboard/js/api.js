/** API layer — same-origin relative paths under /api. */
export const api = {
  async get(path) {
    const res = await fetch(path, { headers: { Accept: "application/json" } });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
  async getText(path) {
    const res = await fetch(path);
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.text();
  },
  async postFile(path, file, params = {}) {
    const fd = new FormData();
    fd.append("file", file);
    const qs = new URLSearchParams(params).toString();
    const url = qs ? `${path}?${qs}` : path;
    const res = await fetch(url, { method: "POST", body: fd });
    if (!res.ok) {
      let detail = `${res.status} ${res.statusText}`;
      try { const j = await res.json(); if (j.detail) detail = j.detail; } catch (_) {}
      throw new Error(detail);
    }
    return res.json();
  },
  async post(path) {
    const res = await fetch(path, { method: "POST" });
    if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
    return res.json();
  },
};
