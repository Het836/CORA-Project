"use strict";

/* ---------- Config & state ---------- */
const API = ""; // same origin as FastAPI; set to "http://localhost:8000" if serving the UI elsewhere
const NUM_FEATURES = 1433;
const CORA_NODES = 2708;
const CLASSES = [
  "Case Based", "Genetic Algorithms", "Neural Networks", "Probabilistic Methods",
  "Reinforcement Learning", "Rule Learning", "Theory",
];
const color = (i) => `var(--c${i})`;

const state = { last: null, sortByConf: false, busy: false };
const $ = (id) => document.getElementById(id);

/* ---------- Small helpers ---------- */
function el(tag, props = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (k === "class") node.className = v;
    else if (k === "style") node.style.cssText = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) if (c) node.append(c);
  return node;
}

function toast(message, type = "error") {
  const t = el("div", { class: `toast ${type === "info" ? "info" : ""}`, text: message });
  $("toasts").append(t);
  setTimeout(() => t.remove(), 6000);
}

const pct = (x, d = 1) => `${(x * 100).toFixed(d)}%`;

function setBusy(busy, text = "Running the model") {
  state.busy = busy;
  $("loading").classList.toggle("hidden", !busy);
  $("loading-text").textContent = text;
  document.querySelectorAll("#cora-run, #custom-run").forEach((b) => (b.disabled = busy));
  if (busy) { $("empty").classList.add("hidden"); $("output").classList.add("hidden"); }
}

async function api(path, options) {
  const res = await fetch(API + path, options);
  let body = null;
  try { body = await res.json(); } catch { /* non-JSON body */ }
  if (!res.ok) {
    let detail = body && body.detail;
    if (Array.isArray(detail)) detail = detail.map((d) => d.msg || JSON.stringify(d)).join("; ");
    throw new Error(detail || `Request failed (${res.status})`);
  }
  return body;
}

/* ---------- Legends ---------- */
function fillLegend(ul, counts) {
  ul.replaceChildren(
    ...CLASSES.map((name, i) =>
      el("li", {}, [
        el("span", { class: "swatch", style: `background:${color(i)}` }),
        el("span", { text: counts ? `${name} (${counts[i]})` : name }),
      ])
    )
  );
}
fillLegend($("legend-empty"));

/* ---------- Tabs ---------- */
document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => {
      const on = b === btn;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on);
      $("tab-" + b.dataset.tab).classList.toggle("hidden", !on);
    });
    if (btn.dataset.tab === "model") loadModelInfo();
  })
);

/* ---------- Health ---------- */
async function checkHealth() {
  const box = $("status");
  try {
    const h = await api("/health");
    const provider = Array.isArray(h.provider) ? h.provider[0] : h.provider;
    box.className = "status up";
    $("status-text").textContent = `API online${provider ? " · " + provider.replace("ExecutionProvider", "") : ""}`;
  } catch {
    box.className = "status down";
    $("status-text").textContent = "API unreachable";
  }
}
checkHealth();
setInterval(checkHealth, 15000);

/* ---------- Cora tab ---------- */
function parseIndices(text) {
  const good = [], bad = [];
  for (const raw of text.split(/[\s,;]+/).filter(Boolean)) {
    const m = raw.match(/^(\d+)(?:-(\d+))?$/);
    if (!m) { bad.push(raw); continue; }
    const a = +m[1], b = m[2] !== undefined ? +m[2] : a;
    if (b < a || b - a > 200) { bad.push(raw); continue; }
    for (let i = a; i <= b; i++) good.push(i);
  }
  const unique = [...new Set(good)];
  return { indices: unique, bad: [...bad, ...unique.filter((i) => i >= CORA_NODES).map(String)] };
}

function renderChips() {
  const { indices, bad } = parseIndices($("cora-input").value);
  const shown = indices.slice(0, 24).map((i) =>
    el("span", { class: `chip ${i >= CORA_NODES ? "bad" : ""}`, text: `#${i}` })
  );
  const extra = indices.length > 24 ? [el("span", { class: "chip", text: `+${indices.length - 24} more` })] : [];
  const invalid = bad.filter((b) => !(+b >= CORA_NODES)).map((b) => el("span", { class: "chip bad", text: `"${b}" invalid` }));
  $("cora-chips").replaceChildren(...shown, ...extra, ...invalid);
}
$("cora-input").addEventListener("input", renderChips);
renderChips();

$("cora-random").addEventListener("click", () => {
  const picks = new Set();
  while (picks.size < 6) picks.add(Math.floor(Math.random() * CORA_NODES));
  $("cora-input").value = [...picks].sort((a, b) => a - b).join(", ");
  renderChips();
});

$("cora-run").addEventListener("click", async () => {
  const { indices, bad } = parseIndices($("cora-input").value);
  if (!indices.length) return toast("Enter at least one paper index.");
  if (bad.length) return toast(`Fix these entries first: ${bad.slice(0, 4).join(", ")}. Indices must be 0 to ${CORA_NODES - 1}.`);
  await classify("/predict/cora_node", { node_indices: indices }, "Loading Cora and classifying");
});

/* ---------- Custom graph tab ---------- */
$("node-count").addEventListener("input", (e) => ($("node-count-val").textContent = e.target.value));

$("gen-graph").addEventListener("click", () => {
  const n = +$("node-count").value;
  const features = Array.from({ length: n }, () =>
    Array.from({ length: NUM_FEATURES }, () => (Math.random() < 0.013 ? 1 : 0))
  );
  const body = { node_features: features };
  if ($("with-edges").checked) {
    const src = [], dst = [];
    for (let i = 0; i < n; i++) {
      const j = (i + 1 + Math.floor(Math.random() * (n - 1))) % n;
      src.push(i, j); dst.push(j, i); // undirected: add both directions
    }
    body.edge_indices = [src, dst];
  }
  $("graph-json").value = JSON.stringify(body);
  validateGraph();
});

$("graph-file").addEventListener("change", async (e) => {
  const file = e.target.files[0];
  if (!file) return;
  $("graph-json").value = await file.text();
  validateGraph();
  e.target.value = "";
});
$("graph-json").addEventListener("input", validateGraph);

function validateGraph() {
  const meta = $("graph-meta");
  const text = $("graph-json").value.trim();
  const fail = (msg) => { meta.className = "graph-meta bad"; meta.textContent = msg; return null; };
  if (!text) { meta.className = "graph-meta"; meta.textContent = "Nothing loaded yet."; return null; }
  let data;
  try { data = JSON.parse(text); } catch { return fail("This is not valid JSON yet."); }
  const nf = data.node_features;
  if (!Array.isArray(nf) || !nf.length) return fail("node_features must be a non-empty list of rows.");
  const badRow = nf.findIndex((r) => !Array.isArray(r) || r.length !== NUM_FEATURES);
  if (badRow !== -1) return fail(`Row ${badRow} needs exactly ${NUM_FEATURES} numbers.`);
  let edges = 0;
  if (data.edge_indices) {
    const ei = data.edge_indices;
    if (!Array.isArray(ei) || ei.length !== 2 || ei[0].length !== ei[1].length) return fail("edge_indices must be [[sources], [targets]] with equal lengths.");
    if ([...ei[0], ...ei[1]].some((v) => !Number.isInteger(v) || v < 0 || v >= nf.length)) return fail(`Edge ids must be whole numbers from 0 to ${nf.length - 1}.`);
    edges = ei[0].length;
  }
  meta.className = "graph-meta ok";
  meta.textContent = `${nf.length} nodes, ${edges ? edges + " edges" : "no edges (each node only links to itself)"}, ${NUM_FEATURES} features each.`;
  return data;
}

$("custom-run").addEventListener("click", async () => {
  const data = validateGraph();
  if (!data) return toast("Load or generate a valid graph first.");
  await classify("/predict", data, "Classifying your graph");
});

/* ---------- Model tab ---------- */
let modelLoaded = false;
async function loadModelInfo() {
  if (modelLoaded) return;
  const box = $("model-info");
  const classList = el("dl", { class: "kv" }, CLASSES.flatMap((n, i) => [
    el("dt", { text: `Class ${i}` }),
    el("dd", {}, [el("span", { class: "swatch", style: `background:${color(i)};display:inline-block;margin-right:8px` }), n]),
  ]));
  try {
    const info = await api("/model_info");
    const io = (list) => (list || []).map((x) =>
      el("div", { class: "io" }, [el("b", { text: x.name }), el("span", { text: `${x.type} · shape [${(x.shape || []).join(", ")}]` })])
    );
    box.replaceChildren(
      el("dl", { class: "kv" }, [
        el("dt", { text: "Model" }), el("dd", { text: info["Model Name"] || "SimpleGCN" }),
        el("dt", { text: "Features" }), el("dd", { text: String(info.feature_dimension) }),
        el("dt", { text: "Classes" }), el("dd", { text: String(info.num_classes) }),
      ]),
      el("h2", { text: "Inputs" }), ...io(info.inputs),
      el("h2", { text: "Outputs" }), ...io(info.outputs),
      el("h2", { text: "Topics" }), classList
    );
    modelLoaded = true;
  } catch {
    box.replaceChildren(
      el("p", { class: "hint", text: "The /model_info route is not available. In api.py, add @app.get('/model_info') above the model_info() function to enable live details here." }),
      el("dl", { class: "kv" }, [
        el("dt", { text: "Features" }), el("dd", { text: String(NUM_FEATURES) }),
        el("dt", { text: "Classes" }), el("dd", { text: "7" }),
      ]),
      el("h2", { text: "Topics" }), classList
    );
  }
}

/* ---------- Classification & rendering ---------- */
async function classify(path, body, loadingText) {
  if (state.busy) return;
  setBusy(true, loadingText);
  const t0 = performance.now();
  try {
    const data = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    state.last = { data, ms: performance.now() - t0 };
    setBusy(false);
    render();
    $("output").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    setBusy(false);
    (state.last ? $("output") : $("empty")).classList.remove("hidden");
    toast(err.message || "Something went wrong. Check the server logs.");
  }
}

function render() {
  const { data, ms } = state.last;
  const preds = data.predictions || [];
  const counts = Array(CLASSES.length).fill(0);
  preds.forEach((p) => counts[p.predicted_classes_id]++);
  const avgConf = preds.reduce((s, p) => s + Math.max(...p.probabilities), 0) / (preds.length || 1);
  const topClass = counts.indexOf(Math.max(...counts));

  $("stats").replaceChildren(
    ...[
      [preds.length, "papers classified"],
      [data.num_nodes.toLocaleString(), "nodes in graph"],
      [data.num_edges.toLocaleString(), "edges in graph"],
      [pct(avgConf, 0), "average confidence"],
      [`${Math.round(ms)} ms`, "round trip"],
    ].map(([v, l]) => el("div", { class: "stat" }, [el("b", { text: String(v) }), el("span", { text: l })]))
  );

  // Topic mix bar
  $("mixbar").replaceChildren(
    ...counts.map((c, i) => el("div", { style: `background:${color(i)};flex-grow:${c};flex-basis:0`, title: `${CLASSES[i]}: ${c}` }))
  );
  fillLegend($("mix-legend"), counts);
  $("mix-legend").children[topClass].style.color = "var(--text)";

  renderCards();
  $("output").classList.remove("hidden");
}

function renderCards() {
  const { data } = state.last;
  let preds = [...data.predictions];
  if (state.sortByConf) preds.sort((a, b) => Math.max(...b.probabilities) - Math.max(...a.probabilities));

  $("cards").replaceChildren(
    ...preds.map((p) => {
      const cls = p.predicted_classes_id;
      const conf = Math.max(...p.probabilities);
      const spectrum = el("div", { class: "spectrum", role: "img", "aria-label": "Probability spread across topics" },
        p.probabilities.map((v, i) => el("div", { style: `background:${color(i)};flex:${Math.max(v, 0.002)} 0 0`, title: `${CLASSES[i]}: ${pct(v)}` }))
      );
      const rows = p.probabilities
        .map((v, i) => ({ v, i, logit: p.logits[i] }))
        .sort((a, b) => b.v - a.v)
        .map((r, idx) => el("tr", { class: idx === 0 ? "top" : "" }, [
          el("td", {}, [el("span", { class: "name" }, [el("span", { class: "swatch", style: `background:${color(r.i)}` }), CLASSES[r.i]])]),
          el("td", { text: pct(r.v, 2) }),
          el("td", { text: r.logit.toFixed(2) }),
        ]));
      const table = el("table", { class: "prob-table" }, [
        el("thead", {}, [el("tr", {}, [el("td", { text: "Topic" }), el("td", { text: "Probability" }), el("td", { text: "Logit" })])]),
        el("tbody", {}, rows),
      ]);
      return el("article", { class: "card", style: `--cc:${color(cls)}` }, [
        el("div", { class: "card-head" }, [
          el("span", { class: "node", text: `Paper #${p.node_index}` }),
          el("span", { class: "conf", text: pct(conf, 1) }),
        ]),
        el("div", { class: "topic", text: CLASSES[cls] || p.predicted_classes_name }),
        spectrum,
        el("details", {}, [el("summary", { text: "All topic scores" }), table]),
      ]);
    })
  );
  $("sort-toggle").textContent = `Sort: ${state.sortByConf ? "highest confidence" : "node order"}`;
}

$("sort-toggle").addEventListener("click", () => { state.sortByConf = !state.sortByConf; renderCards(); });

$("copy-json").addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(JSON.stringify(state.last.data, null, 2));
    toast("Response copied to clipboard.", "info");
  } catch {
    toast("Copy was blocked by the browser.");
  }
});

/* ---------- Ambient background: a faint citation network ---------- */
(function network() {
  const canvas = $("bg"), ctx = canvas.getContext("2d");
  const still = matchMedia("(prefers-reduced-motion: reduce)").matches;
  let w, h, nodes;
  const init = () => {
    w = canvas.width = innerWidth; h = canvas.height = innerHeight;
    const n = Math.min(70, Math.floor((w * h) / 22000));
    nodes = Array.from({ length: n }, (_, i) => ({
      x: Math.random() * w, y: Math.random() * h,
      vx: (Math.random() - 0.5) * 0.25, vy: (Math.random() - 0.5) * 0.25, c: i % 7,
    }));
  };
  const frame = () => {
    ctx.clearRect(0, 0, w, h);
    const style = getComputedStyle(document.documentElement);
    for (const a of nodes) {
      if (!still) { a.x += a.vx; a.y += a.vy; if (a.x < 0 || a.x > w) a.vx *= -1; if (a.y < 0 || a.y > h) a.vy *= -1; }
      for (const b of nodes) {
        const d = Math.hypot(a.x - b.x, a.y - b.y);
        if (d < 130 && a < b) { ctx.strokeStyle = `rgba(143,152,174,${0.16 * (1 - d / 130)})`; ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); }
      }
      ctx.fillStyle = style.getPropertyValue(`--c${a.c}`).trim() || "#7c8cff";
      ctx.globalAlpha = 0.5; ctx.beginPath(); ctx.arc(a.x, a.y, 2, 0, 6.283); ctx.fill(); ctx.globalAlpha = 1;
    }
    if (!still) requestAnimationFrame(frame);
  };
  addEventListener("resize", () => { init(); if (still) frame(); });
  init(); frame();
})();
