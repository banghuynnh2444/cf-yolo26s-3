"use strict";

const $ = (id) => document.getElementById(id);
const view = $("view");
const vctx = view.getContext("2d");

const S = {
  classes: [],
  conf: {},          // name -> ngưỡng
  visible: {},       // name -> hiển thị trên ảnh
  mode: "camera",
  ws: null,
  streaming: false,
  waiting: false,
  sentAt: 0,
  frameTimes: [],
  pending: document.createElement("canvas"),   // khung đang chờ kết quả
  shown: document.createElement("canvas"),     // khung đang hiển thị
  lastSource: null,
  lastRes: null,
  highlight: -1,
  images: [],        // {name, url, img, res}
  current: -1,
  spark: [],
  log: [],
  lastLog: 0,
  stream: null,
};
const video = document.createElement("video");
video.muted = true;
video.playsInline = true;

const SEVERITY = [
  [0, "Khỏe", "#2f9e44"],
  [5, "Rất nhẹ", "#94d82d"],
  [10, "Nhẹ", "#fab005"],
  [15, "Nặng", "#f76707"],
  [Infinity, "Rất nặng", "#e03131"],
];

function severity(pct) {
  if (pct == null) return null;
  if (pct <= 0) return SEVERITY[0];
  for (let i = 1; i < SEVERITY.length; i++) if (pct <= SEVERITY[i][0]) return SEVERITY[i];
  return SEVERITY[SEVERITY.length - 1];
}

const fmt = (v, d = 1) => (v == null ? "–" : Number(v).toFixed(d));
const cls = (name) => S.classes.find((c) => c.name === name) || { vi: name, color: "#888" };
const hexA = (hex, a) => {
  const n = parseInt(hex.slice(1), 16);
  return `rgba(${n >> 16},${(n >> 8) & 255},${n & 255},${a})`;
};

// ---------- khởi tạo ----------
async function loadInfo() {
  const info = await (await fetch("/api/info")).json();
  S.classes = info.classes;
  for (const c of S.classes) {
    if (!(c.name in S.conf)) S.conf[c.name] = c.conf;
    if (!(c.name in S.visible)) S.visible[c.name] = true;
  }
  const sel = $("modelSelect");
  sel.innerHTML = "";
  for (const m of info.models) {
    const o = document.createElement("option");
    o.value = m;
    o.textContent = m + (m === info.models[0] ? "  (tốt nhất)" : "");
    if (m === info.model) o.selected = true;
    sel.appendChild(o);
  }
  const b = $("deviceBadge");
  b.textContent = info.gpu ? `GPU: ${info.gpu}${info.half ? " · FP16" : ""}` : `Thiết bị: ${info.device}`;
  b.className = "badge" + (info.gpu ? " gpu" : "");
  buildConfs();
  renderStats(null);
}

function buildConfs() {
  const box = $("confs");
  box.innerHTML = "";
  for (const c of S.classes) {
    const row = document.createElement("div");
    row.className = "conf-row";
    row.innerHTML = `
      <label><input type="checkbox" ${S.visible[c.name] ? "checked" : ""}>
        <span class="dot" style="background:${c.color}"></span>${c.vi}</label>
      <input type="range" min="0.05" max="0.95" step="0.05" value="${S.conf[c.name]}">
      <span>${fmt(S.conf[c.name], 2)}</span>`;
    const [chk, rng, out] = [row.querySelector("input[type=checkbox]"), row.querySelector("input[type=range]"), row.querySelector("span:last-child")];
    chk.onchange = () => { S.visible[c.name] = chk.checked; redraw(); };
    rng.oninput = () => { S.conf[c.name] = +rng.value; out.textContent = fmt(rng.value, 2); };
    rng.onchange = () => settingsChanged();
    box.appendChild(row);
  }
}

$("modelSelect").onchange = async (e) => {
  const fd = new FormData();
  fd.append("path", e.target.value);
  $("deviceBadge").textContent = "đang nạp model…";
  const r = await fetch("/api/model", { method: "POST", body: fd });
  if (!r.ok) alert("Không đổi được model");
  await loadInfo();
  settingsChanged();
};

// ---------- tab nguồn ----------
document.querySelectorAll(".tab").forEach((t) => {
  t.onclick = () => setMode(t.dataset.src);
});

function setMode(mode) {
  stopStream();
  S.mode = mode;
  document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t.dataset.src === mode));
  for (const m of ["camera", "video", "image"]) $("bar-" + m).classList.toggle("hidden", m !== mode);
  $("gallery").classList.toggle("hidden", mode !== "image" || !S.images.length);
  $("imgsz").value = mode === "image" ? "1024" : "640";
  $("sparkPanel").classList.toggle("hidden", mode === "image");
  if (mode !== "image") { S.spark = []; drawSpark(); }
  if (mode === "image" && S.current >= 0) showImage(S.current);
}

// ---------- WebSocket ----------
function wsConfig() {
  return JSON.stringify({ imgsz: +$("imgsz").value, conf: S.conf, leaf: $("leafEst").checked });
}

function openWS() {
  return new Promise((resolve, reject) => {
    if (S.ws && S.ws.readyState === 1) return resolve(S.ws);
    const ws = new WebSocket((location.protocol === "https:" ? "wss://" : "ws://") + location.host + "/ws");
    ws.binaryType = "arraybuffer";
    ws.onopen = () => { ws.send(wsConfig()); S.ws = ws; resolve(ws); };
    ws.onerror = () => reject(new Error("Không kết nối được máy chủ"));
    ws.onclose = () => { S.ws = null; S.waiting = false; };
    ws.onmessage = (ev) => onStreamResult(JSON.parse(ev.data));
  });
}

function settingsChanged() {
  if (S.ws && S.ws.readyState === 1) S.ws.send(wsConfig());
  if (S.mode === "image" && S.current >= 0) {
    clearTimeout(S._t);
    S._t = setTimeout(() => predictImage(S.current, true), 200);
  }
}
["imgsz", "leafEst"].forEach((id) => ($(id).onchange = settingsChanged));
["alpha", "showMask", "showBox", "showLabel"].forEach((id) => ($(id).oninput = redraw));

// ---------- camera / video ----------
async function listCameras() {
  try {
    const devs = await navigator.mediaDevices.enumerateDevices();
    const sel = $("camSelect");
    const cur = sel.value;
    sel.innerHTML = '<option value="">Camera mặc định</option>';
    devs.filter((d) => d.kind === "videoinput").forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Camera ${i + 1}`;
      sel.appendChild(o);
    });
    sel.value = cur;
  } catch (_) { /* bỏ qua */ }
}

$("camStart").onclick = async () => {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert("Trình duyệt chặn camera. Mở bằng http://localhost hoặc chạy app với --https khi dùng điện thoại.");
    return;
  }
  stopStream();
  try {
    const id = $("camSelect").value;
    const constraints = {
      video: id ? { deviceId: { exact: id }, width: { ideal: 1280 }, height: { ideal: 720 } }
                : { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    };
    S.stream = await navigator.mediaDevices.getUserMedia(constraints);
    video.srcObject = S.stream;
    await video.play();
    listCameras();
    startStream();
  } catch (e) {
    alert("Không mở được camera: " + e.message);
  }
};
$("camStop").onclick = stopStream;

$("vidStart").onclick = async () => {
  const f = $("videoFile").files[0];
  if (!f) return alert("Chọn file video trước");
  stopStream();
  video.srcObject = null;
  video.src = URL.createObjectURL(f);
  video.loop = false;
  video.onended = () => stopStream();
  await video.play();
  startStream();
};
$("vidStop").onclick = stopStream;
$("videoFile").onchange = () => $("vidStart").click();

async function startStream() {
  try { await openWS(); } catch (e) { alert(e.message); return; }
  S.streaming = true;
  S.waiting = false;
  S.frameTimes = [];
  $("placeholder").classList.add("hidden");
  pump();
}

function stopStream() {
  S.streaming = false;
  if (S.stream) { S.stream.getTracks().forEach((t) => t.stop()); S.stream = null; }
  if (!video.paused) video.pause();
}

function pump() {
  if (!S.streaming || S.waiting) return;
  if (video.readyState < 2) { requestAnimationFrame(pump); return; }
  const vw = video.videoWidth, vh = video.videoHeight;
  const scale = Math.min(1, 1280 / Math.max(vw, vh));
  const c = S.pending;
  c.width = Math.round(vw * scale);
  c.height = Math.round(vh * scale);
  c.getContext("2d").drawImage(video, 0, 0, c.width, c.height);
  S.waiting = true;
  c.toBlob((blob) => {
    if (!S.ws || S.ws.readyState !== 1 || !blob) { S.waiting = false; return; }
    S.sentAt = performance.now();
    blob.arrayBuffer().then((buf) => S.ws.send(buf));
  }, "image/jpeg", 0.82);
}

function onStreamResult(res) {
  S.waiting = false;
  if (res.error) { if (S.streaming) requestAnimationFrame(pump); return; }
  const now = performance.now();
  res.latency_ms = now - S.sentAt;
  S.frameTimes.push(now);
  while (S.frameTimes.length && now - S.frameTimes[0] > 2000) S.frameTimes.shift();
  res.fps = S.frameTimes.length > 1 ? (S.frameTimes.length - 1) / ((now - S.frameTimes[0]) / 1000) : null;

  // đổi vai 2 canvas: khung vừa xử lý thành khung hiển thị
  [S.shown, S.pending] = [S.pending, S.shown];
  show(S.shown, res);
  if (res.total > 0 && now - S.lastLog > 1000) {
    S.lastLog = now;
    addLog(S.mode === "camera" ? "camera" : ($("videoFile").files[0]?.name || "video") + ` @${video.currentTime.toFixed(1)}s`, res);
  }
  if (S.streaming) requestAnimationFrame(pump);
}

// ---------- ảnh tĩnh ----------
$("imageFiles").onchange = (e) => addImages([...e.target.files]);
const stage = $("stage");
stage.ondragover = (e) => { e.preventDefault(); stage.classList.add("drag"); };
stage.ondragleave = () => stage.classList.remove("drag");
stage.ondrop = (e) => {
  e.preventDefault();
  stage.classList.remove("drag");
  const files = [...e.dataTransfer.files].filter((f) => f.type.startsWith("image/"));
  if (files.length) { if (S.mode !== "image") setMode("image"); addImages(files); }
};

async function addImages(files) {
  const start = S.images.length;
  for (const f of files) {
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.src = url;
    S.images.push({ name: f.name, file: f, url, img, res: null });
  }
  renderGallery();
  for (let i = start; i < S.images.length; i++) {
    await predictImage(i, i === start);
  }
}

async function predictImage(i, display) {
  const it = S.images[i];
  if (!it) return;
  await it.img.decode().catch(() => {});
  const fd = new FormData();
  fd.append("file", it.file, it.name);
  fd.append("imgsz", $("imgsz").value);
  fd.append("conf", JSON.stringify(S.conf));
  fd.append("leaf", $("leafEst").checked);
  const t0 = performance.now();
  const r = await fetch("/api/predict", { method: "POST", body: fd });
  const res = await r.json();
  if (res.error) { alert(it.name + ": " + res.error); return; }
  res.latency_ms = performance.now() - t0;
  const first = !it.res;
  it.res = res;
  if (first) addLog(it.name, res);
  renderGallery();
  if (display || S.current === i) showImage(i);
}

function showImage(i) {
  S.current = i;
  const it = S.images[i];
  $("placeholder").classList.add("hidden");
  renderGallery();
  if (it.res) show(it.img, it.res);
}

function renderGallery() {
  const g = $("gallery");
  g.classList.toggle("hidden", S.mode !== "image" || !S.images.length);
  g.innerHTML = "";
  S.images.forEach((it, i) => {
    const d = document.createElement("div");
    d.className = "thumb" + (i === S.current ? " active" : "");
    const label = it.res ? `${it.res.total} vết · ${dominant(it.res)?.vi.split(" (")[0] || "khỏe"}` : "đang quét…";
    d.innerHTML = `<img src="${it.url}" alt=""><div title="${escapeHtml(it.name)}">${escapeHtml(it.name)}</div><div>${label}</div>`;
    d.onclick = () => showImage(i);
    g.appendChild(d);
  });
}

// ---------- vẽ ----------
function show(source, res) {
  S.lastSource = source;
  S.lastRes = res;
  S.highlight = -1;
  redraw();
  renderStats(res);
  if (S.mode !== "image") pushSpark(res);
  renderDetList(res);
}

function redraw() {
  const src = S.lastSource, res = S.lastRes;
  if (!src || !res) return;
  const w = res.width, h = res.height;
  if (view.width !== w || view.height !== h) { view.width = w; view.height = h; }
  vctx.drawImage(src, 0, 0, w, h);
  const k = Math.max(w, h) / 1000;
  const alpha = +$("alpha").value;
  const showMask = $("showMask").checked, showBox = $("showBox").checked, showLabel = $("showLabel").checked;
  res.detections.forEach((d, i) => {
    if (!S.visible[d.name]) return;
    const c = cls(d.name).color;
    const hl = i === S.highlight;
    if (showMask && d.poly.length) {
      vctx.beginPath();
      d.poly.forEach(([x, y], j) => (j ? vctx.lineTo(x, y) : vctx.moveTo(x, y)));
      vctx.closePath();
      vctx.fillStyle = hexA(c, hl ? Math.min(0.9, alpha + 0.3) : alpha);
      vctx.fill();
      vctx.lineWidth = Math.max(1, 1.5 * k);
      vctx.strokeStyle = c;
      vctx.stroke();
    }
    if (showBox || hl) {
      const [x1, y1, x2, y2] = d.box;
      vctx.lineWidth = Math.max(1.5, (hl ? 4 : 2) * k);
      vctx.strokeStyle = hl ? "#ffffff" : c;
      vctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    if (showLabel) {
      const [x1, y1] = d.box;
      const txt = `${d.name} ${(d.conf * 100).toFixed(0)}%`;
      vctx.font = `600 ${Math.max(11, 15 * k)}px system-ui, sans-serif`;
      const tw = vctx.measureText(txt).width + 8 * k;
      const th = Math.max(14, 20 * k);
      const ty = y1 - th < 0 ? y1 : y1 - th;
      vctx.fillStyle = c;
      vctx.fillRect(x1, ty, tw, th);
      vctx.fillStyle = "#fff";
      vctx.fillText(txt, x1 + 4 * k, ty + th * 0.78);
    }
  });
  $("hud").textContent = `${w}×${h} · imgsz ${res.imgsz} · ${res.total} vết`;
}

function dominant(res) {
  if (!res || !res.total) return null;
  const s = [...res.summary].sort((a, b) => b.area_px - a.area_px || b.count - a.count)[0];
  return s.count ? { ...cls(s.name), ...s } : null;
}

function renderStats(res) {
  $("fps").textContent = res?.fps != null ? fmt(res.fps, 1) : "–";
  $("infer").textContent = res ? `${fmt(res.infer_ms, 0)} ms` : "–";
  $("lat").textContent = res?.latency_ms != null ? `${fmt(res.latency_ms, 0)} ms` : "–";
  $("total").textContent = res ? res.total : "–";

  const body = $("statsBody");
  body.innerHTML = "";
  const maxPct = Math.max(0.001, ...(res ? res.summary.map((s) => s.pct_leaf ?? s.pct_frame) : [0]));
  for (const c of S.classes) {
    const s = res?.summary.find((x) => x.name === c.name);
    const pct = s ? (s.pct_leaf ?? s.pct_frame) : 0;
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><span class="dot" style="background:${c.color}"></span>${c.vi}
        <div class="bar"><i style="width:${(100 * pct) / maxPct}%;background:${c.color}"></i></div></td>
      <td>${s ? s.count : "–"}</td>
      <td>${s ? fmt(s.pct_frame, 2) : "–"}</td>
      <td>${s && s.pct_leaf != null ? fmt(s.pct_leaf, 2) : "–"}</td>
      <td>${s && s.count ? fmt(s.max_conf, 2) : "–"}</td>
      <td>${s && s.count ? fmt(s.mean_conf, 2) : "–"}</td>`;
    body.appendChild(tr);
  }
  $("statsFoot").innerHTML = res
    ? `<tr><td>Tổng</td><td>${res.total}</td><td>${fmt(res.lesion_pct_frame, 2)}</td>
       <td>${res.lesion_pct_leaf != null ? fmt(res.lesion_pct_leaf, 2) : "–"}</td><td></td><td></td></tr>`
    : "";
  $("leafNote").textContent = res && res.leaf_pct_frame != null
    ? `Lá chiếm khoảng ${fmt(res.leaf_pct_frame, 1)}% khung hình. Suy luận ${fmt(res.infer_ms, 0)} ms, hậu xử lý ${fmt(res.post_ms, 0)} ms.`
    : "";

  const v = $("verdict");
  if (!res) {
    v.querySelector(".vt").textContent = "Chưa có kết quả";
    v.querySelector(".vs").textContent = "";
    v.style.borderLeftColor = "var(--line)";
    return;
  }
  const dom = dominant(res);
  const sev = severity(res.lesion_pct_leaf);
  if (!dom) {
    v.querySelector(".vt").textContent = "✅ Không phát hiện vết bệnh";
    v.querySelector(".vs").textContent = "Không có vết nào vượt ngưỡng tin cậy đã đặt.";
    v.style.borderLeftColor = "#2f9e44";
    return;
  }
  const others = res.summary.filter((s) => s.count && s.name !== dom.name).map((s) => `${cls(s.name).vi.split(" (")[0]} (${s.count})`);
  v.querySelector(".vt").textContent = `⚠️ Bệnh chính: ${dom.vi}`;
  v.querySelector(".vs").textContent =
    `${dom.count} vết, ${dom.pct_leaf != null ? fmt(dom.pct_leaf, 2) + "% diện tích lá" : fmt(dom.pct_frame, 2) + "% khung hình"}, conf cao nhất ${fmt(dom.max_conf, 2)}` +
    (sev ? ` · Mức độ (tổng ${fmt(res.lesion_pct_leaf, 1)}% lá): ${sev[1]}` : "") +
    (others.length ? ` · Kèm theo: ${others.join(", ")}` : "");
  v.style.borderLeftColor = dom.color;
}

function renderDetList(res) {
  const box = $("detList");
  box.innerHTML = "";
  $("detCount").textContent = res ? `(${res.total})` : "";
  const order = res.detections.map((d, i) => [d, i]).sort((a, b) => b[0].conf - a[0].conf);
  for (const [d, i] of order.slice(0, 300)) {
    const c = cls(d.name);
    const row = document.createElement("div");
    row.className = "det";
    const pct = (100 * d.area_px) / (res.width * res.height);
    row.innerHTML = `<span class="n">#${i + 1}</span>
      <span><span class="dot" style="background:${c.color}"></span>${c.vi}</span>
      <span>${fmt(pct, 2)}%</span><b>${fmt(d.conf * 100, 0)}%</b>`;
    row.onmouseenter = () => { S.highlight = i; redraw(); };
    row.onmouseleave = () => { S.highlight = -1; redraw(); };
    box.appendChild(row);
  }
}

// ---------- biểu đồ diễn biến ----------
function pushSpark(res) {
  S.spark.push(S.classes.map((c) => res.summary.find((s) => s.name === c.name)?.count || 0));
  if (S.spark.length > 60) S.spark.shift();
  drawSpark();
}

function drawSpark() {
  const cv = $("spark");
  const w = (cv.width = cv.clientWidth * devicePixelRatio);
  const h = (cv.height = 110 * devicePixelRatio);
  const g = cv.getContext("2d");
  g.clearRect(0, 0, w, h);
  if (!S.spark.length) return;
  const max = Math.max(1, ...S.spark.flat());
  const pad = 6 * devicePixelRatio;
  g.strokeStyle = getComputedStyle(document.body).getPropertyValue("--line");
  g.lineWidth = 1;
  g.beginPath(); g.moveTo(0, h - pad); g.lineTo(w, h - pad); g.stroke();
  S.classes.forEach((c, ci) => {
    g.beginPath();
    g.strokeStyle = c.color;
    g.lineWidth = 2 * devicePixelRatio;
    S.spark.forEach((row, i) => {
      const x = S.spark.length > 1 ? (i / (S.spark.length - 1)) * w : w / 2;
      const y = h - pad - (row[ci] / max) * (h - 2 * pad);
      i ? g.lineTo(x, y) : g.moveTo(x, y);
    });
    g.stroke();
  });
  g.fillStyle = getComputedStyle(document.body).getPropertyValue("--muted");
  g.font = `${11 * devicePixelRatio}px system-ui`;
  g.fillText(`max ${max}`, 4, 12 * devicePixelRatio);
}

// ---------- nhật ký, lưu, CSV ----------
function addLog(source, res) {
  const row = {
    time: new Date().toLocaleTimeString("vi-VN"),
    source,
    total: res.total,
    counts: Object.fromEntries(res.summary.map((s) => [s.name, s.count])),
    pct_frame: Object.fromEntries(res.summary.map((s) => [s.name, s.pct_frame])),
    pct_leaf: Object.fromEntries(res.summary.map((s) => [s.name, s.pct_leaf])),
    lesion_pct_leaf: res.lesion_pct_leaf,
    severity: severity(res.lesion_pct_leaf)?.[1] || "",
    dominant: dominant(res)?.name || "",
    infer_ms: res.infer_ms,
    imgsz: res.imgsz,
  };
  S.log.unshift(row);
  if (S.log.length > 2000) S.log.pop();
  renderLog();
}

function renderLog() {
  const box = $("log");
  $("logCount").textContent = `(${S.log.length})`;
  box.innerHTML = S.log.slice(0, 200).map((r) => {
    const parts = S.classes.filter((c) => r.counts[c.name]).map((c) =>
      `<span class="dot" style="background:${c.color}"></span>${r.counts[c.name]}`).join(" ");
    return `<div class="log-row"><b>${r.time}</b> · ${escapeHtml(r.source)} · ${r.total} vết ${parts}${r.severity ? " · " + r.severity : ""}</div>`;
  }).join("");
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

$("clearLog").onclick = () => { S.log = []; renderLog(); };

$("snap").onclick = () => {
  if (!S.lastRes) return;
  view.toBlob((b) => {
    const a = document.createElement("a");
    a.href = URL.createObjectURL(b);
    const base = S.mode === "image" && S.images[S.current] ? S.images[S.current].name.replace(/\.[^.]+$/, "") : "camera";
    a.download = `ket_qua_${base}_${Date.now()}.png`;
    a.click();
  }, "image/png");
};

$("csv").onclick = () => {
  if (!S.log.length) return alert("Nhật ký đang trống");
  const names = S.classes.map((c) => c.name);
  const head = ["thoi_gian", "nguon", "tong_vet", ...names.map((n) => `so_${n}`), ...names.map((n) => `pct_khung_${n}`),
    ...names.map((n) => `pct_la_${n}`), "pct_la_tong", "muc_do", "benh_chinh", "infer_ms", "imgsz"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = [head.join(",")].concat([...S.log].reverse().map((r) => [
    q(r.time), q(r.source), r.total, ...names.map((n) => r.counts[n]), ...names.map((n) => r.pct_frame[n]),
    ...names.map((n) => r.pct_leaf[n] ?? ""), r.lesion_pct_leaf ?? "", q(r.severity), q(r.dominant), r.infer_ms, r.imgsz,
  ].join(",")));
  const blob = new Blob(["﻿" + lines.join("\n")], { type: "text/csv;charset=utf-8" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `nhat_ky_quet_benh_${Date.now()}.csv`;
  a.click();
};

window.addEventListener("resize", drawSpark);
loadInfo().then(listCameras).catch((e) => {
  $("deviceBadge").textContent = "Lỗi kết nối máy chủ";
  console.error(e);
});
