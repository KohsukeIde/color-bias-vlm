/* Interactive hue-sweep visualization for color-bias-vlm.
   Reads ./static/data/hue_data.json (real CLIP-ViT-L/14-336 measurements). */
(function () {
  "use strict";
  const DATA_URL = "./static/data/hue_data.json";
  const CURVE_AXES = ["valence", "emotion", "safety", "temperature"];
  // each word's "own" bipolar axis (its meaning vs. its opposite)
  const PRIMARY = { warm: "temperature", cold: "temperature", safe: "safety", dangerous: "safety" };
  const primaryAxis = (w) => PRIMARY[w] || "valence";

  const state = { word: "warm", hue: 0, view: "axis" };
  let DATA, hueHex = {}, axisLabel = {}, axisInfo = {};

  // theme-aware neutral palette read from CSS variables
  function pal() {
    const cs = getComputedStyle(document.documentElement);
    const v = (n, d) => (cs.getPropertyValue(n).trim() || d);
    return {
      line: v("--viz-line", "#c9cedb"), grid: v("--viz-grid", "#e3e6ef"),
      axis: v("--viz-axis", "#7a8090"), strong: v("--viz-strong", "#1d1d28"),
      ring: v("--viz-ring", "#ffffff"),
    };
  }

  function hsv2hex(h, s, v) {
    const c = v * s, x = c * (1 - Math.abs(((h / 60) % 2) - 1)), m = v - c;
    let r = 0, g = 0, b = 0;
    if (h < 60) [r, g, b] = [c, x, 0]; else if (h < 120) [r, g, b] = [x, c, 0];
    else if (h < 180) [r, g, b] = [0, c, x]; else if (h < 240) [r, g, b] = [0, x, c];
    else if (h < 300) [r, g, b] = [x, 0, c]; else [r, g, b] = [c, 0, x];
    const to = (n) => Math.round((n + m) * 255).toString(16).padStart(2, "0");
    return "#" + to(r) + to(g) + to(b);
  }
  const colorForHue = (h) => hueHex[h] || hsv2hex(h, 1.0, 0.9);
  const rows = () => DATA.data[state.word];
  const rowAt = (h) => rows().reduce((a, b) => Math.abs(b.hue - h) < Math.abs(a.hue - h) ? b : a);

  // ---------------- stimulus ----------------
  function renderStimulus() {
    const hex = colorForHue(state.hue);
    const w = document.getElementById("stimulusWord");
    w.textContent = state.word; w.style.color = hex;
    document.getElementById("swatch").style.background = hex;
    document.getElementById("swatchHex").textContent = hex + "  ·  " + state.hue + "°";
    document.getElementById("hueVal").textContent = state.hue + "°";
  }

  // ---------------- embedding panel ----------------
  function renderPlane() { return state.view === "pca" ? renderPca() : renderAxisView(); }

  // Default view: where the encoder places the word between its OWN opposites
  // (e.g. for "warm", how far toward warm vs. cold). One interpretable axis.
  function renderAxisView() {
    const svg = d3.select("#planeSvg"); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, cy = 168, cx0 = 52, cx1 = W - 52;
    const key = primaryAxis(state.word);
    const info = axisInfo[key] || { pos: key, neg: "not-" + key };
    const rs = rows();
    const vals = rs.map((d) => d.proj[key]);
    const M = Math.max(...vals.map((v) => Math.abs(v))) * 1.18 || 0.1;
    const x = d3.scaleLinear().domain([-M, M]).range([cx0, cx1]);
    const cur = rowAt(state.hue);
    const red = rs.reduce((a, b) => Math.abs(b.hue) < Math.abs(a.hue) ? b : a);
    const sgn = (n) => (n >= 0 ? "+" : "");

    document.getElementById("planeLabel").innerHTML = `Antonym axis &middot; ${info.neg} ↔ ${info.pos}`;

    // top readout (lean toward whichever pole the sign points to)
    const pole = cur.proj[key] >= 0 ? info.pos : info.neg;
    svg.append("text").attr("x", W / 2).attr("y", 40).attr("text-anchor", "middle")
      .attr("class", "ct").attr("fill", P.strong).attr("font-size", "14px").attr("font-weight", 700)
      .text(`“${state.word}” leans ${Math.abs(cur.proj[key]).toFixed(3)} toward ${pole}`);
    svg.append("text").attr("x", W / 2).attr("y", 60).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis)
      .text(`${sgn(cur.proj[key] - red.proj[key])}${(cur.proj[key] - red.proj[key]).toFixed(3)} toward ${info.pos} vs. red ink (0°)`);

    // range bracket over all hues
    const lo = Math.min(...vals), hi = Math.max(...vals);
    svg.append("line").attr("x1", x(lo)).attr("x2", x(hi)).attr("y1", cy - 34).attr("y2", cy - 34)
      .attr("stroke", P.line).attr("stroke-width", 3).attr("stroke-linecap", "round");
    svg.append("text").attr("x", (x(lo) + x(hi)) / 2).attr("y", cy - 42).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis).text("range over all 72 hues");

    // main axis line + poles + neutral tick
    svg.append("line").attr("x1", cx0).attr("x2", cx1).attr("y1", cy).attr("y2", cy)
      .attr("stroke", P.axis).attr("stroke-width", 1.5);
    svg.append("text").attr("x", cx0 - 6).attr("y", cy + 4).attr("text-anchor", "end")
      .attr("class", "ax").attr("fill", P.strong).attr("font-weight", 700).text(info.neg);
    svg.append("text").attr("x", cx1 + 6).attr("y", cy + 4).attr("text-anchor", "start")
      .attr("class", "ax").attr("fill", P.strong).attr("font-weight", 700).text(info.pos);
    svg.append("line").attr("x1", x(0)).attr("x2", x(0)).attr("y1", cy - 9).attr("y2", cy + 9).attr("stroke", P.grid);
    svg.append("text").attr("x", x(0)).attr("y", cy + 28).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis).text("0 (neutral)");

    // hue-colored rug: every hue's position on the axis
    svg.selectAll("line.rug").data(rs).enter().append("line").attr("class", "rug")
      .attr("x1", (d) => x(d.proj[key])).attr("x2", (d) => x(d.proj[key]))
      .attr("y1", cy + 4).attr("y2", cy + 13)
      .attr("stroke", (d) => colorForHue(d.hue)).attr("stroke-width", 2).attr("opacity", .85);

    // current marker
    const xc = x(cur.proj[key]);
    svg.append("line").attr("x1", xc).attr("x2", xc).attr("y1", cy - 20).attr("y2", cy + 20)
      .attr("stroke", P.strong).attr("stroke-width", 1).attr("opacity", .45);
    svg.append("circle").attr("cx", xc).attr("cy", cy).attr("r", 9)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
  }

  // Secondary view: the raw 768-d embedding reduced to 2D (axes not meaningful)
  function renderPca() {
    const svg = d3.select("#planeSvg"); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, H = 300, m = { t: 26, r: 16, b: 28, l: 28 };
    const rs = rows();
    const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a), p = (hi - lo) * 0.18 || 0.05; return [lo - p, hi + p]; };
    const x = d3.scaleLinear().domain(pad(rs.map((d) => d.wx))).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain(pad(rs.map((d) => d.wy))).range([H - m.b, m.t]);
    document.getElementById("planeLabel").innerHTML = "Raw embedding &middot; within-word PCA(2)";
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("line").attr("x1", m.l).attr("x2", m.l).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("text").attr("x", (W + m.l - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).text("PC1 →");
    svg.append("text").attr("transform", `translate(11,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).text("PC2 →");
    svg.append("text").attr("x", W - m.r).attr("y", 14).attr("text-anchor", "end").attr("class", "ax").attr("fill", P.axis)
      .text("axes aren't individually meaningful — watch the motion");
    const line = d3.line().x((d) => x(d.wx)).y((d) => y(d.wy));
    svg.append("path").datum(rs).attr("d", line).attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1.4);
    svg.selectAll("circle.pt").data(rs).enter().append("circle")
      .attr("cx", (d) => x(d.wx)).attr("cy", (d) => y(d.wy)).attr("r", 3)
      .attr("fill", (d) => colorForHue(d.hue)).attr("stroke", P.ring).attr("stroke-width", .6);
    const cur = rowAt(state.hue);
    svg.append("circle").attr("cx", x(cur.wx)).attr("cy", y(cur.wy)).attr("r", 8)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
  }

  // ---------------- tuning curves (2x2) ----------------
  function renderCurves() {
    const svg = d3.select("#curvesSvg"); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, H = 300, cols = 2, rowsN = 2, gx = 12, gy = 26;
    const cw = (W - gx * (cols + 1)) / cols, ch = (H - gy * (rowsN + 1)) / rowsN;
    const rs = rows();
    // show the word's own axis first
    const pk = primaryAxis(state.word);
    const order = [pk].concat(CURVE_AXES.filter((a) => a !== pk));
    order.forEach((axis, i) => {
      const cxi = i % cols, cyi = Math.floor(i / cols);
      const ox = gx + cxi * (cw + gx), oy = gy + cyi * (ch + gy);
      const info = axisInfo[axis] || { pos: axis, neg: "" };
      const vals = rs.map((d) => d.proj[axis]);
      const lo = Math.min(...vals), hi = Math.max(...vals), p = (hi - lo) * 0.15 || 0.02;
      const x = d3.scaleLinear().domain([0, 360]).range([ox, ox + cw]);
      const y = d3.scaleLinear().domain([lo - p, hi + p]).range([oy + ch, oy]);
      const cell = svg.append("g");
      if (lo - p < 0 && hi + p > 0)
        cell.append("line").attr("x1", ox).attr("x2", ox + cw).attr("y1", y(0)).attr("y2", y(0))
          .attr("stroke", P.grid).attr("stroke-dasharray", "3 3");
      cell.append("rect").attr("x", ox).attr("y", oy).attr("width", cw).attr("height", ch)
        .attr("fill", "none").attr("stroke", P.grid);
      // title: poles + live readout (the poles ARE the meaning of the y-axis)
      const cur = rowAt(state.hue);
      cell.append("text").attr("x", ox).attr("y", oy - 8).attr("class", "ct").attr("fill", P.axis)
        .html(`${info.neg}→${info.pos}: <tspan class="val">${cur.proj[axis] >= 0 ? "+" : ""}${cur.proj[axis].toFixed(3)}</tspan>`);
      // x ticks (ink hue)
      [0, 360].forEach((hv) => cell.append("text").attr("x", x(hv)).attr("y", oy + ch + 9)
        .attr("text-anchor", hv === 0 ? "start" : "end").attr("class", "ax").attr("fill", P.axis)
        .attr("font-size", "8px").text(hv + "°"));
      // line + hue dots
      const line = d3.line().x((d) => x(d.hue)).y((d) => y(d.proj[axis]));
      cell.append("path").datum(rs).attr("d", line).attr("fill", "none")
        .attr("stroke", P.line).attr("stroke-width", 1.2);
      cell.selectAll("circle.d" + i).data(rs).enter().append("circle")
        .attr("cx", (d) => x(d.hue)).attr("cy", (d) => y(d.proj[axis])).attr("r", 1.8)
        .attr("fill", (d) => colorForHue(d.hue));
      // cursor
      cell.append("line").attr("x1", x(state.hue)).attr("x2", x(state.hue)).attr("y1", oy).attr("y2", oy + ch)
        .attr("stroke", P.strong).attr("stroke-width", 1).attr("opacity", .5);
      cell.append("circle").attr("cx", x(cur.hue)).attr("cy", y(cur.proj[axis])).attr("r", 5)
        .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 1.6);
    });
  }

  function renderAll() { renderStimulus(); renderPlane(); renderCurves(); }

  // ---------------- init ----------------
  function init(data) {
    DATA = data;
    (data.hueColors || []).forEach((d) => (hueHex[d.hue] = d.hex));
    (data.axes || []).forEach((a) => {
      axisLabel[a.key] = a.label;
      axisInfo[a.key] = { pos: a.pos, neg: a.neg, label: a.label };
    });

    const sel = document.getElementById("wordSelect");
    const sentiment = new Set(data.meta.sentimentWords || []);
    data.meta.words.forEach((w) => {
      const o = document.createElement("option");
      o.value = w; o.textContent = sentiment.has(w) ? w + "  (sentiment)" : w;
      sel.appendChild(o);
    });
    sel.value = state.word;
    sel.addEventListener("change", (e) => { state.word = e.target.value; renderAll(); });

    document.getElementById("hueSlider").addEventListener("input", (e) => {
      state.hue = +e.target.value; renderAll();
    });
    document.querySelectorAll("#viewToggle button").forEach((b) => {
      b.addEventListener("click", () => {
        state.view = b.dataset.view;
        document.querySelectorAll("#viewToggle button").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active"); renderPlane();
      });
    });

    document.getElementById("vizNote").innerHTML =
      `<b>Antonym axis</b>: the word's projection onto its own opposite (e.g. warm vs. cold), measured as a ` +
      `CLIP cosine direction — higher = the encoder reads it more strongly as that meaning. ` +
      `<b>Raw embedding (PCA)</b>: the full 768-d CLIP image embedding reduced to 2D; the axes aren't ` +
      `individually interpretable — what matters is that the point moves as hue changes. ` +
      `Small charts: x = ink hue (0–360°), y = projection onto each axis. ` +
      `Probe rendering: ${data.meta.rendering}.`;

    renderAll();
  }

  // SVG text styling injected (kept with the viz)
  const css = document.createElement("style");
  css.textContent = "#planeSvg .ax,#curvesSvg .ct{font-family:Inter,sans-serif;font-size:10px}" +
    "#curvesSvg .ct .val{fill:var(--viz-strong);font-weight:700}";
  document.head.appendChild(css);

  // re-render the viz when the page theme changes
  window.addEventListener("themechange", function () { if (DATA) renderAll(); });

  d3.json(DATA_URL).then(init).catch((e) => {
    document.getElementById("vizNote").textContent = "Could not load visualization data (" + e + ").";
    console.error(e);
  });
})();
