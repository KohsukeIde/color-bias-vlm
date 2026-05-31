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

  // Default view: a big curve of the word's OWN axis (e.g. cold↔warm) vs. ink
  // hue. Fills the panel; the y-axis poles make "which way is which" explicit.
  function renderAxisView() {
    const svg = d3.select("#planeSvg"); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, H = 300, m = { t: 54, r: 16, b: 30, l: 30 };
    const key = primaryAxis(state.word);
    const info = axisInfo[key] || { pos: key, neg: "not-" + key };
    const rs = rows();
    const vals = rs.map((d) => d.proj[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), p = (hi - lo) * 0.18 || 0.02;
    const x = d3.scaleLinear().domain([0, 360]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([lo - p, hi + p]).range([H - m.b, m.t]);
    const cur = rowAt(state.hue);
    const red = rs.reduce((a, b) => Math.abs(b.hue) < Math.abs(a.hue) ? b : a);
    const sgn = (n) => (n >= 0 ? "+" : "");

    document.getElementById("planeLabel").innerHTML = `Antonym axis &middot; ${info.neg} ↔ ${info.pos}`;

    // top readout: lean toward whichever pole the sign points to, + Δ vs red
    const pole = cur.proj[key] >= 0 ? info.pos : info.neg;
    svg.append("text").attr("x", W / 2).attr("y", 22).attr("text-anchor", "middle")
      .attr("class", "ct").attr("fill", P.strong).attr("font-size", "14px").attr("font-weight", 700)
      .text(`“${state.word}” leans ${Math.abs(cur.proj[key]).toFixed(3)} toward ${pole}`);
    svg.append("text").attr("x", W / 2).attr("y", 41).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis)
      .text(`${sgn(cur.proj[key] - red.proj[key])}${(cur.proj[key] - red.proj[key]).toFixed(3)} toward ${info.pos} vs. red ink (0°)`);

    // frame + zero line
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("line").attr("x1", m.l).attr("x2", m.l).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", P.grid);
    if (lo - p < 0 && hi + p > 0)
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
        .attr("stroke", P.grid).attr("stroke-dasharray", "3 3");
    // y-axis poles (top = pos, bottom = neg) — makes the vertical meaning explicit
    svg.append("text").attr("x", m.l + 4).attr("y", m.t + 4).attr("class", "ax").attr("fill", P.strong)
      .attr("font-weight", 700).text("↑ " + info.pos);
    svg.append("text").attr("x", m.l + 4).attr("y", H - m.b - 5).attr("class", "ax").attr("fill", P.strong)
      .attr("font-weight", 700).text("↓ " + info.neg);
    // x ticks (ink hue)
    [0, 90, 180, 270, 360].forEach((hv) => svg.append("text").attr("x", x(hv)).attr("y", H - m.b + 12)
      .attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).attr("font-size", "8px").text(hv));
    svg.append("text").attr("x", (W + m.l - m.r) / 2).attr("y", H - 3).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis).text("ink hue (°)");
    // hue-colored line + dots
    const line = d3.line().x((d) => x(d.hue)).y((d) => y(d.proj[key]));
    svg.append("path").datum(rs).attr("d", line).attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1.6);
    svg.selectAll("circle.pt").data(rs).enter().append("circle")
      .attr("cx", (d) => x(d.hue)).attr("cy", (d) => y(d.proj[key])).attr("r", 2.6)
      .attr("fill", (d) => colorForHue(d.hue)).attr("stroke", P.ring).attr("stroke-width", .5);
    // cursor + current marker
    svg.append("line").attr("x1", x(cur.hue)).attr("x2", x(cur.hue)).attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", P.strong).attr("stroke-width", 1).attr("opacity", .4);
    svg.append("circle").attr("cx", x(cur.hue)).attr("cy", y(cur.proj[key])).attr("r", 8)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
  }

  // Secondary view: the raw 768-d embedding reduced to 2D (axes not meaningful)
  function renderPca() {
    const svg = d3.select("#planeSvg"); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, H = 300, m = { t: 26, r: 16, b: 28, l: 28 };
    const rs = rows();
    const pad = (a) => { const lo = Math.min(...a), hi = Math.max(...a), p = (hi - lo) * 0.07 || 0.05; return [lo - p, hi + p]; };
    const x = d3.scaleLinear().domain(pad(rs.map((d) => d.wx))).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain(pad(rs.map((d) => d.wy))).range([H - m.b, m.t]);
    document.getElementById("planeLabel").innerHTML = "Raw embedding &middot; within-word PCA(2)";
    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("line").attr("x1", m.l).attr("x2", m.l).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("text").attr("x", (W + m.l - m.r) / 2).attr("y", H - 6).attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).text("PC1 →");
    svg.append("text").attr("transform", `translate(11,${(H - m.b + m.t) / 2}) rotate(-90)`).attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).text("PC2 →");
    const line = d3.line().x((d) => x(d.wx)).y((d) => y(d.wy));
    svg.append("path").datum(rs).attr("d", line).attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1.4);
    svg.selectAll("circle.pt").data(rs).enter().append("circle")
      .attr("cx", (d) => x(d.wx)).attr("cy", (d) => y(d.wy)).attr("r", 3)
      .attr("fill", (d) => colorForHue(d.hue)).attr("stroke", P.ring).attr("stroke-width", .6);
    const cur = rowAt(state.hue);
    svg.append("circle").attr("cx", x(cur.wx)).attr("cy", y(cur.wy)).attr("r", 8)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
  }

  // ---------------- right panel (stacked, full-width), view-aware ----------------
  // axis view -> the OTHER semantic axes; pca view -> the loop unrolled (PC1,PC2 vs hue)
  function renderCurves() {
    const svg = d3.select("#curvesSvg"); svg.selectAll("*").remove();
    const P = pal();
    const rs = rows(), cur = rowAt(state.hue);
    let series, label;
    if (state.view === "pca") {
      label = "The loop, unrolled: each PCA coordinate vs. ink hue";
      series = [
        { name: "PC1", val: (d) => d.wx, pos: "", neg: "" },
        { name: "PC2", val: (d) => d.wy, pos: "", neg: "" },
      ];
    } else {
      label = "The other semantic axes vs. ink hue";
      const pk = primaryAxis(state.word);
      series = CURVE_AXES.filter((a) => a !== pk).map((a) => {
        const info = axisInfo[a] || { pos: a, neg: "" };
        return { name: axisLabel[a] || a, val: (d) => d.proj[a], pos: info.pos, neg: info.neg };
      });
    }
    const cl = document.getElementById("curvesLabel"); if (cl) cl.textContent = label;

    const W = 360, H = 300, top = 14, mL = 60, mR = 12, rowGap = 18, bottom = 24;
    const n = series.length;
    const rh = (H - top - rowGap * (n - 1) - bottom) / n;
    series.forEach((s, i) => {
      const oy = top + i * (rh + rowGap);
      const vals = rs.map(s.val);
      const lo = Math.min(...vals), hi = Math.max(...vals), p = (hi - lo) * 0.18 || 0.02;
      const x = d3.scaleLinear().domain([0, 360]).range([mL, W - mR]);
      const y = d3.scaleLinear().domain([lo - p, hi + p]).range([oy + rh, oy]);
      svg.append("rect").attr("x", mL).attr("y", oy).attr("width", W - mR - mL).attr("height", rh)
        .attr("fill", "none").attr("stroke", P.grid);
      if (lo - p < 0 && hi + p > 0)
        svg.append("line").attr("x1", mL).attr("x2", W - mR).attr("y1", y(0)).attr("y2", y(0))
          .attr("stroke", P.grid).attr("stroke-dasharray", "3 3");
      // y-axis poles (semantic view only)
      if (s.pos) {
        svg.append("text").attr("x", mL - 6).attr("y", oy + 8).attr("text-anchor", "end")
          .attr("class", "ax").attr("fill", P.axis).attr("font-size", "8.5px").text(s.pos);
        svg.append("text").attr("x", mL - 6).attr("y", oy + rh - 1).attr("text-anchor", "end")
          .attr("class", "ax").attr("fill", P.axis).attr("font-size", "8.5px").text(s.neg);
      }
      // name + live value
      const v = s.val(cur);
      svg.append("text").attr("x", mL + 4).attr("y", oy - 3).attr("class", "ct").attr("fill", P.axis)
        .html(`${s.name}: <tspan class="val">${v >= 0 ? "+" : ""}${v.toFixed(3)}</tspan>`);
      // line + hue dots
      const line = d3.line().x((d) => x(d.hue)).y((d) => y(s.val(d)));
      svg.append("path").datum(rs).attr("d", line).attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1.2);
      svg.selectAll("c" + i).data(rs).enter().append("circle")
        .attr("cx", (d) => x(d.hue)).attr("cy", (d) => y(s.val(d))).attr("r", 1.6)
        .attr("fill", (d) => colorForHue(d.hue));
      // cursor + marker
      svg.append("line").attr("x1", x(cur.hue)).attr("x2", x(cur.hue)).attr("y1", oy).attr("y2", oy + rh)
        .attr("stroke", P.strong).attr("stroke-width", 1).attr("opacity", .4);
      svg.append("circle").attr("cx", x(cur.hue)).attr("cy", y(s.val(cur))).attr("r", 4)
        .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 1.4);
      // x ticks on the last row only
      if (i === n - 1) {
        [0, 90, 180, 270, 360].forEach((hv) => svg.append("text").attr("x", x(hv)).attr("y", oy + rh + 11)
          .attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).attr("font-size", "8px").text(hv));
        svg.append("text").attr("x", (mL + W - mR) / 2).attr("y", oy + rh + 21).attr("text-anchor", "middle")
          .attr("class", "ax").attr("fill", P.axis).text("ink hue (°)");
      }
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
        b.classList.add("is-active"); renderPlane(); renderCurves();
      });
    });

    document.getElementById("vizNote").innerHTML =
      `<b>The takeaway:</b> the curves bend <i>smoothly and systematically</i> with hue — a purely visual ` +
      `property (ink colour) is mapped onto <i>meaning</i> axes the text never invoked, and that entanglement ` +
      `is what lets recolouring a word steer a VLM's judgement. The <b>Raw embedding (PCA)</b> loop is closed ` +
      `(hue 0° rejoins 360°) and is a property of the whole 768-d embedding; it isn't a clean circle because ` +
      `the real path is ~4–6-dimensional (you are seeing a flattened 2-D shadow) and hue itself is ` +
      `perceptually non-uniform. <span class="prov">Probe: CLIP ViT-L/14-336, single-word renders, 72 hues.</span>`;

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
