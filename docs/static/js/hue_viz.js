/* Interactive hue-sweep visualization for color-bias-vlm.
   Reads ./static/data/hue_data.json (real CLIP-ViT-L/14-336 measurements). */
(function () {
  "use strict";
  const DATA_URL = "./static/data/hue_data.json";
  const PLANE_AXES = ["valence", "emotion"];        // semantic plane x,y
  const CURVE_AXES = ["valence", "emotion", "safety", "temperature"];

  const state = { word: "terrible", hue: 0, view: "semantic" };
  let DATA, hueHex = {}, axisLabel = {};

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

  // ---------------- embedding plane ----------------
  function renderPlane() {
    const svg = d3.select("#planeSvg"); svg.selectAll("*").remove();
    const W = 360, H = 300, m = { t: 22, r: 18, b: 34, l: 40 };
    const semantic = state.view === "semantic";
    const xKey = semantic ? PLANE_AXES[0] : "wx";
    const yKey = semantic ? PLANE_AXES[1] : "wy";
    const getX = (d) => semantic ? d.proj[xKey] : d[xKey];
    const getY = (d) => semantic ? d.proj[yKey] : d[yKey];
    const rs = rows();
    const xs = rs.map(getX), ys = rs.map(getY);
    const pad = (arr) => { const lo = Math.min(...arr), hi = Math.max(...arr), p = (hi - lo) * 0.18 || 0.05; return [lo - p, hi + p]; };
    const x = d3.scaleLinear().domain(pad(xs)).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain(pad(ys)).range([H - m.b, m.t]);

    document.getElementById("planeLabel").innerHTML = semantic
      ? "Semantic plane &middot; valence × emotion"
      : "CLIP embedding &middot; within-word PCA(2)";

    // axes
    const g = svg.append("g");
    g.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b)
      .attr("stroke", "#d7dae4");
    g.append("line").attr("x1", m.l).attr("x2", m.l).attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", "#d7dae4");
    g.append("text").attr("x", (W + m.l - m.r) / 2).attr("y", H - 8).attr("text-anchor", "middle")
      .attr("class", "ax").text(semantic ? (axisLabel.valence || "valence") + " →" : "PC1 →");
    g.append("text").attr("transform", `translate(12,${(H - m.b + m.t) / 2}) rotate(-90)`)
      .attr("text-anchor", "middle").attr("class", "ax")
      .text(semantic ? (axisLabel.emotion || "emotion") + " →" : "PC2 →");

    // trajectory path (hue order)
    const line = d3.line().x((d) => x(getX(d))).y((d) => y(getY(d)));
    svg.append("path").datum(rs).attr("d", line).attr("fill", "none")
      .attr("stroke", "#c9cedb").attr("stroke-width", 1.4);
    // hue-colored dots
    svg.selectAll("circle.pt").data(rs).enter().append("circle").attr("class", "pt")
      .attr("cx", (d) => x(getX(d))).attr("cy", (d) => y(getY(d))).attr("r", 3)
      .attr("fill", (d) => colorForHue(d.hue)).attr("stroke", "#fff").attr("stroke-width", .6);
    // current point
    const cur = rowAt(state.hue);
    svg.append("circle").attr("cx", x(getX(cur))).attr("cy", y(getY(cur))).attr("r", 8)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", "#1d1d28").attr("stroke-width", 2);
  }

  // ---------------- tuning curves (2x2) ----------------
  function renderCurves() {
    const svg = d3.select("#curvesSvg"); svg.selectAll("*").remove();
    const W = 360, H = 300, cols = 2, rowsN = 2, gx = 12, gy = 26;
    const cw = (W - gx * (cols + 1)) / cols, ch = (H - gy * (rowsN + 1)) / rowsN;
    const rs = rows();
    CURVE_AXES.forEach((axis, i) => {
      const cxi = i % cols, cyi = Math.floor(i / cols);
      const ox = gx + cxi * (cw + gx), oy = gy + cyi * (ch + gy);
      const vals = rs.map((d) => d.proj[axis]);
      const lo = Math.min(...vals), hi = Math.max(...vals), p = (hi - lo) * 0.15 || 0.02;
      const x = d3.scaleLinear().domain([0, 360]).range([ox, ox + cw]);
      const y = d3.scaleLinear().domain([lo - p, hi + p]).range([oy + ch, oy]);
      const cell = svg.append("g");
      // zero line
      if (lo - p < 0 && hi + p > 0)
        cell.append("line").attr("x1", ox).attr("x2", ox + cw).attr("y1", y(0)).attr("y2", y(0))
          .attr("stroke", "#eceef4").attr("stroke-dasharray", "3 3");
      cell.append("rect").attr("x", ox).attr("y", oy).attr("width", cw).attr("height", ch)
        .attr("fill", "none").attr("stroke", "#e3e6ef");
      // title + live readout
      const cur = rowAt(state.hue);
      cell.append("text").attr("x", ox).attr("y", oy - 8).attr("class", "ct")
        .html(`${axisLabel[axis] || axis}: <tspan class="val">${cur.proj[axis] >= 0 ? "+" : ""}${cur.proj[axis].toFixed(3)}</tspan>`);
      // line
      const line = d3.line().x((d) => x(d.hue)).y((d) => y(d.proj[axis]));
      cell.append("path").datum(rs).attr("d", line).attr("fill", "none")
        .attr("stroke", "#c9cedb").attr("stroke-width", 1.2);
      cell.selectAll("circle.d" + i).data(rs).enter().append("circle")
        .attr("cx", (d) => x(d.hue)).attr("cy", (d) => y(d.proj[axis])).attr("r", 1.8)
        .attr("fill", (d) => colorForHue(d.hue));
      // cursor
      cell.append("line").attr("x1", x(state.hue)).attr("x2", x(state.hue)).attr("y1", oy).attr("y2", oy + ch)
        .attr("stroke", "#1d1d28").attr("stroke-width", 1).attr("opacity", .5);
      cell.append("circle").attr("cx", x(cur.hue)).attr("cy", y(cur.proj[axis])).attr("r", 5)
        .attr("fill", colorForHue(cur.hue)).attr("stroke", "#1d1d28").attr("stroke-width", 1.6);
    });
  }

  function renderAll() { renderStimulus(); renderPlane(); renderCurves(); }

  // ---------------- init ----------------
  function init(data) {
    DATA = data;
    (data.hueColors || []).forEach((d) => (hueHex[d.hue] = d.hex));
    (data.axes || []).forEach((a) => (axisLabel[a.key] = a.label));

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

    const v = data.meta.globalPcaVar;
    document.getElementById("vizNote").innerHTML =
      `Probe rendering: ${data.meta.rendering}. Semantic plane axes are CLIP text-embedding directions ` +
      `(good–bad, happy–sad). "CLIP PCA" projects the raw 768-d image embeddings of one word across hue ` +
      `onto its first two principal components.`;

    renderAll();
  }

  // SVG text styling injected (kept with the viz)
  const css = document.createElement("style");
  css.textContent = "#planeSvg .ax,#curvesSvg .ct{font-family:Inter,sans-serif;font-size:10px;fill:#7a8090}" +
    "#curvesSvg .ct .val{fill:#1d1d28;font-weight:700}";
  document.head.appendChild(css);

  d3.json(DATA_URL).then(init).catch((e) => {
    document.getElementById("vizNote").textContent = "Could not load visualization data (" + e + ").";
    console.error(e);
  });
})();
