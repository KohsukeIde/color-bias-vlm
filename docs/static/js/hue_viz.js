/* Interactive hue-sweep visualization for color-bias-vlm.
   Reads ./static/data/hue_data.json (real CLIP-ViT-L/14-336 measurements). */
(function () {
  "use strict";
  const DATA_URL = "./static/data/hue_data.json?v=5";  // bump when hue_data.json changes
  const CURVE_AXES = ["valence", "emotion", "safety", "temperature"];
  // each word's "own" bipolar axis (its meaning vs. its opposite)
  const PRIMARY = { warm: "temperature", cold: "temperature", safe: "safety", dangerous: "safety" };
  const primaryAxis = (w) => PRIMARY[w] || "valence";

  const state = { word: "warm", hue: 0, view: "axis" };
  const rot = { x: -0.45, y: 0.7 };   // 3D rotation for the raw-embedding view
  let DATA, hueHex = {}, axisLabel = {}, axisInfo = {}, COLOR_ONLY = "(color only)";
  const mean = (a) => a.reduce((s, v) => s + v, 0) / a.length;
  const dispName = (w) => (w === COLOR_ONLY ? "pure color" : w);

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
    const canvas = document.querySelector(".stimulus-canvas");
    if (state.word === COLOR_ONLY) {           // pure color: show a filled swatch, no text
      w.textContent = ""; if (canvas) canvas.style.background = hex;
    } else {
      w.textContent = state.word; w.style.color = hex; if (canvas) canvas.style.background = "#fff";
    }
    document.getElementById("swatch").style.background = hex;
    document.getElementById("swatchHex").textContent = hex + "  ·  " + state.hue + "°";
    document.getElementById("hueVal").textContent = state.hue + "°";
  }

  // ============ panels ============
  // Middle = the selected word; Right = pure color (no word), SAME view, so
  // you compare "word in color" vs "color alone" side by side.

  // Antonym-axis big curve of `rs` on bipolar axis `key`, titled by `name`.
  function renderAxis(svgSel, rs, key, name, labelId, labelHtml) {
    const svg = d3.select(svgSel); svg.selectAll("*").remove();
    svg.style("cursor", "default");
    const P = pal();
    const W = 360, H = 300, m = { t: 54, r: 16, b: 30, l: 30 };
    const info = axisInfo[key] || { pos: key, neg: "not-" + key };
    const vals = rs.map((d) => d.proj[key]);
    const lo = Math.min(...vals), hi = Math.max(...vals), p = (hi - lo) * 0.18 || 0.02;
    const x = d3.scaleLinear().domain([0, 360]).range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([lo - p, hi + p]).range([H - m.b, m.t]);
    const cur = rs.reduce((a, b) => Math.abs(b.hue - state.hue) < Math.abs(a.hue - state.hue) ? b : a);
    const red = rs.reduce((a, b) => Math.abs(b.hue) < Math.abs(a.hue) ? b : a);
    const sgn = (n) => (n >= 0 ? "+" : "");
    document.getElementById(labelId).innerHTML = labelHtml;

    const pole = cur.proj[key] >= 0 ? info.pos : info.neg;
    svg.append("text").attr("x", W / 2).attr("y", 22).attr("text-anchor", "middle")
      .attr("class", "ct").attr("fill", P.strong).attr("font-size", "14px").attr("font-weight", 700)
      .text(`“${name}” leans ${Math.abs(cur.proj[key]).toFixed(3)} toward ${pole}`);
    svg.append("text").attr("x", W / 2).attr("y", 41).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis)
      .text(`${sgn(cur.proj[key] - red.proj[key])}${(cur.proj[key] - red.proj[key]).toFixed(3)} toward ${info.pos} vs. red (0°)`);

    svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", H - m.b).attr("y2", H - m.b).attr("stroke", P.grid);
    svg.append("line").attr("x1", m.l).attr("x2", m.l).attr("y1", m.t).attr("y2", H - m.b).attr("stroke", P.grid);
    if (lo - p < 0 && hi + p > 0)
      svg.append("line").attr("x1", m.l).attr("x2", W - m.r).attr("y1", y(0)).attr("y2", y(0))
        .attr("stroke", P.grid).attr("stroke-dasharray", "3 3");
    svg.append("text").attr("x", m.l + 4).attr("y", m.t + 4).attr("class", "ax").attr("fill", P.strong)
      .attr("font-weight", 700).text("↑ " + info.pos);
    svg.append("text").attr("x", m.l + 4).attr("y", H - m.b - 5).attr("class", "ax").attr("fill", P.strong)
      .attr("font-weight", 700).text("↓ " + info.neg);
    [0, 90, 180, 270, 360].forEach((hv) => svg.append("text").attr("x", x(hv)).attr("y", H - m.b + 12)
      .attr("text-anchor", "middle").attr("class", "ax").attr("fill", P.axis).attr("font-size", "8px").text(hv));
    svg.append("text").attr("x", (W + m.l - m.r) / 2).attr("y", H - 3).attr("text-anchor", "middle")
      .attr("class", "ax").attr("fill", P.axis).text("text hue (°)");
    svg.append("path").datum(rs).attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1.6)
      .attr("d", d3.line().x((d) => x(d.hue)).y((d) => y(d.proj[key])));
    svg.selectAll("circle.pt").data(rs).enter().append("circle")
      .attr("cx", (d) => x(d.hue)).attr("cy", (d) => y(d.proj[key])).attr("r", 2.6)
      .attr("fill", (d) => colorForHue(d.hue)).attr("stroke", P.ring).attr("stroke-width", .5);
    svg.append("line").attr("x1", x(cur.hue)).attr("x2", x(cur.hue)).attr("y1", m.t).attr("y2", H - m.b)
      .attr("stroke", P.strong).attr("stroke-width", 1).attr("opacity", .4);
    svg.append("circle").attr("cx", x(cur.hue)).attr("cy", y(cur.proj[key])).attr("r", 8)
      .attr("fill", colorForHue(cur.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
  }

  // Rotatable 3-D PCA loop of `rs` (drag to rotate). varKey -> variance readout.
  function render3D(svgSel, rs, varKey, labelId, labelHtml) {
    const svg = d3.select(svgSel); svg.selectAll("*").remove();
    const P = pal();
    const W = 360, H = 300, cx = W / 2, cy = H / 2 + 4;
    svg.style("cursor", "grab");
    const mx = mean(rs.map((d) => d.wx)), my = mean(rs.map((d) => d.wy)), mz = mean(rs.map((d) => d.wz || 0));
    const pts = rs.map((d) => ({ x: d.wx - mx, y: d.wy - my, z: (d.wz || 0) - mz, hue: d.hue }));
    const maxr = Math.max(...pts.map((q) => Math.hypot(q.x, q.y, q.z))) || 1;
    const scale = (Math.min(W, H) / 2 - 30) / maxr;
    const cr = rs.reduce((a, b) => Math.abs(b.hue - state.hue) < Math.abs(a.hue - state.hue) ? b : a);
    const curC = { x: cr.wx - mx, y: cr.wy - my, z: (cr.wz || 0) - mz, hue: cr.hue };
    const vv = (DATA.meta.withinWordPcaVar || {})[varKey] || [];
    const pct = vv.length >= 3 && vv.every(Number.isFinite) ? Math.round((vv[0] + vv[1] + vv[2]) * 100) : null;
    document.getElementById(labelId).innerHTML = labelHtml;

    const proj = (q) => {
      const cY = Math.cos(rot.y), sY = Math.sin(rot.y);
      const x1 = q.x * cY - q.z * sY, z1 = q.x * sY + q.z * cY;
      const cX = Math.cos(rot.x), sX = Math.sin(rot.x);
      const y1 = q.y * cX - z1 * sX, z2 = q.y * sX + z1 * cX;
      return { sx: cx + x1 * scale, sy: cy - y1 * scale, depth: z2 };
    };
    const pr = pts.map((q) => ({ ...proj(q), hue: q.hue }));
    svg.append("path").attr("fill", "none").attr("stroke", P.line).attr("stroke-width", 1).attr("opacity", .55)
      .attr("d", d3.line().x((q) => q.sx).y((q) => q.sy)(pr.concat([pr[0]])));
    const ds = pr.map((d) => d.depth), dmin = Math.min(...ds), dmax = Math.max(...ds);
    const dep = (d) => (dmax === dmin ? 0.5 : (d.depth - dmin) / (dmax - dmin));
    svg.selectAll("circle.p3").data([...pr].sort((a, b) => a.depth - b.depth)).enter().append("circle")
      .attr("cx", (d) => d.sx).attr("cy", (d) => d.sy).attr("r", (d) => 2 + 2 * dep(d))
      .attr("fill", (d) => colorForHue(d.hue)).attr("opacity", (d) => 0.45 + 0.55 * dep(d))
      .attr("stroke", P.ring).attr("stroke-width", .4);
    const pc = proj(curC);
    svg.append("circle").attr("cx", pc.sx).attr("cy", pc.sy).attr("r", 8)
      .attr("fill", colorForHue(cr.hue)).attr("stroke", P.strong).attr("stroke-width", 2);
    svg.append("text").attr("x", W - 8).attr("y", H - 8).attr("text-anchor", "end")
      .attr("class", "ax").attr("fill", P.axis).attr("font-size", "9px")
      .text("drag to rotate" + (pct != null ? " · PC1–3 ≈ " + pct + "%" : ""));
  }

  // render both panels: middle = word, right = pure color, same view & axis
  function renderViews() {
    const wordRows = rows();
    const colorRows = DATA.data[COLOR_ONLY] || wordRows;
    if (state.view === "pca") {
      render3D("#planeSvg", wordRows, state.word, "planeLabel", `“${dispName(state.word)}” &middot; raw embedding (3-D)`);
      render3D("#curvesSvg", colorRows, COLOR_ONLY, "curvesLabel", "Pure color &middot; raw embedding (3-D)");
    } else {
      const key = primaryAxis(state.word);
      const info = axisInfo[key] || { pos: key, neg: "?" };
      renderAxis("#planeSvg", wordRows, key, dispName(state.word), "planeLabel", `“${dispName(state.word)}” &middot; ${info.neg} ↔ ${info.pos}`);
      renderAxis("#curvesSvg", colorRows, key, "pure color", "curvesLabel", `Pure color &middot; ${info.neg} ↔ ${info.pos}`);
    }
  }

  function renderAll() { renderStimulus(); renderViews(); }

  // ---------------- init ----------------
  function init(data) {
    DATA = data;
    COLOR_ONLY = data.meta.colorOnlyKey || COLOR_ONLY;
    (data.hueColors || []).forEach((d) => (hueHex[d.hue] = d.hex));
    (data.axes || []).forEach((a) => {
      axisLabel[a.key] = a.label;
      axisInfo[a.key] = { pos: a.pos, neg: a.neg, label: a.label };
    });

    const sel = document.getElementById("wordSelect");
    const sentiment = new Set(data.meta.sentimentWords || []);
    data.meta.words.forEach((w) => {
      if (w === COLOR_ONLY) return;        // shown as the fixed right-hand comparison, not selectable
      const o = document.createElement("option");
      o.value = w;
      o.textContent = sentiment.has(w) ? w + "  (sentiment)" : w;
      sel.appendChild(o);
    });
    sel.value = state.word;
    sel.addEventListener("change", (e) => { state.word = e.target.value; renderAll(); });

    document.getElementById("hueSlider").addEventListener("input", (e) => {
      state.hue = +e.target.value; renderAll();
    });
    // drag either 3-D panel to rotate both (shared orientation)
    const dragH = d3.drag().on("drag", (e) => {
      if (state.view !== "pca") return;
      rot.y -= e.dx * 0.012; rot.x += e.dy * 0.012; renderViews();
    });
    d3.select("#planeSvg").call(dragH);
    d3.select("#curvesSvg").call(dragH);
    document.querySelectorAll("#viewToggle button").forEach((b) => {
      b.addEventListener("click", () => {
        state.view = b.dataset.view;
        document.querySelectorAll("#viewToggle button").forEach((x) => x.classList.remove("is-active"));
        b.classList.add("is-active"); renderViews();
      });
    });

    document.getElementById("vizNote").innerHTML =
      `<b>The takeaway:</b> the right panel is always <b>pure color</b> (no word), on the same axis as the ` +
      `word — compare them. <b>(1) Color itself carries meaning:</b> a plain green swatch already leans ` +
      `<i>positive / safe</i>, a red one <i>negative</i>. <b>(2) But coloring a <i>word</i> is more than ` +
      `adding that color vector:</b> a word's hue-driven movement overlaps the pure-color direction by only ` +
      `~8%, so color largely <i>reshapes how that specific word is represented</i> — a color×word interaction. ` +
      `Either way a non-semantic property pushes the representation along meaning axes, which is what lets ` +
      `styling bias a VLM. <span class="prov">Probe: CLIP ViT-L/14-336, 72 hues; pure color = filled swatches.</span>`;

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
