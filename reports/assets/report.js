(function renderJLensReport() {
  "use strict";

  const registry = window.JLensReportRegistry;
  const ROUTES = ["concept", "jspace"];
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = {
    ink: "#171714",
    soft: "#5b5a52",
    line: "#d2cec1",
    concept: "#d55e00",
    jspace: "#0072b2",
    green: "#009e73",
    yellow: "#e69f00",
    pink: "#cc79a7",
    blueLight: "#56b4e9",
    danger: "#9b3328",
  };
  const SPECTRUM_COLORS = [
    "#440154",
    "#482878",
    "#3e4a89",
    "#31688e",
    "#26828e",
    "#1f9e89",
    "#35b779",
    "#6ece58",
    "#b5de2b",
  ];

  const state = {
    route: "concept",
    selectedRun: { concept: 0, jspace: 0 },
  };

  function escapeHTML(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function formatNumber(value, digits = 0) {
    return new Intl.NumberFormat("en-US", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    }).format(value);
  }

  function formatMetric(value, digits = 3) {
    return Number(value).toFixed(digits);
  }

  function mean(values) {
    return values.reduce((total, value) => total + value, 0) / values.length;
  }

  function svgElement(tag, attributes = {}, text = null) {
    const element = document.createElementNS(SVG_NS, tag);
    Object.entries(attributes).forEach(([key, value]) => {
      element.setAttribute(key, String(value));
    });
    if (text !== null) {
      element.textContent = text;
    }
    return element;
  }

  function createChart(host, label, width, height) {
    host.replaceChildren();
    const svg = svgElement("svg", {
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": label,
      preserveAspectRatio: "xMidYMid meet",
    });
    host.append(svg);
    return svg;
  }

  function linearScale(domainMin, domainMax, rangeMin, rangeMax) {
    const span = domainMax - domainMin || 1;
    return (value) => rangeMin + ((value - domainMin) / span) * (rangeMax - rangeMin);
  }

  function appendText(svg, x, y, text, className = "chart-label", anchor = "middle") {
    const element = svgElement(
      "text",
      { x, y, class: className, "text-anchor": anchor, "dominant-baseline": "middle" },
      text,
    );
    svg.append(element);
    return element;
  }

  function drawAxes(svg, options) {
    const {
      width,
      height,
      margin,
      xTicks,
      yTicks,
      xScale,
      yScale,
      xFormat = String,
      yFormat = String,
      xLabel,
      yLabel,
    } = options;
    const left = margin.left;
    const right = width - margin.right;
    const top = margin.top;
    const bottom = height - margin.bottom;

    yTicks.forEach((tick) => {
      const y = yScale(tick);
      svg.append(svgElement("line", { x1: left, x2: right, y1: y, y2: y, class: "chart-gridline" }));
      appendText(svg, left - 10, y, yFormat(tick), "chart-label", "end");
    });

    xTicks.forEach((tick) => {
      const x = xScale(tick);
      svg.append(svgElement("line", { x1: x, x2: x, y1: top, y2: bottom, class: "chart-gridline" }));
      appendText(svg, x, bottom + 20, xFormat(tick));
    });

    svg.append(svgElement("line", { x1: left, x2: right, y1: bottom, y2: bottom, class: "chart-axis" }));
    svg.append(svgElement("line", { x1: left, x2: left, y1: top, y2: bottom, class: "chart-axis" }));

    if (xLabel) {
      appendText(svg, (left + right) / 2, height - 12, xLabel, "chart-label");
    }
    if (yLabel) {
      const label = svgElement(
        "text",
        {
          x: 16,
          y: (top + bottom) / 2,
          class: "chart-label",
          "text-anchor": "middle",
          transform: `rotate(-90 16 ${(top + bottom) / 2})`,
        },
        yLabel,
      );
      svg.append(label);
    }
  }

  function linePath(points) {
    return points
      .map(([x, y], index) => `${index === 0 ? "M" : "L"}${x.toFixed(2)},${y.toFixed(2)}`)
      .join(" ");
  }

  function legendHTML(items) {
    return `<div class="legend">${items
      .map(
        ([label, color]) => `
          <span class="legend-item">
            <span class="legend-swatch" style="--swatch:${escapeHTML(color)}"></span>
            ${escapeHTML(label)}
          </span>`,
      )
      .join("")}</div>`;
  }

  function sectionHeading(index, title, description) {
    return `
      <div class="section-heading">
        <p class="section-kicker">${escapeHTML(index)}</p>
        <div>
          <h3>${escapeHTML(title)}</h3>
          <p>${escapeHTML(description)}</p>
        </div>
      </div>`;
  }

  function metric(label, value, note) {
    return `
      <div class="metric">
        <span class="metric-label">${escapeHTML(label)}</span>
        <span class="metric-value">${escapeHTML(value)}</span>
        <span class="metric-note">${escapeHTML(note)}</span>
      </div>`;
  }

  function provenanceHTML(run) {
    return run.provenance
      .map(
        ([label, value]) => `
          <div class="provenance-row">
            <dt>${escapeHTML(label)}</dt>
            <dd>${escapeHTML(value)}</dd>
          </div>`,
      )
      .join("");
  }

  function limitationsHTML(run) {
    return run.limitations.map((item) => `<li>${escapeHTML(item)}</li>`).join("");
  }

  function chartPanel(id, eyebrow, title, description, legend = "", wide = false) {
    return `
      <article class="chart-panel${wide ? " is-wide" : ""}">
        <header class="chart-header">
          <div>
            <p class="chart-eyebrow">${escapeHTML(eyebrow)}</p>
            <h4>${escapeHTML(title)}</h4>
            ${legend}
          </div>
          <p>${escapeHTML(description)}</p>
        </header>
        <div class="chart-host" id="${escapeHTML(id)}"></div>
      </article>`;
  }

  function drawProbeChart(host, run) {
    const width = 760;
    const rowHeight = 45;
    const height = 88 + run.probeResults.length * rowHeight;
    const margin = { top: 24, right: 54, bottom: 48, left: 112 };
    const svg = createChart(host, "Concept probe held-out ROC AUC and average precision", width, height);
    const x = linearScale(0, 1, margin.left, width - margin.right);
    const yAt = (index) => margin.top + index * rowHeight + 20;

    [0, 0.25, 0.5, 0.75, 1].forEach((tick) => {
      const xValue = x(tick);
      svg.append(
        svgElement("line", {
          x1: xValue,
          x2: xValue,
          y1: margin.top,
          y2: height - margin.bottom,
          class: "chart-gridline",
        }),
      );
      appendText(svg, xValue, height - 25, tick.toFixed(2));
    });

    run.probeResults.forEach((result, index) => {
      const y = yAt(index);
      appendText(svg, margin.left - 12, y, result.concept, "chart-value", "end");
      svg.append(
        svgElement("line", {
          x1: x(result.minAuc),
          x2: x(result.maxAuc),
          y1: y,
          y2: y,
          stroke: COLORS.jspace,
          "stroke-width": 5,
          "stroke-linecap": "round",
          opacity: 0.28,
        }),
      );
      svg.append(
        svgElement("circle", {
          cx: x(result.meanAuc),
          cy: y,
          r: 6,
          fill: COLORS.jspace,
          stroke: "#fbfaf6",
          "stroke-width": 2,
        }),
      );
      svg.append(
        svgElement("rect", {
          x: x(result.meanAp) - 5,
          y: y - 5,
          width: 10,
          height: 10,
          fill: COLORS.concept,
          transform: `rotate(45 ${x(result.meanAp)} ${y})`,
        }),
      );
      appendText(svg, x(result.meanAuc) + 10, y - 12, result.meanAuc.toFixed(3), "chart-value", "start");
    });

    appendText(svg, (margin.left + width - margin.right) / 2, height - 7, "Held-out score (0–1)");
  }

  function drawConceptLayerChart(host, run) {
    const width = 650;
    const height = 390;
    const margin = { top: 28, right: 32, bottom: 55, left: 72 };
    const svg = createChart(host, "Mean held-out ROC AUC across concepts by source layer", width, height);
    const values = run.layerMeanAuc.map((item) => item.auc);
    const yMin = Math.floor((Math.min(...values) - 0.004) * 100) / 100;
    const yMax = Math.ceil((Math.max(...values) + 0.004) * 100) / 100;
    const x = linearScale(Math.min(...run.layers), Math.max(...run.layers), margin.left, width - margin.right);
    const y = linearScale(yMin, yMax, height - margin.bottom, margin.top);
    const yTicks = Array.from({ length: 5 }, (_, index) => yMin + ((yMax - yMin) * index) / 4);

    drawAxes(svg, {
      width,
      height,
      margin,
      xTicks: run.layers,
      yTicks,
      xScale: x,
      yScale: y,
      yFormat: (value) => value.toFixed(3),
      xLabel: "Source layer",
      yLabel: "Mean test ROC AUC",
    });

    const points = run.layerMeanAuc.map((item) => [x(item.layer), y(item.auc)]);
    svg.append(
      svgElement("path", {
        d: linePath(points),
        fill: "none",
        stroke: COLORS.concept,
        "stroke-width": 3,
      }),
    );
    run.layerMeanAuc.forEach((item) => {
      svg.append(
        svgElement("circle", {
          cx: x(item.layer),
          cy: y(item.auc),
          r: 5,
          fill: COLORS.concept,
          stroke: "#fbfaf6",
          "stroke-width": 2,
        }),
      );
      appendText(svg, x(item.layer), y(item.auc) - 15, item.auc.toFixed(3), "chart-value");
    });
  }

  function conceptTableHTML(run) {
    return `
      <div class="table-wrap">
        <table class="result-table">
          <thead>
            <tr>
              <th>Concept</th>
              <th>Mean AUC</th>
              <th>AUC range</th>
              <th>Mean AP</th>
              <th>Best layer</th>
              <th>Gate</th>
            </tr>
          </thead>
          <tbody>
            ${run.probeResults
              .map(
                (result) => `
                <tr>
                  <td>${escapeHTML(result.concept)}</td>
                  <td>${formatMetric(result.meanAuc, 4)}</td>
                  <td>${formatMetric(result.minAuc, 4)}–${formatMetric(result.maxAuc, 4)}</td>
                  <td>${formatMetric(result.meanAp, 4)}</td>
                  <td>${result.bestLayer}</td>
                  <td><span class="gate-pill gate-${escapeHTML(result.gate)}">${escapeHTML(result.gate)}</span></td>
                </tr>`,
              )
              .join("")}
          </tbody>
        </table>
      </div>`;
  }

  function conceptTokensHTML(run) {
    return run.alignments
      .map(
        (alignment) => `
          <article class="token-card">
            <div class="token-card-header">
              <h4>${escapeHTML(alignment.concept)}</h4>
              <span class="gate-pill gate-${escapeHTML(alignment.gate)}">${escapeHTML(alignment.gate)}</span>
            </div>
            <p>${escapeHTML(alignment.note)}</p>
            <div class="token-list">
              ${alignment.tokens
                .map(
                  (token) => `
                    <span class="token-chip">
                      ${escapeHTML(token.token)}
                      <small>${formatMetric(token.score, 3)}</small>
                    </span>`,
                )
                .join("")}
            </div>
          </article>`,
      )
      .join("");
  }

  function renderConcept(view, run) {
    const bestLayer = run.layerMeanAuc.reduce((best, item) => (item.auc > best.auc ? item : best));
    const strongest = [...run.probeResults].sort((a, b) => b.meanAuc - a.meanAuc)[0];
    view.className = "report-view concept-theme";
    view.innerHTML = `
      <header class="direction-hero">
        <div>
          <p class="section-kicker">Direction 01 / ${escapeHTML(run.shortTitle)}</p>
          <h2>${escapeHTML(run.title)}</h2>
          <p class="question">${escapeHTML(run.question)}</p>
        </div>
        <aside class="hero-summary">
          <span class="status-pill">${escapeHTML(run.status)}</span>
          <p>${escapeHTML(run.summary)}</p>
          <a class="report-link" href="${escapeHTML(run.sourceReport)}">Read narrative Markdown →</a>
        </aside>
      </header>

      <div class="metric-strip">
        ${metric("source examples", formatNumber(run.counts.examples), "group-preserving source texts")}
        ${metric("held-out probes", String(run.counts.probes), "7 concepts × 6 layers")}
        ${metric("best layer mean AUC", formatMetric(bestLayer.auc, 3), `layer ${bestLayer.layer}`)}
        ${metric("strongest concept", strongest.concept, `mean AUC ${formatMetric(strongest.meanAuc, 3)}`)}
      </div>

      <section class="report-section">
        ${sectionHeading("01 / Probe evidence", "Held-out linear decodability", "ROC AUC measures ranking performance; average precision also reflects class imbalance. The pale blue segment shows the six-layer AUC range, and the dot shows the layer mean.")}
        <div class="chart-grid">
          ${chartPanel(
            "concept-probe-chart",
            "Figure C1",
            "Probe performance by concept",
            "Blue circle: mean ROC AUC; orange diamond: mean average precision; segment: cross-layer AUC range.",
            legendHTML([
              ["ROC AUC", COLORS.jspace],
              ["Average precision", COLORS.concept],
            ]),
            true,
          )}
          ${chartPanel(
            "concept-layer-chart",
            "Figure C2",
            "Layer-wise mean AUC",
            "Each point is the mean held-out ROC AUC across seven concepts at one source layer. The local y-axis range is labeled explicitly.",
          )}
          <article class="text-panel">
            <p class="chart-eyebrow">Interpretation</p>
            <h4>What is established—and what is not</h4>
            <p>
              Gratitude, curiosity, and love have the strongest held-out linear signals;
              approval is weakest. All 42 AUC values reproduce exactly from saved activations
              and raw-coordinate probe vectors, with a maximum error of 0. This establishes
              decodability, not a causal effect on generation.
            </p>
            <p>
              The appropriate next step is to run matched full, J, non-J, and random controls
              for concepts that pass the strong gate before expanding intervention to all seven concepts.
            </p>
          </article>
        </div>
        <div style="margin-top:22px">${conceptTableHTML(run)}</div>
      </section>

      <section class="report-section">
        ${sectionHeading("02 / Alignment", "Token J-direction semantic readout", "The cards show the three highest positive layer-28 tokens and their cosine scores. Gates are cautious evidence-quality categories, not significance tests.")}
        <div class="token-grid">${conceptTokensHTML(run)}</div>
      </section>

      <section class="report-section">
        ${sectionHeading("03 / Audit", "Provenance and evidence limits", "The report retains immutable artifact identity while making the limits of the v1 evidence explicit.")}
        <div class="two-column">
          <article class="provenance-panel">
            <h4>Run identity</h4>
            <dl class="provenance-list">${provenanceHTML(run)}</dl>
          </article>
          <article class="text-panel">
            <h4>Limitations</h4>
            <ol class="limitation-list">${limitationsHTML(run)}</ol>
          </article>
        </div>
      </section>`;

    drawProbeChart(document.getElementById("concept-probe-chart"), run);
    drawConceptLayerChart(document.getElementById("concept-layer-chart"), run);
  }

  function drawSpectrumChart(host, run, cumulative = false) {
    const width = 740;
    const height = 430;
    const margin = { top: 24, right: 36, bottom: 58, left: 78 };
    const label = cumulative
      ? "Cumulative singular-value energy by source layer"
      : "Normalized singular-value spectrum by source layer";
    const svg = createChart(host, label, width, height);
    const x = linearScale(0, 1, margin.left, width - margin.right);
    const y = cumulative
      ? linearScale(0, 1, height - margin.bottom, margin.top)
      : linearScale(-7, 0, height - margin.bottom, margin.top);
    const yTicks = cumulative ? [0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99] : [-6, -4, -2, 0];

    drawAxes(svg, {
      width,
      height,
      margin,
      xTicks: [0, 0.25, 0.5, 0.75, 1],
      yTicks,
      xScale: x,
      yScale: y,
      xFormat: (value) => value.toFixed(2),
      yFormat: cumulative ? (value) => value.toFixed(2) : (value) => `10^${value}`,
      xLabel: "Normalized rank r / D",
      yLabel: cumulative ? "Cumulative spectral energy" : "Normalized singular value sᵣ / s₁",
    });

    run.spectra.forEach((spectrum, seriesIndex) => {
      const values = cumulative ? spectrum.cumulative : spectrum.singular;
      const points = run.spectrumRanks.map((rank, index) => [
        x(rank),
        y(cumulative ? values[index] : Math.log10(Math.max(values[index], 1e-7))),
      ]);
      svg.append(
        svgElement("path", {
          d: linePath(points),
          fill: "none",
          stroke: SPECTRUM_COLORS[seriesIndex],
          "stroke-width": 2.2,
          opacity: 0.95,
        }),
      );
    });
  }

  function drawDimensionChart(host, run) {
    const width = 740;
    const height = 430;
    const margin = { top: 24, right: 36, bottom: 58, left: 72 };
    const svg = createChart(host, "J-space effective dimension across source layers", width, height);
    const x = linearScale(Math.min(...run.layers), Math.max(...run.layers), margin.left, width - margin.right);
    const y = linearScale(0, 1, height - margin.bottom, margin.top);
    const series = [
      ["k90", COLORS.jspace],
      ["k95", COLORS.concept],
      ["k99", COLORS.green],
      ["entropyRank", COLORS.pink],
      ["participationRatio", COLORS.ink],
    ];

    drawAxes(svg, {
      width,
      height,
      margin,
      xTicks: run.layers,
      yTicks: [0, 0.25, 0.5, 0.75, 1],
      xScale: x,
      yScale: y,
      yFormat: (value) => value.toFixed(2),
      xLabel: "Source layer",
      yLabel: "Dimension / D (D = 2560)",
    });

    series.forEach(([key, color], seriesIndex) => {
      const points = run.metrics.map((row) => [x(row.layer), y(row[key] / run.dimension)]);
      svg.append(
        svgElement("path", {
          d: linePath(points),
          fill: "none",
          stroke: color,
          "stroke-width": key === "participationRatio" ? 2 : 2.6,
          "stroke-dasharray": key === "participationRatio" ? "4 4" : seriesIndex === 3 ? "8 4" : "none",
        }),
      );
      points.forEach(([cx, cy]) => {
        svg.append(svgElement("circle", { cx, cy, r: 3.5, fill: color }));
      });
    });
  }

  function hexToRgb(hex) {
    const value = hex.replace("#", "");
    return [0, 2, 4].map((offset) => Number.parseInt(value.slice(offset, offset + 2), 16));
  }

  function interpolateColor(value) {
    const stops = [
      [0, "#00204c"],
      [0.25, "#414d6b"],
      [0.5, "#7d7c78"],
      [0.75, "#bcae6c"],
      [1, "#fee838"],
    ];
    const clamped = Math.max(0, Math.min(1, value));
    const upperIndex = Math.min(
      stops.length - 1,
      Math.max(1, stops.findIndex(([position]) => position >= clamped)),
    );
    const [lowPosition, lowColor] = stops[upperIndex - 1];
    const [highPosition, highColor] = stops[upperIndex];
    const amount = (clamped - lowPosition) / (highPosition - lowPosition || 1);
    const low = hexToRgb(lowColor);
    const high = hexToRgb(highColor);
    const rgb = low.map((channel, index) => Math.round(channel + (high[index] - channel) * amount));
    return `rgb(${rgb.join(" ")})`;
  }

  function drawOverlapHeatmap(host, run) {
    const width = 680;
    const height = 630;
    const margin = { top: 34, right: 34, bottom: 72, left: 72 };
    const svg = createChart(host, "Top-64 right-singular subspace overlap heatmap", width, height);
    const gridSize = Math.min(width - margin.left - margin.right, height - margin.top - margin.bottom);
    const cell = gridSize / run.layers.length;

    run.top64Overlap.forEach((row, rowIndex) => {
      row.forEach((value, columnIndex) => {
        const x = margin.left + columnIndex * cell;
        const y = margin.top + rowIndex * cell;
        svg.append(
          svgElement("rect", {
            x,
            y,
            width: cell + 0.3,
            height: cell + 0.3,
            fill: interpolateColor(value),
          }),
        );
        appendText(
          svg,
          x + cell / 2,
          y + cell / 2,
          value.toFixed(2),
          "chart-value",
          "middle",
        ).setAttribute?.("fill", value > 0.57 ? COLORS.ink : "#ffffff");
      });
    });

    run.layers.forEach((layer, index) => {
      appendText(svg, margin.left - 10, margin.top + (index + 0.5) * cell, `L${layer}`, "chart-label", "end");
      appendText(svg, margin.left + (index + 0.5) * cell, margin.top + gridSize + 22, `L${layer}`);
    });
    appendText(svg, margin.left + gridSize / 2, height - 18, "Source layer");
    const yLabel = svgElement(
      "text",
      {
        x: 16,
        y: margin.top + gridSize / 2,
        class: "chart-label",
        "text-anchor": "middle",
        transform: `rotate(-90 16 ${margin.top + gridSize / 2})`,
      },
      "Source layer",
    );
    svg.append(yLabel);
  }

  function jspaceTableHTML(run) {
    return `
      <div class="table-wrap">
        <table class="result-table">
          <thead>
            <tr>
              <th>Layer</th>
              <th>Numerical rank</th>
              <th>Entropy rank</th>
              <th>Participation</th>
              <th>Stable rank</th>
              <th>k90</th>
              <th>k95</th>
              <th>k99</th>
            </tr>
          </thead>
          <tbody>
            ${run.metrics
              .map(
                (row) => `
                  <tr>
                    <td>L${row.layer}</td>
                    <td>${row.rank}</td>
                    <td>${formatMetric(row.entropyRank, 2)}</td>
                    <td>${formatMetric(row.participationRatio, 2)}</td>
                    <td>${formatMetric(row.stableRank, 2)}</td>
                    <td>${row.k90}</td>
                    <td>${row.k95}</td>
                    <td>${row.k99}</td>
                  </tr>`,
              )
              .join("")}
          </tbody>
        </table>
      </div>`;
  }

  function renderJspace(view, run) {
    const first = run.metrics[0];
    const last = run.metrics.at(-1);
    const distantOverlap = run.top64Overlap[0].at(-1);
    const spectrumLegend = legendHTML(run.layers.map((layer, index) => [`L${layer}`, SPECTRUM_COLORS[index]]));
    const dimensionLegend = legendHTML([
      ["90% energy", COLORS.jspace],
      ["95% energy", COLORS.concept],
      ["99% energy", COLORS.green],
      ["Entropy rank", COLORS.pink],
      ["Participation ratio", COLORS.ink],
    ]);

    view.className = "report-view jspace-theme";
    view.innerHTML = `
      <header class="direction-hero">
        <div>
          <p class="section-kicker">Direction 02 / ${escapeHTML(run.shortTitle)}</p>
          <h2>${escapeHTML(run.title)}</h2>
          <p class="question">${escapeHTML(run.question)}</p>
        </div>
        <aside class="hero-summary">
          <span class="status-pill">${escapeHTML(run.status)}</span>
          <p>${escapeHTML(run.summary)}</p>
          <a class="report-link" href="${escapeHTML(run.sourceReport)}">Read narrative Markdown →</a>
        </aside>
      </header>

      <div class="metric-strip">
        ${metric("numerical rank", String(first.rank), "every analyzed layer")}
        ${metric("L0 k90", String(first.k90), `${formatMetric(first.k90 / run.dimension, 3)} × D`)}
        ${metric("L30 k90", String(last.k90), `${formatMetric(last.k90 / run.dimension, 3)} × D`)}
        ${metric("L0 ↔ L30 overlap", formatMetric(distantOverlap, 3), "dimension-matched top-64")}
      </div>

      <section class="report-section">
        ${sectionHeading("01 / Spectrum", "From concentrated to distributed sensitivity", "A singular value measures the vocabulary-output change caused by a unit residual direction; its square contributes spectral energy. Each layer is normalized by its own largest singular value.")}
        <div class="method-note" style="margin-bottom:18px">
          <h4>How to read a singular value</h4>
          <p>
            A steep drop means a few directions dominate; a flatter curve means many directions
            have comparable influence. Normalized curves compare spectral shape, not absolute
            sensitivity across layers.
          </p>
        </div>
        <div class="chart-grid">
          ${chartPanel(
            "jspace-spectrum-chart",
            "Figure J1",
            "Normalized singular-value spectrum",
            "The y-axis is logarithmic. Early layers decay rapidly; late-layer spectra flatten.",
            spectrumLegend,
          )}
          ${chartPanel(
            "jspace-cumulative-chart",
            "Figure J2",
            "Cumulative spectral energy",
            "Earlier approach to 1 indicates energy concentrated in fewer right-singular directions.",
            spectrumLegend,
          )}
        </div>
      </section>

      <section class="report-section">
        ${sectionHeading("02 / Effective dimension", "J-space expands sharply with depth", "Numerical rank remains 2560. Structural change appears in energy-threshold dimensions and effective-rank measures, not exact rank deficiency.")}
        <div class="chart-grid">
          ${chartPanel(
            "jspace-dimension-chart",
            "Figure J3",
            "Energy dimensions and effective ranks",
            "All measures are divided by D=2560 so their fraction of residual dimension is directly comparable.",
            dimensionLegend,
            true,
          )}
        </div>
        <div style="margin-top:22px">${jspaceTableHTML(run)}</div>
      </section>

      <section class="report-section">
        ${sectionHeading("03 / Subspace drift", "Local continuity, global reorganization", "The first 64 right-singular directions are compared at every layer, avoiding trivial overlap from unequal energy-basis dimensions.")}
        <div class="chart-grid">
          ${chartPanel(
            "jspace-overlap-chart",
            "Figure J4",
            "Top-64 subspace overlap",
            "Values are ‖BᵢᵀBⱼ‖²F / 64. Adjacent layers retain moderate-to-high overlap; distant layers are nearly orthogonal.",
            "",
            true,
          )}
        </div>
      </section>

      <section class="report-section">
        ${sectionHeading("04 / Audit", "Provenance and evidence limits", "v1 is one centered run on one fit-prompt sample. Interpretations must remain within that design.")}
        <div class="two-column">
          <article class="provenance-panel">
            <h4>Run identity</h4>
            <dl class="provenance-list">${provenanceHTML(run)}</dl>
          </article>
          <article class="text-panel">
            <h4>Limitations</h4>
            <ol class="limitation-list">${limitationsHTML(run)}</ol>
          </article>
        </div>
      </section>`;

    drawSpectrumChart(document.getElementById("jspace-spectrum-chart"), run, false);
    drawSpectrumChart(document.getElementById("jspace-cumulative-chart"), run, true);
    drawDimensionChart(document.getElementById("jspace-dimension-chart"), run);
    drawOverlapHeatmap(document.getElementById("jspace-overlap-chart"), run);
  }

  function runsFor(route) {
    return registry.get(route);
  }

  function renderRoute(route) {
    const runs = runsFor(route);
    const view = document.getElementById(`${route}-view`);
    if (runs.length === 0) {
      view.innerHTML = `<p class="app-error">No registered ${escapeHTML(route)} runs.</p>`;
      return;
    }
    const selected = Math.min(state.selectedRun[route], runs.length - 1);
    state.selectedRun[route] = selected;
    if (route === "concept") {
      renderConcept(view, runs[selected]);
    } else {
      renderJspace(view, runs[selected]);
    }
  }

  function updateRunPicker(route) {
    const select = document.getElementById("run-select");
    const runs = runsFor(route);
    select.replaceChildren(
      ...runs.map((run, index) => {
        const option = document.createElement("option");
        option.value = String(index);
        option.textContent = run.shortTitle || run.title;
        option.selected = index === state.selectedRun[route];
        return option;
      }),
    );
  }

  function activateRoute(route, updateHash = true) {
    const safeRoute = ROUTES.includes(route) ? route : "concept";
    state.route = safeRoute;
    document.querySelectorAll("[data-route]").forEach((button) => {
      const active = button.dataset.route === safeRoute;
      button.classList.toggle("is-active", active);
      button.setAttribute("aria-selected", String(active));
    });
    ROUTES.forEach((candidate) => {
      document.getElementById(`${candidate}-view`).hidden = candidate !== safeRoute;
    });
    updateRunPicker(safeRoute);
    if (updateHash && window.location.hash !== `#${safeRoute}`) {
      window.history.replaceState(null, "", `#${safeRoute}`);
    }
  }

  function showError(error) {
    const target = document.getElementById("app-error");
    target.hidden = false;
    target.textContent = `Report rendering failed: ${error.message}`;
    console.error(error);
  }

  function initialize() {
    if (!registry) {
      throw new Error("Report registry did not load.");
    }
    ROUTES.forEach(renderRoute);

    document.querySelectorAll("[data-route]").forEach((button) => {
      button.addEventListener("click", () => activateRoute(button.dataset.route));
    });
    document.getElementById("run-select").addEventListener("change", (event) => {
      state.selectedRun[state.route] = Number.parseInt(event.target.value, 10);
      renderRoute(state.route);
    });
    document.getElementById("print-report").addEventListener("click", () => window.print());
    window.addEventListener("hashchange", () => activateRoute(window.location.hash.slice(1), false));

    activateRoute(window.location.hash.slice(1), false);
  }

  try {
    initialize();
  } catch (error) {
    showError(error);
  }
})();
