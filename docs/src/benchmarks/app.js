"use strict";

const repo = "https://github.com/vtjeng/MIPVerify.jl";
const raw = "https://raw.githubusercontent.com/vtjeng/MIPVerify.jl/benchmark-results";
const groups = {
  Timings: [
    [
      "wall_clock_seconds",
      "Total runtime",
      "Elapsed wall-clock time for the entire benchmark run.",
    ],
    ["sum_total_time_seconds", "Total sample time", "Sum of end-to-end times across all samples."],
    [
      "sum_solve_time_seconds",
      "Total solver time",
      "Sum of per-sample solve times reported by the solver.",
    ],
    [
      "sum_formulation_time_seconds",
      "Model formulation time",
      "Time spent building models, including bound tightening.",
    ],
    [
      "sum_bound_tightening_time_seconds",
      "Bound-tightening time",
      "Time spent tightening bounds during model formulation.",
    ],
    [
      "sum_formulation_residual_time_seconds",
      "Other formulation time",
      "Formulation time remaining after bound tightening.",
    ],
    [
      "sum_main_solve_wall_time_seconds",
      "Main solve wall time",
      "Wall-clock time spent in the final solve across samples.",
    ],
    [
      "median_solve_time_seconds",
      "Median sample solve time",
      "50th percentile of the per-sample solve times.",
    ],
    [
      "p90_solve_time_seconds",
      "90th percentile solve time",
      "90th percentile of the per-sample solve times.",
    ],
  ],
  "Sample counts": [
    ["num_samples", "Samples evaluated", "Number of inputs evaluated in the run."],
    ["num_rows_in_summary", "Sample summary rows", "Number of rows in the per-sample results."],
    [
      "num_missing_objective_value",
      "Missing objective values",
      "Samples with a missing objective value.",
    ],
  ],
  "Solver statuses": [
    ["num_optimal_status", "Optimal", "Samples with solver status OPTIMAL."],
    ["num_infeasible_status", "Infeasible", "Samples with solver status INFEASIBLE."],
    [
      "num_infeasible_or_unbounded_status",
      "Infeasible or unbounded",
      "Samples with solver status INFEASIBLE_OR_UNBOUNDED.",
    ],
    [
      "num_time_limit_status",
      "Time limit reached",
      "Samples with solver status TIME_LIMIT, including those with a usable result.",
    ],
    [
      "num_solution_limit_status",
      "Solution limit reached",
      "Samples with solver status SOLUTION_LIMIT.",
    ],
    [
      "num_objective_limit_status",
      "Objective limit reached",
      "Samples with solver status OBJECTIVE_LIMIT.",
    ],
    [
      "num_skipped_predicted_in_targeted",
      "Skipped before modeling",
      "Inputs whose original prediction was already in the target set.",
    ],
  ],
  "Verification outcomes": [
    [
      "num_certified_no_adversarial_example",
      "Certified robust",
      "Inputs proven to have no adversarial example within the budget.",
    ],
    [
      "num_adversarial_example_found_or_best_known",
      "Adversarial example found",
      "Inputs recorded as having an adversarial example under that run’s counting rules.",
    ],
    [
      "num_time_limit_unresolved",
      "Unresolved at time limit",
      "Inputs still unresolved when the solver reached its time limit.",
    ],
    [
      "num_no_primal_solution_other",
      "Other cases without a solution",
      "Inputs classified as having no primal solution for another reason.",
    ],
    [
      "num_witness_verification_failed",
      "Witness verification failed",
      "Inputs classified as having a witness that failed independent verification.",
    ],
  ],
  "Witness checks": [
    [
      "num_witness_available",
      "Witness available",
      "Inputs with an available candidate adversarial example.",
    ],
    [
      "num_witness_target_verified",
      "Target check passed",
      "Available witnesses that passed the target check.",
    ],
    [
      "num_witness_perturbation_verified",
      "Perturbation check passed",
      "Available witnesses that passed the perturbation check.",
    ],
    [
      "num_witness_target_verification_failed",
      "Target check failed",
      "Available witnesses that failed the target check.",
    ],
    [
      "num_witness_perturbation_verification_failed",
      "Perturbation check failed",
      "Available witnesses that failed the perturbation check.",
    ],
    [
      "num_witness_verified",
      "Witness verified",
      "Available witnesses that passed both independent checks.",
    ],
  ],
};
const metrics = new Map();
const el = (id) => document.getElementById(id);
for (const [group, entries] of Object.entries(groups)) {
  const options = document.createElement("optgroup");
  options.label = group;
  for (const [key, label, description] of entries) {
    metrics.set(key, { label, description });
    options.append(new Option(label, key));
  }
  el("metric").append(options);
}
const dateFormat = new Intl.DateTimeFormat("en-GB", {
  timeZone: "UTC",
  day: "numeric",
  month: "short",
  year: "numeric",
});
const dateLabel = (day) => dateFormat.format(new Date(day + "T00:00:00Z"));
const numberFormat = new Intl.NumberFormat("en-US", { maximumFractionDigits: 2 });
const objective = (row) => row.adversarial_example_objective || "closest";
const valueOf = (row, key) => {
  const value = row[key];
  return value != null && value !== "" && Number.isFinite(Number(value)) ? Number(value) : null;
};
function scale(key) {
  if (["median_solve_time_seconds", "p90_solve_time_seconds"].includes(key)) return [1000, "ms"];
  if (key.endsWith("_seconds")) return [1 / 60, "min"];
  return [1, ""];
}
function readout(row, key) {
  const value = valueOf(row, key);
  if (value === null) return "Unavailable";
  const [factor, unit] = scale(key);
  return (numberFormat.format(value * factor) + " " + unit).trim();
}
function runPath(row) {
  return "runs/" + row.date + (row.run_id ? "/" + encodeURIComponent(row.run_id) : "");
}
// Plotly expects log-axis limits in base-10 exponents, but trace values in display units.
function axisRange(values, logScale) {
  if (!values.length) return logScale ? [0, 1] : [-0.12, 1];
  const max = Math.max(...values);
  if (logScale) {
    const low = Math.log10(Math.min(...values));
    const high = Math.log10(max);
    const padding = high === low ? 0.05 : (high - low) * 0.05;
    return [low - padding, high + padding];
  }
  return max === 0 ? [-0.12, 1] : [-0.03 * max, 1.05 * max];
}
let preferLog = false;
let history = [],
  visible = [],
  selected = "",
  bound = false;
function restoreView() {
  const params = new URLSearchParams(location.search);
  const key = params.get("metric");
  el("metric").value = metrics.has(key) ? key : "wall_clock_seconds";
  preferLog = params.get("log") === "1";
  el("fixed-axis").checked = params.get("fixed") !== "0";

  const end = params.get("end");
  const day = utcDay(end);
  const validDate = /^\d{4}-\d{2}-\d{2}$/.test(end) && Number.isFinite(day) && isoDay(day) === end;
  windowEnd = validDate ? Math.min(afterLastDay, Math.max(firstDay + 1, day + 1)) : afterLastDay;
  const interval = params.get("interval");
  if (interval === "all") windowStart = firstDay;
  else {
    const days = Number(interval);
    presetWindow(Number.isSafeInteger(days) && days > 0 ? days : 90);
  }
}

function shareView() {
  if (!history.length) return;
  const url = new URL(location.href);
  const all = windowStart === firstDay && windowEnd === afterLastDay;
  url.searchParams.set("interval", all ? "all" : String(windowEnd - windowStart));
  const endDate = isoDay(windowEnd - 1);
  if (windowEnd < afterLastDay) url.searchParams.set("end", endDate);
  // At the newest run, keep an explicitly linked date pinned. Returning here
  // from an older date restores the rolling view.
  else if (url.searchParams.get("end") !== endDate) url.searchParams.delete("end");
  url.searchParams.set("metric", el("metric").value);
  url.searchParams.set("log", el("log-scale").checked ? "1" : "0");
  url.searchParams.set("fixed", el("fixed-axis").checked ? "1" : "0");
  if (url.href !== location.href) window.history.replaceState(window.history.state, "", url);
}

function renderAndShare() {
  render();
  shareView();
}

function inspect() {
  const row = visible.find((r) => r.date === selected);
  if (!row) return;
  el("day").value = selected;
  const key = el("metric").value;
  const stats = [
    [metrics.get(key).label, readout(row, key)],
    ["Objective", objective(row)],
    ["Julia version", row.julia_version || "Unavailable"],
    ["Outcome schema", row.semantic_outcome_schema_version || "Unversioned"],
  ];
  el("stats").replaceChildren(
    ...stats.map(([label, value]) => {
      const item = document.createElement("div"),
        caption = document.createElement("span"),
        result = document.createElement("strong");
      item.className = "stat";
      caption.textContent = label;
      result.textContent = value;
      item.append(caption, result);
      return item;
    }),
  );
  el("env").textContent = [
    row.num_samples && row.num_samples + " samples",
    row.tightening_algorithm && row.tightening_algorithm.toUpperCase() + " tightening",
    row.main_time_limit_seconds && Number(row.main_time_limit_seconds) + " s main-solve limit",
  ]
    .filter(Boolean)
    .join(" · ");
  el("artifact").href = repo + "/tree/benchmark-results/" + runPath(row);
  el("csv").href = repo + "/blob/benchmark-results/" + runPath(row) + "/benchmark_metrics.csv";
  el("commit").href = repo + "/commit/" + row.commit_sha;
  el("commit").textContent = "Commit " + row.commit_sha.slice(0, 7);
  showRunPRs(row);
}
function render(resetSelection = false) {
  if (!history.length) return;
  const latest = history.at(-1);
  visible = history.filter((r) => utcDay(r.date) >= windowStart && utcDay(r.date) < windowEnd);
  if (resetSelection || !visible.some((r) => r.date === selected)) selected = visible.at(-1)?.date;
  document.querySelector(".inspect").hidden = !visible.length;
  el("latest").textContent = dateLabel(latest.date);
  el("day").replaceChildren(
    ...visible
      .slice()
      .reverse()
      .map((r) => new Option(dateLabel(r.date), r.date)),
  );
  const key = el("metric").value,
    metric = metrics.get(key);
  el("history").textContent = metric.label;
  el("column").textContent = key;
  el("description").textContent = metric.description;
  for (const b of el("ranges").children) {
    const days = Number(b.dataset.days);
    const span = windowEnd - windowStart;
    const full = span === afterLastDay - firstDay;
    b.setAttribute("aria-pressed", String(full ? days === 0 : span === days));
  }
  const prsByDate = new Map();
  for (const pr of performancePRs) {
    if (!pr.date) continue;
    if (!prsByDate.has(pr.date)) prsByDate.set(pr.date, []);
    prsByDate.get(pr.date).push(pr);
  }
  const [factor, unit] = scale(key);
  const historyValues = history.map((row) => {
    const value = valueOf(row, key);
    return value === null ? null : value * factor;
  });
  const recorded = historyValues.filter((value) => value !== null);
  const canLog = recorded.length > 0 && recorded.every((value) => value > 0);
  const logScale = preferLog && canLog;
  el("log-scale").disabled = !canLog;
  el("log-scale").checked = logScale;
  const fullRange = axisRange(recorded, logScale);
  drawOverview(historyValues, fullRange, logScale, prsByDate);
  const dates = [],
    values = [],
    tips = [];
  let count = 0;
  for (const [i, row] of visible.entries()) {
    if (i && Date.parse(row.date) - Date.parse(visible[i - 1].date) > dayMs) {
      dates.push(new Date(Date.parse(row.date) - dayMs).toISOString().slice(0, 10));
      values.push(null);
      tips.push("");
    }
    const value = valueOf(row, key);
    dates.push(row.date);
    values.push(value === null ? null : value * factor);
    const prs = prsByDate.get(row.date);
    tips.push(
      readout(row, key) +
        " · " +
        objective(row) +
        (prs ? "<br>" + prs.map((pr) => "#" + pr.number).join(", ") : ""),
    );
    if (value !== null) count++;
  }
  el("extent").textContent =
    dateLabel(isoDay(windowStart)) +
    " – " +
    dateLabel(isoDay(windowEnd - 1)) +
    " · " +
    count +
    " of " +
    visible.length +
    " days recorded";
  const annotations = [],
    shapes = [];
  for (let i = 1; i < history.length; i++) {
    const row = history[i];
    if (utcDay(row.date) < windowStart || utcDay(row.date) >= windowEnd) continue;
    const previous = history[i - 1];
    const changes = [];
    if (objective(row) !== objective(previous)) changes.push("Objective");
    if (
      row.tightening_algorithm &&
      previous.tightening_algorithm &&
      row.tightening_algorithm !== previous.tightening_algorithm
    )
      changes.push("Tightening");
    if (!changes.length) continue;
    shapes.push({
      type: "line",
      x0: row.date,
      x1: row.date,
      y0: 0,
      y1: 1,
      yref: "paper",
      line: { color: "#758078", width: 1, dash: "dot" },
      layer: "below",
    });
    annotations.push({
      x: row.date,
      y: 1.02,
      yref: "paper",
      text: changes.join(" + ") + " changed",
      yanchor: "bottom",
      showarrow: false,
      font: { size: 12, color: "#52665e" },
    });
  }
  const valid = values.filter((v) => v !== null);
  if (!valid.length)
    annotations.push({
      xref: "paper",
      yref: "paper",
      x: 0.5,
      y: 0.5,
      text: "No recorded values in this period",
      showarrow: false,
    });
  const fixed = el("fixed-axis").checked;
  const domainValues = fixed ? recorded : valid;
  const max = domainValues.length ? Math.max(...domainValues) : null;
  const axis = {
    title: { text: unit || "Count", font: { size: 13 } },
    type: logScale ? "log" : "linear",
    range: fixed || !valid.length ? fullRange : axisRange(valid, logScale),
    fixedrange: true,
    gridcolor: "#e0e7e1",
    zeroline: false,
    nticks: 6,
  };
  // Plotly abbreviates minor log ticks to their leading digit unless a format is set.
  if (logScale) axis.tickformat = "~g";
  if (max === 0) axis.tickvals = [0, 1];
  else if (!logScale && !unit && max !== null && max <= 5) axis.dtick = 1;
  const chart = el("graph");
  Plotly.react(
    chart,
    [
      {
        x: dates,
        y: values,
        text: tips,
        customdata: dates,
        type: "scatter",
        mode: "lines+markers",
        line: { width: 2.1, color: "#006b50" },
        marker: {
          size: dates.map((date) => (prsByDate.has(date) ? 9 : 3.4)),
          color: dates.map((date) => (prsByDate.has(date) ? "#704392" : "#006b50")),
          symbol: dates.map((date) => (prsByDate.has(date) ? "diamond" : "circle")),
          opacity: 1,
        },
        connectgaps: false,
        hovertemplate: "%{x|%d %b %Y}<br>%{text}<extra></extra>",
      },
    ],
    {
      autosize: true,
      margin: { l: 56, r: 12, t: 33, b: 38 },
      font: { family: "Instrument, sans-serif", size: 13, color: "#40584c" },
      hoverlabel: { bgcolor: "#164c3d", font: { color: "#fff" } },
      hovermode: "closest",
      hoverdistance: 25,
      dragmode: false,
      showlegend: false,
      xaxis: {
        type: "date",
        range: [
          new Date((windowStart - 0.5) * dayMs).toISOString(),
          new Date((windowEnd - 0.5) * dayMs).toISOString(),
        ],
        fixedrange: true,
        showgrid: false,
        nticks: chart.clientWidth < 550 ? 4 : 8,
        tickformat: windowEnd - windowStart <= 90 ? "%d %b" : "%b %Y",
        zeroline: false,
      },
      yaxis: axis,
      shapes,
      annotations,
      paper_bgcolor: "#fff",
      plot_bgcolor: "#fff",
    },
    { displayModeBar: false, responsive: true, scrollZoom: false },
  ).then(() => {
    if (!bound) {
      chart.on("plotly_click", (e) => {
        const day = e.points[0].customdata;
        if (visible.some((r) => r.date === day)) {
          selected = day;
          activePR = null;
          activeDay = day;
          inspect();
          drawPRRail();
        }
      });
      bound = true;
    }
  });
  inspect();
  drawPRRail();
}
el("ranges").addEventListener("click", (e) => {
  if (e.target.dataset.days !== undefined) {
    if (!history.length) return;
    presetWindow(Number(e.target.dataset.days));
    renderAndShare();
  }
});
el("log-scale").addEventListener("change", (event) => {
  preferLog = event.target.checked;
  renderAndShare();
});
el("fixed-axis").addEventListener("change", renderAndShare);
el("metric").addEventListener("change", renderAndShare);
el("day").addEventListener("change", (e) => {
  selected = e.target.value;
  activeDay = null;
  activePR = null;
  inspect();
  drawPRRail();
});
async function get(url, type = "text") {
  const response = await fetch(url, { cache: "no-cache", signal: AbortSignal.timeout(20000) });
  if (!response.ok) throw Error(response.status);
  return response[type]();
}
async function start() {
  try {
    history = await get(raw + "/daily.json", "json");
    if (!history.length) throw Error("Empty archive");
    firstDay = utcDay(history[0].date);
    afterLastDay = utcDay(history.at(-1).date) + 1;
    restoreView();
    render(true);
    shareView();
    el("status").textContent = "GitHub archive · " + history.length + " daily runs";
  } catch {
    el("latest").textContent = "Unavailable";
    el("status").textContent =
      "Could not load the chart. Reload to retry or open the run files on GitHub.";
  }
}
function wireTermDefinitions() {
  let active;
  function show(term) {
    if (active) active.querySelector('[role="tooltip"]').hidden = true;
    active = term;
    if (active) active.querySelector('[role="tooltip"]').hidden = false;
  }
  for (const term of document.querySelectorAll(".benchmark-terms .term")) {
    const button = term.querySelector("button");
    term.addEventListener("pointerenter", (event) => {
      if (event.pointerType === "mouse") show(term);
    });
    term.addEventListener("pointerleave", () => {
      if (active === term && document.activeElement !== button) show(null);
    });
    button.addEventListener("focus", () => show(term));
    button.addEventListener("click", () => show(term));
    button.addEventListener("blur", () => {
      if (active === term) show(null);
    });
  }
  document.addEventListener("pointerdown", (event) => {
    if (active && !active.contains(event.target)) show(null);
  });
  document.addEventListener("keydown", (event) => {
    if (active && event.key === "Escape") show(null);
  });
}
wireTermDefinitions();
wireOverview();
startPRs(start());
