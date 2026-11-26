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
const dayMs = 86400000;
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
let history = [],
  visible = [],
  days = 0,
  selected = "",
  bound = false;
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
  const cutoff = days ? Date.parse(latest.date) - (days - 1) * dayMs : -Infinity;
  visible = history.filter((r) => Date.parse(r.date) >= cutoff);
  if (resetSelection || !visible.some((r) => r.date === selected)) selected = latest.date;
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
  for (const b of el("ranges").children)
    b.setAttribute("aria-pressed", String(Number(b.dataset.days) === days));
  const [factor, unit] = scale(key);
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
    tips.push(readout(row, key) + " · " + objective(row));
    if (value !== null) count++;
  }
  el("extent").textContent =
    dateLabel(visible[0].date) +
    " – " +
    dateLabel(latest.date) +
    " · " +
    count +
    " of " +
    visible.length +
    " days recorded";
  const annotations = [],
    shapes = [];
  for (let i = 1; i < history.length; i++) {
    const row = history[i];
    if (row.date < visible[0].date) continue;
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
  const flatZero = valid.length && valid.every((v) => v === 0);
  const axis = {
    title: { text: unit || "Count", font: { size: 13 } },
    rangemode: "tozero",
    fixedrange: true,
    gridcolor: "#e0e7e1",
    zeroline: false,
    nticks: 6,
  };
  if (flatZero) Object.assign(axis, { range: [-0.12, 1], tickvals: [0, 1] });
  if (!unit && valid.length && Math.max(...valid) <= 5 && !flatZero) axis.dtick = 1;
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
        marker: { size: 3.4, color: "#006b50" },
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
          new Date(Date.parse(visible[0].date) - dayMs * 0.6),
          new Date(Date.parse(latest.date) + dayMs * 0.6),
        ],
        fixedrange: true,
        showgrid: false,
        nticks: chart.clientWidth < 550 ? 4 : 8,
        tickformat: days ? "%d %b" : "%b %Y",
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
          inspect();
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
    days = Number(e.target.dataset.days);
    render();
  }
});
el("metric").addEventListener("change", () => render());
el("day").addEventListener("change", (e) => {
  selected = e.target.value;
  inspect();
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
    render(true);
    el("status").textContent = "GitHub archive · " + history.length + " daily runs";
  } catch {
    el("latest").textContent = "Unavailable";
    el("status").textContent =
      "Could not load the chart. Reload to retry or open the run files on GitHub.";
  }
}
startPRs(start());
