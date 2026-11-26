"use strict";

let performancePRs = [];
let activePR = null;
const api = "https://api.github.com/repos/vtjeng/MIPVerify.jl";
function prLink(pr, short = false) {
  const link = document.createElement("a");
  link.href = repo + "/pull/" + pr.number;
  link.textContent = "#" + pr.number + (short ? "" : " " + pr.title);
  return link;
}
function prList(items) {
  return items.map((pr) => {
    const li = document.createElement("li"),
      note = document.createElement("small");
    const when = pr.date
      ? dateLabel(pr.date)
      : !pr.merged
        ? "Unmerged"
        : pr.unverified
          ? "First nightly run unverified"
          : "Awaiting first archived run";
    li.append(prLink(pr), note);
    note.textContent = when + (pr.note ? " · " + pr.note : "");
    return li;
  });
}
function showRunPRs(row) {
  const prs = performancePRs.filter((pr) => pr.date === row.date);
  const panel = el("run-prs");
  panel.hidden = !prs.length;
  panel.replaceChildren("First archived run with ");
  prs.forEach((pr, i) => panel.append(i ? ", " : "", prLink(pr, true)));
}
function drawPRRail() {
  if (!visible.length) return;
  const track = el("pr-track"),
    width = track.clientWidth;
  const first = visible[0].date,
    last = visible.at(-1).date;
  const start = Date.parse(first) - 0.6 * dayMs,
    end = Date.parse(last) + 0.6 * dayMs;
  const candidates = performancePRs
    .filter((p) => p.date >= first && p.date <= last)
    .sort((a, b) => a.date.localeCompare(b.date) || a.number - b.number);
  const clusters = [];
  for (const pr of candidates) {
    const x = ((Date.parse(pr.date) - start) / (end - start)) * width;
    const cluster = clusters.at(-1);
    if (cluster && x - cluster.last < 62) {
      cluster.items.push(pr);
      cluster.last = x;
    } else clusters.push({ first: x, last: x, items: [pr] });
  }
  const buttons = [],
    panel = el("pr-selection");
  panel.hidden = true;
  for (const group of clusters) {
    const { items } = group,
      button = document.createElement("button");
    button.textContent = items.length === 1 ? "#" + items[0].number : items.length + " PRs";
    button.title = items
      .map((p) => dateLabel(p.date) + " · #" + p.number + " " + p.title)
      .join("\n");
    button.setAttribute("aria-label", button.title);
    button.setAttribute("aria-controls", "pr-selection");
    const expanded = items.some((pr) => pr.number === activePR);
    button.setAttribute("aria-expanded", String(expanded));
    button.style.left =
      Math.max(0, Math.min(width - 56, (group.first + group.last) / 2 - 28)) + "px";
    button.style.width = "56px";
    if (expanded) {
      const ul = document.createElement("ul");
      ul.append(...prList(items));
      panel.replaceChildren(ul);
      panel.hidden = false;
    }
    button.onclick = () => {
      activePR = expanded ? null : items[0].number;
      drawPRRail();
      track.querySelectorAll("button")[clusters.indexOf(group)]?.focus({ preventScroll: true });
    };
    buttons.push(button);
  }
  if (panel.hidden) activePR = null;
  track.replaceChildren(...buttons);
}
function updatePRDisplay() {
  el("pr-list").replaceChildren(...prList(performancePRs));
  drawPRRail();
  if (visible.length) inspect();
}
async function firstNight(pr) {
  if (!pr.merged) return;
  if (pr.date && history.some((r) => r.date === pr.date && r.commit_sha === pr.runSha)) return;
  delete pr.date;
  const runs = history.filter((r) => r.date >= pr.merged.slice(0, 10));
  if (!runs.length) return;
  if (!pr.sha) pr.sha = (await get(api + "/pulls/" + pr.number, "json")).merge_commit_sha;
  if (!pr.sha) throw Error("Merge commit unavailable");
  const checked = new Set();
  for (const run of runs) {
    if (checked.has(run.commit_sha)) continue;
    checked.add(run.commit_sha);
    const status =
      run.commit_sha === pr.sha
        ? "identical"
        : (await get(api + "/compare/" + pr.sha + "..." + run.commit_sha + "?per_page=1", "json"))
            .status;
    if (status === "ahead" || status === "identical") {
      pr.date = run.date;
      pr.runSha = run.commit_sha;
      return;
    }
  }
}
async function startPRs(runsReady) {
  const snapshot = await get("performance-prs.json", "json");
  performancePRs = snapshot;
  updatePRDisplay();
  new ResizeObserver(drawPRRail).observe(el("pr-track"));
  await runsReady;
  try {
    const byNumber = new Map(snapshot.map((pr) => [pr.number, pr])),
      tagged = [];
    for (let page = 1; ; page++) {
      const list = await get(
        api + "/issues?state=all&labels=verification-performance&per_page=100&page=" + page,
        "json",
      );
      for (const issue of list.filter((i) => i.pull_request))
        tagged.push({
          ...byNumber.get(issue.number),
          number: issue.number,
          title: issue.title,
          merged: issue.pull_request.merged_at,
        });
      if (list.length < 100) break;
    }
    for (const pr of tagged) {
      try {
        await firstNight(pr);
      } catch {
        pr.unverified = true;
      }
    }
    performancePRs = tagged.sort((a, b) => a.number - b.number);
    el("pr-source").textContent = "· GitHub labels";
  } catch {
    el("pr-source").textContent = "· Bundled label snapshot";
    for (const pr of performancePRs.filter((p) => p.merged && !p.date)) {
      try {
        await firstNight(pr);
      } catch {
        pr.unverified = true;
      }
    }
  }
  updatePRDisplay();
}
