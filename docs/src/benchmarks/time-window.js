"use strict";

// Window boundaries are UTC day numbers. The end is exclusive, so even a
// one-day window has a width in the overview.
const dayMs = 86400000;
const utcDay = (date) => Date.parse(date) / dayMs;
const isoDay = (day) => new Date(day * dayMs).toISOString().slice(0, 10);
let firstDay, afterLastDay, windowStart, windowEnd;

function presetWindow(days) {
  const total = afterLastDay - firstDay;
  const span = days ? Math.min(days, total) : total;
  // Keep the right edge unless the interval would start before the archive.
  windowEnd = Math.max(windowEnd ?? afterLastDay, firstDay + span);
  windowStart = windowEnd - span;
}

function drawOverview(values, [low, high], logScale, prsByDate) {
  const total = afterLastDay - firstDay;
  const x = (day) => ((day - firstDay + 0.5) / total) * 1000;
  let path = "";
  for (const [i, row] of history.entries()) {
    if (values[i] === null) continue;
    const day = utcDay(row.date);
    const continued = i && values[i - 1] !== null && utcDay(history[i - 1].date) === day - 1;
    const height = ((logScale ? Math.log10(values[i]) : values[i]) - low) / (high - low);
    path += (continued ? "L" : "M") + x(day) + " " + (35 - 32 * height) + " ";
  }
  el("overview-line").setAttribute("d", path);
  el("overview-prs").setAttribute(
    "d",
    history
      .filter((r) => prsByDate.has(r.date))
      .map((r) => "M" + x(utcDay(r.date)) + " 38v5")
      .join(" "),
  );
  el("overview-first").textContent = dateLabel(history[0].date);
  el("overview-last").textContent = dateLabel(history.at(-1).date);
  const strip = el("overview");
  strip.style.setProperty("--start", ((windowStart - firstDay) / total) * 100 + "%");
  strip.style.setProperty("--end", ((windowEnd - firstDay) / total) * 100 + "%");
  for (const [id, value, min, max, text] of [
    ["window-start", windowStart, firstDay, windowEnd - 1, dateLabel(isoDay(windowStart))],
    ["window-end", windowEnd - 1, windowStart, afterLastDay - 1, dateLabel(isoDay(windowEnd - 1))],
    [
      "window-pan",
      windowStart,
      firstDay,
      afterLastDay - (windowEnd - windowStart),
      dateLabel(isoDay(windowStart)) + " – " + dateLabel(isoDay(windowEnd - 1)),
    ],
  ]) {
    const control = el(id);
    control.setAttribute("aria-valuemin", min);
    control.setAttribute("aria-valuemax", max);
    control.setAttribute("aria-valuenow", value);
    control.setAttribute("aria-valuetext", text);
  }
}

function wireOverview() {
  const strip = el("overview");
  const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
  let drag, frame;
  function move(mode, delta, start, end) {
    if (mode === "window-start") windowStart = clamp(start + delta, firstDay, end - 1);
    else if (mode === "window-end") windowEnd = clamp(end + delta, start + 1, afterLastDay);
    else {
      windowStart = clamp(start + delta, firstDay, afterLastDay - (end - start));
      windowEnd = windowStart + end - start;
    }
    if (!frame)
      frame = requestAnimationFrame(() => {
        frame = 0;
        render();
      });
  }
  strip.addEventListener("keydown", (event) => {
    if (!history.length || event.altKey || event.ctrlKey || event.metaKey) return;
    const step = event.shiftKey ? 7 : 1;
    const delta = {
      ArrowLeft: -step,
      ArrowRight: step,
      Home: firstDay - afterLastDay,
      End: afterLastDay - firstDay,
    }[event.key];
    if (delta === undefined) return;
    event.preventDefault();
    move(event.target.id, delta, windowStart, windowEnd);
    if (!event.repeat) shareView();
  });
  strip.addEventListener("keyup", shareView);
  strip.addEventListener("pointerdown", (event) => {
    if (!history.length || !event.isPrimary || event.button !== 0) return;
    const box = strip.getBoundingClientRect();
    const perPixel = (afterLastDay - firstDay) / box.width;
    const control = event.target.closest('[role="slider"]');
    const mode = control?.id || "window-pan";
    if (!control) {
      const center = firstDay + (event.clientX - box.left) * perPixel;
      move(mode, Math.round(center - (windowStart + windowEnd) / 2), windowStart, windowEnd);
    }
    drag = { mode, x: event.clientX, perPixel, start: windowStart, end: windowEnd };
    strip.setPointerCapture(event.pointerId);
    el(mode).focus({ preventScroll: true });
  });
  strip.addEventListener("pointermove", (event) => {
    if (!drag || !strip.hasPointerCapture(event.pointerId)) return;
    move(drag.mode, Math.round((event.clientX - drag.x) * drag.perPixel), drag.start, drag.end);
  });
  strip.addEventListener("lostpointercapture", () => {
    drag = null;
    shareView();
  });
}
