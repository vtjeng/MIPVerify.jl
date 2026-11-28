// The selected input interval and probe are shared by all four plots.
(() => {
  const root = document.getElementById("relu-geometry");
  const interval = root.querySelector("select");
  const coefficient = root.querySelector("#geometry-coeff");
  const probe = root.querySelector("#geometry-probe");
  const readout = root.querySelector("output");
  const panels = Array.from(root.querySelectorAll("[data-shape]"));
  const relu = z => Math.max(z, 0);
  const fmt = figures.nearest;

  const outputTerm = (edge, input) => {
    const h = edge ? '<msub><mi>h</mi><mtext>' + edge + '</mtext></msub>' : '<mi>h</mi>';
    const arg = input === undefined ? '<mi>z</mi>' : '<mn>' + fmt(input) + '</mn>';
    return '<mrow data-relu="' + (edge || 'actual') + '">' + h + '<mo>(</mo>' + arg + '<mo>)</mo></mrow>';
  };
  const math = terms => "<math>" + terms + "</math>";
  const outputSymbol = edge => math(outputTerm(edge));
  const relaxedSymbol = document.getElementById("relaxed-output-symbol").cloneNode(true);
  relaxedSymbol.removeAttribute("id");
  const relaxedMath = relaxedSymbol.outerHTML;
  const relaxedTerm = '<mrow data-relu="relaxed">' + relaxedSymbol.innerHTML + '</mrow>';
  const roleFormat = {actual: fmt, upper: figures.upper, lower: figures.lower};
  const valueMarkup = (role, value) => '<mn data-relu="' + role + '">' + roleFormat[role](value) + '</mn>';
  const rangeSwatch = '<path class="relu-range" d="M12 1V17"/>';
  const keyEntries = {
    actual: ['<path class="relu-actual" d="M0 9H26"/>', 'Actual ' + outputSymbol()],
    upper: ['<path class="relu-upper" d="M0 9H26"/>', 'Upper ' + outputSymbol('upper')],
    lower: ['<path class="relu-lower" d="M0 9H26"/>', 'Lower ' + outputSymbol('lower')],
    range: [rangeSwatch, 'Range of ' + relaxedMath],
    slider: [rangeSwatch + '<path class="relu-diamond" d="M12 4l5 5-5 5-5-5Z"/>', 'Range and selected ' + relaxedMath],
    window: ['<rect x="1" y="3" width="24" height="12" fill="var(--soft)" stroke="var(--rule)"/>', 'Allowed inputs'],
  };
  for (const key of document.querySelectorAll('[data-relu-key]')) {
    key.setAttribute('role', 'group');
    key.setAttribute('aria-label', 'Plot legend');
    key.innerHTML = key.dataset.reluKey.split(' ').map(name => {
      const [mark, label] = keyEntries[name];
      return '<span><svg aria-hidden="true" viewBox="0 0 26 18">' + mark + '</svg>' + label + '</span>';
    }).join('');
  }

  // Overlap the coefficient wordings so wrapping never shifts the plots below.
  function showCoefficientText(container, selected, wording) {
    container.classList.add("budget-caption");
    container.replaceChildren(...Array.from(coefficient.options, option => {
      const value = Number(option.value);
      const span = document.createElement("span");
      if (value !== selected) span.setAttribute("aria-hidden", "true");
      span.innerHTML = wording(value);
      return span;
    }));
  }

  function decisionText(lo, hi, c) {
    const unstable = lo < 0 && hi > 0;
    if (c === 0) {
      return "This neuron’s paths cancel in the logit difference. Its contribution is zero for every allowed output.";
    } else if (!unstable) {
      return "The ReLU stays " + (lo >= 0 ? "active, so h(z) = z" : "off, so h(z) = 0") +
        ". The adaptive rule selects the exact line. The highlighted edge bounds " + (c > 0 ? "+h̃." : "−h̃.");
    } else if (c > 0) {
      return "Adding h̃ makes a larger output worse for this target. The adaptive method uses the upper chord through the two endpoints. The lower slope has no effect on this term.";
    } else {
      const diagonal = hi >= -lo;
      return "Subtracting h̃ makes a smaller output worse for this target. The adaptive method uses " +
        (diagonal ? "the diagonal lower line." : "the horizontal lower line.") + " The diagonal can underestimate by up to " +
        fmt(-lo) + " on the negative side. The horizontal line can underestimate by up to " + fmt(hi) +
        " on the positive side. " + (hi === -lo ? "On a tie it takes the diagonal." : "It takes the smaller worst-case error.");
    }
  }

  function render() {
    const [lo, hi] = interval.value.split(",").map(Number);
    const z = Number(probe.value);
    const c = Number(coefficient.value);
    readout.innerHTML = math("<mi>z</mi><mo>=</mo><mn>" + fmt(z) + "</mn>") + " · " +
      math(outputTerm() + "<mo>=</mo>" + valueMarkup("actual", relu(z)));
    drawShapes(panels, lo, hi, z, c);
  }

  function drawShapes(panels, lo, hi, z, c) {
    const unstable = lo < 0 && hi > 0;
    const slope = (relu(hi) - relu(lo)) / (hi - lo);
    const chord = x => relu(lo) + slope * (x - lo);
    const adaptiveShape = hi >= -lo ? "diagonal" : "horizontal";
    const pixelsPerUnit = 40;
    const xPos = x => 116 + x * pixelsPerUnit;
    // Both candidate lines must fit, including h = z at the input −2.
    const yPos = h => 112 - h * pixelsPerUnit;
    const coord = (x, h) => xPos(x) + "," + yPos(h);
    const bends = unstable ? [lo, 0, hi] : [lo, hi];
    const graph = bends.map(x => coord(x, relu(x))).join(" ");
    for (const panel of panels) {
      const shape = panel.dataset.shape;
      let lower = relu, upper = unstable ? chord : relu;
      if (shape === "interval") {
        lower = () => relu(lo);
        upper = () => relu(hi);
      } else if (shape === "diagonal" || shape === "horizontal") {
        const diagonal = shape === "diagonal";
        lower = diagonal ? x => x : () => 0;
        const error = diagonal ? Math.max(0, -lo) : Math.max(0, hi);
        const caption = panel.querySelector("[data-choice]");
        if (caption) {
          const message = document.createElement("strong");
          const chosen = shape === adaptiveShape;
          panel.classList.toggle("alternative-line", !chosen);
          message.textContent = chosen ? "Selected by the adaptive rule. " : "Alternative lower line. ";
          const detail = document.createElement("span");
          setInlineMathText(detail, error === 0 ? "It matches the ReLU throughout this interval." :
            "Its largest underestimate is " + fmt(error) + " at z = " + fmt(diagonal ? lo : hi) + ".");
          caption.replaceChildren(message, detail);
        }
      }
      const lowerEdge = bends.map(x => coord(x, lower(x))).join(" ");
      const upperEdge = [coord(lo, upper(lo)), coord(hi, upper(hi))].join(" ");
      const boundary = [coord(hi, upper(hi)), coord(lo, upper(lo)), lowerEdge].join(" ");
      const selected = c > 0 ? upper : lower;
      const contribution = c === 0 ? 0 : c * selected(z);
      const svg = panel.querySelector("svg");
      const relaxedRange = "Relaxed " + math(relaxedTerm + "<mo>∈</mo><mo>[</mo>" +
        valueMarkup("lower", lower(z)) + "<mo>,</mo>" + valueMarkup("upper", upper(z)) + "<mo>]</mo>");
      const actual = "Actual h(" + fmt(z) + ") = " + fmt(relu(z));
      const actualReading = "Actual " + math(outputTerm(null, z) + "<mo>=</mo>" + valueMarkup("actual", relu(z)));
      const output = panel.querySelector("output");
      if (c == null) output.innerHTML = actualReading + ". " + relaxedRange;
      else showCoefficientText(output, c, value => {
        const bound = value > 0 ? valueMarkup("upper", upper(z)) : "<mn>" + figures.upper(value * lower(z)) + "</mn>";
        const weight = value === 1 ? "<mo>+</mo>" : value === -1 ? "<mo>−</mo>" : "<mn>0</mn>";
        return relaxedRange + " · " + math(weight + relaxedTerm + "<mo>≤</mo>" + bound);
      });
      svg.setAttribute("aria-label", panel.querySelector("h4").textContent + ". " + actual +
        ". The relaxation permits outputs from " + figures.lower(lower(z)) + " to " + figures.upper(upper(z)) + "." +
        (c == null ? "" : " Upper bound on contribution " + figures.upper(contribution) + "."));
      svg.innerHTML = `
        <path d="M22 ${yPos(0)}H207 M${xPos(0)} 200V20" fill="none" stroke="var(--muted)"/>
        <g fill="var(--muted)" font-size="12"><text class="math-axis" x="205" y="126">𝑧</text><text x="116" y="15" text-anchor="middle">Output</text>
          <text x="${xPos(0)}" text-anchor="middle" y="213">0</text><text x="${xPos(-2)}" y="213" text-anchor="middle">−2</text>
          <text x="${xPos(2)}" y="213" text-anchor="middle">2</text><text x="9" y="${yPos(2)+4}">2</text><text x="5" y="${yPos(-2)+4}">−2</text></g>
        <polygon points="${boundary}" stroke="var(--rule)" fill="var(--relaxed-output)" fill-opacity=".10"/>
        <polyline points="${graph}" class="relu-actual"/>
        <polyline points="${upperEdge}" class="relu-upper${c > 0 ? ' relu-used' : ''}"/>
        <polyline points="${lowerEdge}" class="relu-lower${c < 0 ? ' relu-used' : ''}"/>
        <line x1="${xPos(z)}" x2="${xPos(z)}" y1="${yPos(lower(z))}" y2="${yPos(upper(z))}" class="relu-range"/>
        <circle cx="${xPos(z)}" cy="${yPos(relu(z))}" r="3.6" class="relu-dot"/>`;
      if (panel.closest("#interval-rectangle")) {
        addRelaxedSlider(svg, output, z, lower(z), upper(z), xPos(z), yPos, actualReading);
      }
    }
  }

  function addRelaxedSlider(svg, output, z, min, max, x, yPos, actualReading) {
    const top = yPos(max), bottom = yPos(min);
    svg.insertAdjacentHTML("beforeend", `
      <g class="relaxed-slider" role="slider" tabindex="0" aria-label="Relaxed output at fixed input ${fmt(z)}"
         aria-orientation="vertical" aria-valuemin="${min}" aria-valuemax="${max}">
        <rect x="${x-18}" y="${top-9}" width="36" height="${bottom-top+18}" fill="transparent"/>
        <path class="relaxed-diamond relu-diamond" d="M0 -8 8 0 0 8 -8 0Z"/>
      </g>`);
    const slider = svg.querySelector(".relaxed-slider");
    const diamond = slider.querySelector(".relaxed-diamond");
    output.innerHTML = actualReading + "<br>Selected " +
      math(relaxedTerm + '<mo>=</mo><mn data-relu="relaxed"></mn>');
    const selectedNumber = output.querySelector('mn[data-relu="relaxed"]');
    let selected;
    function select(value) {
      selected = Math.max(min, Math.min(max, Math.round(value * 100) / 100));
      diamond.setAttribute("transform", `translate(${x} ${yPos(selected)})`);
      slider.setAttribute("aria-valuenow", selected);
      selectedNumber.textContent = selected.toFixed(2);
    }
    function selectAtPointer(event) {
      const point = new DOMPoint(event.clientX, event.clientY).matrixTransform(svg.getScreenCTM().inverse());
      select(min + (bottom - point.y) / (bottom - top) * (max - min));
    }
    slider.addEventListener("pointerdown", event => {
      if (!event.isPrimary || event.button !== 0) return;
      event.preventDefault();
      slider.focus({preventScroll:true});
      slider.setPointerCapture(event.pointerId);
      selectAtPointer(event);
    });
    slider.addEventListener("pointermove", event => {
      if (slider.hasPointerCapture(event.pointerId)) selectAtPointer(event);
    });
    slider.addEventListener("keydown", event => {
      const choices = {
        ArrowUp: selected + .01, ArrowRight: selected + .01,
        ArrowDown: selected - .01, ArrowLeft: selected - .01,
        PageUp: selected + .1, PageDown: selected - .1, Home: min, End: max,
      };
      if (!Object.hasOwn(choices, event.key)) return;
      event.preventDefault();
      select(choices[event.key]);
    });
    select((min + max) / 2);
  }
  const decisionForms = new Set();
  for (const range of interval.options) {
    const [lo, hi] = range.value.split(",").map(Number);
    for (const option of coefficient.options) {
      decisionForms.add(decisionText(lo, hi, Number(option.value)));
    }
  }
  const decision = reserveBudgetCaption("geometry-decision", ...decisionForms);

  function showCoefficient() {
    const [lo, hi] = interval.value.split(",").map(Number);
    setInlineMathText(decision, decisionText(lo, hi, Number(coefficient.value)));
    render();
  }

  function changeInterval() {
    const [lo, hi] = interval.value.split(",").map(Number);
    const previous = Number(probe.value);
    probe.min = lo;
    probe.max = hi;
    probe.value = Math.min(hi, Math.max(lo, previous));
    showCoefficient();
  }
  interval.addEventListener("change", changeInterval);
  coefficient.addEventListener("change", showCoefficient);
  probe.addEventListener("input", render);
  for (const panel of document.querySelectorAll("[data-shape-example] [data-shape]")) {
    const input = Number(panel.dataset.exampleInput ?? 0.5);
    drawShapes([panel], -1, 2, input, null);
  }
  changeInterval();
})();
