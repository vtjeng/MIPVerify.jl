(() => {
  const termDefinitions = {
    corner: "A corner of the pixel box is an image in which each pixel is at its lowest or highest allowed brightness.",
    backCorner: "A candidate image made by setting each pixel to its brightest or darkest allowed value, whichever makes the rival’s back-substitution bound largest. The verifier tests that image on the network.",
    adversarialExample: "In this benchmark, an adversarial example is any image within the pixel budget where another label ties or beats the true label’s score. The original image counts if it already meets that condition.",
    relaxation: "A relaxation weakens a rule so that every solution to the original problem is still allowed. For a ReLU, the exact output at z = 0.5 is h(0.5) = 0.5. A relaxed rule can also permit h̃ = 1 at that same input. A bound that holds for all the relaxed choices also holds for the actual output.",
    relu: "A ReLU takes an input z. Its output h(z) is the larger of zero and z. For example, h(−0.3) = 0 and h(0.3) = 0.3.",
    piecewiseAffine: "The network divides its possible inputs into regions. Within each region, each output score is a weighted sum of the input values plus a constant. The weights and the constant can differ from one region to another.",
    reluBinary: "A binary variable takes one of two values, 0 or 1. Here it records which rule a ReLU follows. In the off state the output is zero. In the on state the output equals the input.",
    allowedChange: "To perturb an image is to change its pixel brightnesses within specified limits. For example, if a pixel may change by 0.1, a brightness of 0.50 can move anywhere from 0.40 to 0.60. Brightness stays between 0 and 1, so a pixel starting at 0.95 can reach at most 1.",
    toyModel: "A toy model is a simplified network small enough to follow one calculation at a time. This one has two input pixels and two hidden neurons. Its weights and biases come from Raghunathan, Steinhardt and Liang’s trained digit classifier and are rounded to three significant figures.",
    toyImage: "Here an image has just two pixels. The pair (x₀, x₁) gives their brightnesses, so each point on a map represents one complete image.",
    interval: "Interval arithmetic tracks a minimum and a maximum for each quantity. For example, if two pixels can each range from 0.4 to 0.6, subtracting one brightness from the other gives a range from −0.2 to 0.2.",
    logitDifference: "A logit is the score the classifier assigns to a label. Subtract the correct label’s score from a competing label’s score to get their logit difference. A positive difference means the competing label scores higher.",
    witness: "A witness is an image within the pixel budget for which we have checked that a rival ties or beats the true digit on the original network.",
    pgd: "Projected gradient descent (PGD) searches for an allowed change that raises a rival’s score relative to the true digit’s. Each step uses the gradient to choose which pixels to brighten or darken, then clips each brightness to its allowed range.",
    gapLp: "Target LP uses a linear program to bound how far a competing digit’s score can rise above the correct digit’s score. It maximizes this difference under linear constraints that include every allowed image. A negative maximum rules out that competitor.",
    wk17a: "MIPVerify’s name for Wong and Kolter’s MNIST digit classifier. Two convolutional layers extract features from the image, followed by a fully connected hidden layer with 100 neurons.",
    mip: "Mixed-integer programming (MIP) is a way to solve problems with both continuous variables, such as pixel brightness, and integer variables, such as a 0-or-1 switch.",
    lp: "A linear program (LP) finds the largest or smallest value of a weighted sum subject to linear equations and inequalities.",
    image: "A digital image is a grid of pixels. In MNIST, a collection of handwritten digits, each image has 28 × 28 brightness values and a label giving its true digit.",
    rsl18a: "MIPVerify’s name for the MNIST digit classifier trained by Raghunathan, Steinhardt and Liang (2018). Its hidden layer has 500 ReLU neurons. The two-pixel example uses weights from two of them.",
    rival: "A rival is any digit other than the image’s correct digit. The verifier checks whether an allowed change to the image could make a rival tie or beat the correct digit’s score.",
    target: "The target is the particular rival the verifier is checking.",
  };

  // Match the longest name first so “Target LP” gets its method definition.
  const names = {
    wk17a: /WK17a/, rsl18a: /RSL18a/,
    gapLp: /Target LP/,
    lp: /linear program(?:ming|s)?(?: \(LP\))?|LPs?/,
    mip: /mixed[- ]integer program(?:ming|s)?(?: \(MIP\))?|MIPs?/,
    pgd: /projected gradient descent(?: \(PGD\))?|PGD/,
    logitDifference: /logit differences?/,
    interval: /interval arithmetic/,
    piecewiseAffine: /piecewise[- ]affine/,
    reluBinary: /binary variables?/,
    relaxation: /relaxations?/,
    relu: /rectified linear units?(?: \(ReLUs?\))?|ReLUs?/,
    witness: /witness(?:es)?/,
    adversarialExample: /adversarial examples?/,
    backCorner: /(?:back-substitution|candidate) corners?/,
    corner: /corners?/,
    toyModel: /toy (?:models?|networks?)/,
    allowedChange: /perturbing the image|perturbations?/,
    image: /(?:perturbed |the same )?images?/,
    rival: /rivals?/,
    target: /target(?: digits?)?s?/,
  };
  const pattern = new RegExp("\\b(?:" + Object.entries(names).map(
    ([key, name]) => "(?<" + key + ">" + name.source + ")"
  ).join("|") + ")(?!\\w)", "gi");
  const definitions = document.createElement("div");
  document.body.append(definitions);
  const article = document.querySelector("main");
  const nativeControl = "button, label, summary";
  const skip = "a[href], [data-mention], .term-help, .term-wrap, script, style, canvas, svg, annotation, option, [aria-hidden=true]";
  let active, tip, pinned = false;

  function definitionId(el, key) {
    let text = el.dataset.definition || termDefinitions[key];
    let id = "glossary-" + key;
    if (key === "logitDifference") {
      const sample = el.closest("#mnist");
      if (sample) {
        id += "-mnist";
        text += sample.dataset.digitPair;
      } else if (el.closest("#intervals, #ranges, #same-pixel, #relu, #target-lp, #bound-direction, #bound-surfaces, #lp-comparison")) {
        id += "-toy";
        text += " Here that means digit 0’s logit minus digit 9’s.";
      }
    }
    // Aliases share a definition, but each tooltip names the exact term its reader chose.
    const label = el.textContent;
    id = el.dataset.tipId || id + "-" + encodeURIComponent(label);
    let entry = document.getElementById(id);
    if (!entry) {
      entry = document.createElement("div");
      entry.id = id;
      entry.className = "term-tip";
      entry.hidden = true;
      entry.setAttribute("role", "tooltip");
      const heading = document.createElement("strong");
      heading.textContent = label;
      const body = document.createElement("span");
      setInlineMathText(body, text);
      entry.append(heading, body);
      definitions.append(entry);
    } else if (key === "logitDifference" && entry.lastElementChild.textContent !== text) {
      setInlineMathText(entry.lastElementChild, text);
    }
    return id;
  }

  function register(term, key) {
    term.classList.add("term-help");
    term.dataset.glossary = key;
    term.setAttribute("aria-describedby", definitionId(term, key));
    if (!term.closest(nativeControl)) {
      term.tabIndex = 0;
      term.setAttribute("role", "button");
      term.setAttribute("aria-expanded", "false");
    }
  }

  function disableTip(term) {
    if (term === active) close();
    term.classList.remove("term-help", "term-introduction");
    delete term.dataset.glossary;
    for (const attr of ["aria-describedby", "aria-expanded", "role", "tabindex"]) term.removeAttribute(attr);
  }

  // Keep a control's own descriptions when moving its glossary tip.
  let controlTips = new Map();
  function updateControlDescriptions(next) {
    for (const control of new Set([...controlTips.keys(), ...next.keys()])) {
      const before = control.getAttribute("aria-describedby") || "";
      const refs = before.split(/\s+/).filter(id => id && !controlTips.get(control)?.has(id));
      const after = [...new Set([...refs, ...(next.get(control) || [])])].join(" ");
      if (before === after) continue;
      if (after) control.setAttribute("aria-describedby", after);
      else control.removeAttribute("aria-describedby");
    }
    controlTips = next;
  }

  function updateSpacing() {
    const terms = document.querySelectorAll("[data-mention]");
    const readings = [], labels = [];
    for (const term of terms) {
      if (!term.checkVisibility({visibilityProperty:true}) || !term.getClientRects().length) continue;
      const entry = {term, top:term.getBoundingClientRect().top};
      // Prefer the explanation in the text over a nearby menu or chart label.
      const prose = term.closest("p, h1, h2, h3, h4, li, figcaption")
        && !term.closest("nav, button, label, summary");
      (prose ? readings : labels).push(entry);
    }

    const selected = new Set(), positions = new Map();
    const height = window.visualViewport?.height || window.innerHeight;
    for (const {term, top} of [...readings, ...labels]) {
      const key = term.dataset.mention === "toyImage" ? "image" : term.dataset.mention;
      const earlier = positions.get(key) || [];
      if (earlier.some(y => Math.abs(y - top) < height)) continue;
      earlier.push(top);
      positions.set(key, earlier);
      selected.add(term);
    }

    const introduced = new Set(), nextControls = new Map();
    for (const term of terms) {
      if (!selected.has(term)) {
        if (term.dataset.glossary) disableTip(term);
        continue;
      }
      const key = term.dataset.mention;
      if (!term.dataset.glossary) register(term, key);
      const intro = !introduced.has(key) && names[key] && !term.closest("nav, footer");
      term.classList.toggle("term-introduction", Boolean(intro));
      if (intro) introduced.add(key);
      const control = term.closest(nativeControl);
      if (control) {
        const target = control.control || control;
        if (!nextControls.has(target)) nextControls.set(target, new Set());
        nextControls.get(target).add(term.getAttribute("aria-describedby"));
      }
    }
    updateControlDescriptions(nextControls);
    if (active) position();
  }

  let frame;
  function scheduleSpacing() {
    if (!frame) frame = requestAnimationFrame(() => {
      frame = 0;
      updateSpacing();
    });
  }

  function annotate(node) {
    const parent = node.parentElement;
    if (!parent || parent.closest(skip)) return;
    const matches = [...node.data.matchAll(pattern)].filter(match => !/^the same images?$/i.test(match[0]));
    if (!matches.length) return;
    const keyOf = match => Object.keys(match.groups).find(key => match.groups[key] !== undefined);
    if (parent.closest("math")) {
      if (matches[0][0] === node.data) parent.dataset.mention = keyOf(matches[0]);
      return;
    }
    const parts = document.createDocumentFragment();
    let end = 0;
    for (const match of matches) {
      const term = document.createElement("span");
      term.textContent = match[0];
      parts.append(node.data.slice(end, match.index), term);
      end = match.index + match[0].length;
    }
    parts.append(node.data.slice(end));
    const terms = [...parts.children];
    node.replaceWith(parts);
    // Context and associated form controls are available once the terms are in the DOM.
    terms.forEach((term, i) => {
      let key = keyOf(matches[i]);
      if (key === "image" && term.closest("#intervals, #ranges, #same-pixel, #relu, #bound-surfaces, #lp-comparison")) key = "toyImage";
      term.dataset.mention = key;
    });
  }

  function scan(root) {
    if (root.nodeType === Node.TEXT_NODE) return annotate(root);
    if (root.nodeType !== Node.ELEMENT_NODE || root.closest(skip)) return;
    const reader = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (reader.nextNode()) nodes.push(reader.currentNode);
    nodes.forEach(annotate);
  }

  function position() {
    if (!active) return;
    const rect = active.getBoundingClientRect();
    if (!active.isConnected || !rect.height || rect.bottom < 0 || rect.top > innerHeight) return close();
    const left = Math.max(8, Math.min(rect.left, document.documentElement.clientWidth - tip.offsetWidth - 8));
    const top = rect.bottom + tip.offsetHeight <= innerHeight - 8 ? rect.bottom : Math.max(8, rect.top - tip.offsetHeight);
    tip.style.left = left + "px";
    tip.style.top = top + "px";
  }
  function close() {
    if (tip) tip.hidden = true;
    if (active?.getAttribute("role") === "button") active.setAttribute("aria-expanded", "false");
    active = tip = null;
    pinned = false;
  }
  function show(term) {
    if (!term || term === active) return;
    close();
    active = term;
    tip = document.getElementById(definitionId(term, term.dataset.glossary));
    tip.hidden = false;
    if (term.getAttribute("role") === "button") term.setAttribute("aria-expanded", "true");
    position();
  }
  function from(target) {
    if (!(target instanceof Element) || target.closest("a[href]")) return null;
    return target.closest(".term-help") || target.closest("button,summary")?.querySelector(".term-help")
      || target.labels?.[0]?.querySelector(".term-help");
  }
  document.addEventListener("pointermove", e => {
    if (e.pointerType !== "mouse" || pinned || tip?.contains(e.target)) return;
    const term = from(e.target);
    if (term) show(term);
    else if (!active?.matches(":focus")) close();
  });
  document.addEventListener("focusin", e => {
    if (tip?.contains(e.target)) return;
    const term = from(e.target);
    if (term) show(term); else close();
  });
  document.addEventListener("click", e => {
    // A label forwards activation to its input. Handle that click once.
    if (e.target.closest("label")?.control && !e.target.matches("input,select")) return;
    const term = from(e.target);
    if (term) {
      if (term === active && pinned) close();
      else { show(term); pinned = true; }
    } else close();
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Escape") close();
    if ((e.key === "Enter" || e.key === " ") && e.target.matches(".term-help[role=button]")) {
      e.preventDefault();
      e.target.click();
    }
  });
  window.addEventListener("resize", position);
  window.addEventListener("scroll", position, true);
  for (const wrap of article.querySelectorAll(".budget-marks .term-wrap")) {
    // Preserve the slider marker's width and its colored tick.
    const term = document.createElement("span");
    term.textContent = wrap.textContent;
    Object.assign(term.dataset, wrap.dataset);
    wrap.replaceChildren(term);
    register(term, "threshold");
  }
  scan(article);
  scan(document.querySelector("footer"));
  updateSpacing();
  window.addEventListener("resize", scheduleSpacing);
  window.visualViewport?.addEventListener("resize", scheduleSpacing);
  new ResizeObserver(scheduleSpacing).observe(article);
  document.fonts.ready.then(scheduleSpacing);
  new MutationObserver(records => {
    for (const record of records) {
      if (record.type === "characterData") annotate(record.target);
      else for (const node of record.addedNodes) {
        if (node.isConnected) scan(node);
      }
    }
    if (active && !active.isConnected) close();
    scheduleSpacing();
  }).observe(article, {childList:true, characterData:true, subtree:true, attributeFilter:["hidden", "open"]});
})();
