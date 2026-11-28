// Orthographic view of the toy network. Split each face at the ReLU creases
// so even a coarse wire grid follows the exact piecewise-affine surfaces.
function drawBoundSurfaces(surfaceLayers, neurons, boxCorners) {
  const canvas = document.getElementById("surface-chart");
  const angle = document.getElementById("surface-angle");
  const legend = document.getElementById("surface-layers");
  const budget = document.getElementById("surface-budget");
  const budgetValue = document.getElementById("surface-budget-value");
  const domain = document.getElementById("surface-domain");
  const at = ([a,b,c], [x,y]) => a*x + b*y + c;
  const between = (p, q, t) => p.map((v,i) => v + t*(q[i]-v));

  function splitFaces(square) {
    let faces = [square];
    for (const row of neurons) {
      faces = faces.flatMap(face => {
        const sides = [[], []];
        face.forEach((p,i) => {
          const q = face[(i+1) % face.length], a = at(row,p), b = at(row,q);
          if (a <= 0) sides[0].push(p);
          if (a >= 0) sides[1].push(p);
          if (a*b < 0) {
            const edge = between(p,q,a/(a-b));
            for (const side of sides) side.push(edge);
          }
        });
        return sides.filter(side => side.length >= 3);
      });
    }
    return faces;
  }
  function gridLine(p,q) {
    const stops = [0,1];
    for (const row of neurons) {
      const a = at(row,p), b = at(row,q);
      if (a*b < 0) stops.push(a/(a-b));
    }
    return stops.sort((a,b) => a-b).map(t => between(p,q,t));
  }
  const fullSquare = boxCorners(0, 1);
  const fullVertices = splitFaces(fullSquare).flat();
  const fullLayers = surfaceLayers(.5);
  const values = fullLayers.flatMap(layer => fullVertices.map(layer.value));
  const tick = .005, low = Math.floor(Math.min(...values)/tick)*tick;
  const high = Math.ceil(Math.max(...values)/tick)*tick;
  const controls = fullLayers.map(layer => {
    const label = document.createElement("label");
    const toggle = document.createElement("input");
    const maximum = document.createElement("output");
    toggle.type = "checkbox";
    toggle.checked = true;
    label.style.setProperty("--layer",layer.color);
    label.append(toggle, layer.name, maximum);
    legend.append(label);
    return {toggle, maximum};
  });

  let square, faces, layers;
  function updateBox() {
    const radius = budget.valueAsNumber;
    const lo = .5 - radius, hi = .5 + radius;
    square = boxCorners(lo, hi);
    faces = radius === 0 ? [[square[0]]] : splitFaces(square);
    const vertices = faces.flat();
    layers = surfaceLayers(radius);
    for (const [i,layer] of layers.entries()) {
      controls[i].maximum.textContent = signedFigure(Math.max(...vertices.map(layer.value)), i === 0 ? figures.nearest : figures.upper);
      const divisions = i === 0 ? 8 : 3;
      layer.grid = [];
      for (let j=0; j<=divisions; j++) {
        const t = lo + (hi-lo)*j/divisions;
        layer.grid.push(gridLine([t,lo],[t,hi]), gridLine([lo,t],[hi,t]));
      }
    }
    budgetValue.textContent = "±" + radius.toFixed(3);
    budget.setAttribute("aria-valuetext", "plus or minus " + radius.toFixed(3));
    domain.textContent = "Allowed range for each pixel [" + lo.toFixed(3) + ", " + hi.toFixed(3) + "]";
    render();
  }

  function render() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(w*dpr);
    canvas.height = Math.round(h*dpr);
    const ctx = canvas.getContext("2d");
    ctx.scale(dpr,dpr);
    ctx.lineJoin = "round";
    ctx.font = '12px "Instrument Sans", system-ui, sans-serif';
    const yaw = Number(angle.value)*Math.PI/180;
    const c = Math.cos(yaw), s = Math.sin(yaw);
    const extent = (w-78)/(s+c), depth = h*.23/(s+c), rise = h*.77-66;
    const project = ([x,y],z) => [57+extent*(s+x*c-y*s),
      h-34-depth*(x*s+y*c)-rise*(z-low)/(high-low)];
    function path(points, stroke, fill, width=1) {
      ctx.beginPath();
      points.forEach(([x,y],i) => i ? ctx.lineTo(x,y) : ctx.moveTo(x,y));
      if (fill) { ctx.closePath(); ctx.fillStyle=fill; ctx.fill(); }
      if (stroke) { ctx.strokeStyle=stroke; ctx.lineWidth=width; ctx.stroke(); }
    }
    function text(str,[x,y],dx,dy,align="center") {
      ctx.textAlign=align; ctx.fillStyle="#394e48"; ctx.fillText(str,x+dx,y+dy);
    }
    for (let n=Math.round(low/tick); n<=Math.round(high/tick); n++) {
      const z=n*tick, zAxis=project([0,1],z);
      path([zAxis,project([1,1],z),project([1,0],z)], n===0?"#6d8079":"#d6dfda");
      text(n===0?"0":z.toFixed(3).replace("-","−"), zAxis,-6,4,"right");
    }
    const [axisX, axisY] = project([0,1],high);
    const axisLabel = document.getElementById("surface-axis-label");
    axisLabel.style.left = (axisX - 48) + "px";
    axisLabel.style.top = (axisY - 12) + "px";
    path([project([0,1],high),project([0,1],low),project([0,0],low),project([1,0],low)],"#72827c");
    text("0",project([0,0],low),0,16);
    text("1",project([1,0],low),5,16);
    text("1",project([0,1],low),-1,16);
    [[.55, 0], [0, .55]].forEach((point, index) => {
      const [x, y] = project(point, low);
      const label = document.getElementById("surface-pixel-" + index);
      label.style.left = x + "px";
      label.style.top = (y + 12) + "px";
    });

    function dot(position, color) {
      ctx.beginPath();
      ctx.arc(...position, 3, 0, 2*Math.PI);
      ctx.fillStyle=color;
      ctx.fill();
    }
    const pointOnly = budget.valueAsNumber === 0;
    const footprint = square.map(p => project(p,low));
    if (pointOnly) dot(footprint[0], "#456459");
    else path([...footprint,footprint[0]], "#456459", "rgba(63,107,88,.09)", 1.5);

    for (const [i,layer] of layers.entries()) {
      if (!controls[i].toggle.checked) continue;
      const surface = p => project(p,layer.value(p));
      if (pointOnly) {
        dot(surface(square[0]), layer.color);
        continue;
      }
      for (const face of faces) {
        ctx.globalAlpha = i===0 ? .24 : .045;
        path(face.map(surface),null,layer.color);
      }
      ctx.setLineDash(layer.dash || []);
      for (const [j,line] of layer.grid.entries()) {
        const outline = j<2 || j>=layer.grid.length-2;
        ctx.globalAlpha = outline ? .95 : i===0 ? .62 : .4;
        path(line.map(surface),layer.color,null,outline?1.8:1);
      }
      ctx.setLineDash([]);
      ctx.globalAlpha=1;
    }
  }
  legend.addEventListener("change", render);
  angle.addEventListener("input", render);
  budget.addEventListener("input", updateBox);
  updateBox();
  new ResizeObserver(render).observe(canvas);
}
