const BRUSH_RADIUS = 10;

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const result = document.getElementById("result");
const confidence = document.getElementById("confidence");
const bars = document.getElementById("bars");

let drawing = false;
let last = null;

// Build the probability bar rows once
const rows = [];
for (let d = 0; d < 10; d++) {
  const row = document.createElement("div");
  row.className = "bar-row";
  row.innerHTML = `<span>${d}</span><div class="bar-track"><div class="bar-fill"></div></div><span class="bar-value">0%</span>`;
  bars.appendChild(row);
  rows.push({ fill: row.querySelector(".bar-fill"), value: row.querySelector(".bar-value") });
}

function drawBars(probs) {
  const best = probs.some(p => p > 0) ? probs.indexOf(Math.max(...probs)) : -1;
  probs.forEach((p, d) => {
    rows[d].fill.style.width = `${p * 100}%`;
    rows[d].fill.classList.toggle("best", d === best);
    rows[d].value.textContent = `${Math.round(p * 100)}%`;
  });
}

// Fill with black (not transparent) so the PNG matches the MNIST white-on-black convention
function clear() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  result.textContent = "?";
  confidence.innerHTML = "&nbsp;";
  drawBars(new Array(10).fill(0));
}

// Convert pointer position to canvas pixels (the canvas may be scaled by CSS)
function pos(e) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: (e.clientX - rect.left) * canvas.width / rect.width,
    y: (e.clientY - rect.top) * canvas.height / rect.height,
  };
}

function paintDot(p) {
  ctx.fillStyle = "#fff";
  ctx.beginPath();
  ctx.arc(p.x, p.y, BRUSH_RADIUS, 0, Math.PI * 2);
  ctx.fill();
}

canvas.addEventListener("pointerdown", e => {
  drawing = true;
  canvas.setPointerCapture(e.pointerId);
  last = pos(e);
  paintDot(last);
});

canvas.addEventListener("pointermove", e => {
  if (!drawing) return;
  const p = pos(e);
  ctx.strokeStyle = "#fff";
  ctx.lineWidth = BRUSH_RADIUS * 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(last.x, last.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  last = p;
});

canvas.addEventListener("pointerup", () => {
  if (!drawing) return;
  drawing = false;
  predict();
});

function predict() {
  if (!layers) return; // Model still loading

  // Canvas is black and white, so the red channel is the grayscale value
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Uint8ClampedArray(canvas.width * canvas.height);
  for (let i = 0; i < gray.length; i++) gray[i] = rgba[i * 4];

  const x = preprocess(gray, canvas.width, canvas.height);
  if (!x) return; // Nothing drawn
  const probs = predictProba(layers, x);

  const best = probs.indexOf(Math.max(...probs));
  result.textContent = best;
  confidence.textContent = `Confidence: ${(probs[best] * 100).toFixed(1)}%`;
  drawBars(probs);
}

document.getElementById("predict").addEventListener("click", predict);
document.getElementById("clear").addEventListener("click", clear);

// Weights exported by export_model.py
let layers = null;
async function loadModel() {
  const [meta, buffer] = await Promise.all([
    fetch("model.json").then(r => r.json()),
    fetch("model.bin").then(r => r.arrayBuffer()),
  ]);
  layers = parseModel(meta.sizes, buffer);
}

clear();
confidence.textContent = "Loading model...";
loadModel().then(
  () => { confidence.innerHTML = "&nbsp;"; },
  () => { confidence.textContent = "Failed to load model"; },
);
