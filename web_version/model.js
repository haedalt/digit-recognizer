// Preprocessing and MLP inference, ported from the Python version.
// Works on plain arrays so it runs both in the browser and in Node (for tests).

// Pillow's Lanczos filter (a = 3)
function lanczos(x) {
  const sinc = t => (t === 0 ? 1 : Math.sin(Math.PI * t) / (Math.PI * t));
  return x > -3 && x < 3 ? sinc(x) * sinc(x / 3) : 0;
}

// Pillow resamples 8-bit images with fixed-point weights; matching it keeps results identical
const PRECISION_BITS = 22;
const ONE = 2 ** PRECISION_BITS;

function clip8(acc) {
  return Math.min(255, Math.max(0, Math.floor((acc + ONE / 2) / ONE)));
}

// One axis of Pillow's resampling: per output pixel, normalized filter weights over input pixels
function coefficients(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 3 * filterscale;
  const result = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    const xmin = Math.max(Math.trunc(center - support + 0.5), 0);
    const xmax = Math.min(Math.trunc(center + support + 0.5), inSize);
    const k = [];
    let total = 0;
    for (let x = xmin; x < xmax; x++) {
      const w = lanczos((x - center + 0.5) / filterscale);
      k.push(w);
      total += w;
    }
    const fixed = w => {
      const v = (total ? w / total : 0) * ONE;
      return v < 0 ? Math.trunc(v - 0.5) : Math.trunc(v + 0.5);
    };
    result.push({ xmin, k: k.map(fixed) });
  }
  return result;
}

// Equivalent of Image.resize(..., Image.LANCZOS) for an 8-bit grayscale image
function resizeLanczos(src, w, h, newW, newH) {
  const tmp = new Uint8ClampedArray(newW * h);
  const cx = coefficients(w, newW);
  for (let y = 0; y < h; y++) {
    for (let xx = 0; xx < newW; xx++) {
      const { xmin, k } = cx[xx];
      let s = 0;
      for (let i = 0; i < k.length; i++) s += src[y * w + xmin + i] * k[i];
      tmp[y * newW + xx] = clip8(s);
    }
  }
  const out = new Uint8ClampedArray(newW * newH);
  const cy = coefficients(h, newH);
  for (let yy = 0; yy < newH; yy++) {
    const { xmin, k } = cy[yy];
    for (let x = 0; x < newW; x++) {
      let s = 0;
      for (let i = 0; i < k.length; i++) s += tmp[(xmin + i) * newW + x] * k[i];
      out[yy * newW + x] = clip8(s);
    }
  }
  return out;
}

// Convert a drawn grayscale image (white on black) into a 784-length vector in MNIST format.
// MNIST digits fit in a 20x20 box centered by center of mass inside a 28x28 image.
// Must stay in sync with desktop_version's preprocess().
function preprocess(gray, width, height) {
  // Bounding box of non-zero pixels
  let left = width, top = height, right = -1, bottom = -1;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      if (gray[y * width + x]) {
        if (x < left) left = x;
        if (x > right) right = x;
        if (y < top) top = y;
        if (y > bottom) bottom = y;
      }
    }
  }
  if (right < 0) return null; // Nothing drawn

  const w = right - left + 1;
  const h = bottom - top + 1;
  const digit = new Uint8ClampedArray(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) digit[y * w + x] = gray[(top + y) * width + left + x];
  }

  // Scale the longer side to 20 pixels, keeping the aspect ratio
  // (Python's round() rounds halves to even, so match that)
  const roundHalfEven = v => (Math.abs(v % 1) === 0.5 ? 2 * Math.round(v / 2) : Math.round(v));
  const scale = 20 / Math.max(w, h);
  const nw = Math.max(1, roundHalfEven(w * scale));
  const nh = Math.max(1, roundHalfEven(h * scale));
  const small = resizeLanczos(digit, w, h, nw, nh);

  // Paste into the middle of a 28x28 black image
  const canvas = new Uint8ClampedArray(28 * 28);
  const ox = Math.floor((28 - nw) / 2);
  const oy = Math.floor((28 - nh) / 2);
  for (let y = 0; y < nh; y++) {
    for (let x = 0; x < nw; x++) canvas[(oy + y) * 28 + ox + x] = small[y * nw + x];
  }

  // Shift so the center of mass sits at the image center
  let sum = 0, sx = 0, sy = 0;
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const v = canvas[y * 28 + x];
      sum += v;
      sx += v * x;
      sy += v * y;
    }
  }
  if (sum === 0) return null; // Strokes too faint to survive resizing
  const dx = sx / sum - 14;
  const dy = sy / sum - 14;

  // Pillow's AFFINE transform with nearest-neighbor sampling at pixel centers
  const out = new Float32Array(28 * 28);
  for (let y = 0; y < 28; y++) {
    for (let x = 0; x < 28; x++) {
      const srcX = Math.floor(x + 0.5 + dx);
      const srcY = Math.floor(y + 0.5 + dy);
      if (srcX >= 0 && srcX < 28 && srcY >= 0 && srcY < 28) {
        out[y * 28 + x] = canvas[srcY * 28 + srcX] / 255;
      }
    }
  }
  return out;
}

// Load model.json + model.bin written by export_model.py
function parseModel(sizes, buffer) {
  const data = new Float32Array(buffer);
  const layers = [];
  let offset = 0;
  for (let i = 0; i < sizes.length - 1; i++) {
    const n = sizes[i], m = sizes[i + 1];
    const W = data.subarray(offset, offset + n * m);
    offset += n * m;
    const b = data.subarray(offset, offset + m);
    offset += m;
    layers.push({ n, m, W, b });
  }
  return layers;
}

// Forward pass: ReLU hidden layers, softmax output. Returns probabilities for digits 0-9.
function predictProba(layers, input) {
  let a = input;
  layers.forEach((layer, li) => {
    const { n, m, W, b } = layer;
    const z = new Float32Array(b);
    for (let i = 0; i < n; i++) {
      const v = a[i];
      if (v === 0) continue;
      const row = i * m;
      for (let j = 0; j < m; j++) z[j] += v * W[row + j];
    }
    if (li < layers.length - 1) {
      for (let j = 0; j < m; j++) if (z[j] < 0) z[j] = 0;
    }
    a = z;
  });
  const max = Math.max(...a);
  const exps = Array.from(a, v => Math.exp(v - max));
  const total = exps.reduce((s, v) => s + v, 0);
  return exps.map(v => v / total);
}

if (typeof module !== "undefined") {
  module.exports = { preprocess, parseModel, predictProba };
}
