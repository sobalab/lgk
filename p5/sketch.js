/* LGK "Let's Go Knicks" — live p5.js port of work/esb_build.py.
   Driving night footage -> LGK ascii matrix, a built Empire State Building,
   city/car lights as glowing binary, and a small slanted Knicks crest that
   fades in/out at the end of a spotlight. Drives off the video's own time so the
   overlay stays in sync; the ESB spire track is precomputed (spire_track.json). */

// ---------- fixed render grid (matches the python defaults) ----------
const W = 1920, H = 1080, CELL = 20;
const GW = W / CELL, GH = H / CELL;          // 96 x 54 cells
const RF = 512;
const CH = "LGK";
const TRACK_FPS = 30;

// colours (r,g,b) — straight from esb_build.py
const SPIRE_BLUE = [95, 165, 255], SHAFT_BLUE = [48, 92, 245], KNICKS_ORANGE = [255, 140, 40];
const GREEN = [30, 200, 70], GREEN_HEAD = [205, 255, 215], ELECTRIC = [90, 225, 255];
const EDGE = [225, 240, 255], CYBER = [90, 205, 255];
const LOGO_ORANGE = [255, 120, 12], LOGO_BLUE = [22, 110, 255], LOGO_GRAY = [195, 205, 222];
// Empire State Building half-width silhouette (yn: 0 = antenna tip .. 1 = base),
// as a fraction of the base half-width. Captures the Art Deco massing: thin antenna,
// tapered mast, the 86th-floor observation crown bulge, a waist, the tall shaft, then
// the stepped setbacks down to the wide 5-storey podium. Detail comes from sampling
// this (plus a faint window/floor facade) per cell, the same way the Knicks crest
// samples its logo image.
const ESB_EY = [0.00, 0.045, 0.075, 0.105, 0.130, 0.150, 0.165, 0.185, 0.205, 0.660, 0.700, 0.730, 0.800, 0.880, 0.920, 1.000];
const ESB_EW = [0.015, 0.030, 0.075, 0.150, 0.230, 0.330, 0.300, 0.360, 0.500, 0.520, 0.560, 0.680, 0.760, 0.880, 0.960, 1.000];
const ESB_HWCELL = 6.0;            // base half-width in cells (screen width = ESB_HWCELL * wscale)
const ESB_MW = 200, ESB_MH = 900;  // silhouette mask resolution

// ---------- tweakable params (same names/defaults as the CLI flags) ----------
const P = {
  bright: 2.3, basevis: 0.5, greentint: 0.5,
  scene: 0.35, rain: 0.85, raindensity: 0.42, flicker: 12.0,
  lights: 0.6, lightglow: 1.7, lightthr: 0.20,
  noiseamt: 0.38, spark: 0.0045,
  glow: 1.1, glowradius: 15.0,
  buildsecs: 2.0, basey: 0.72, wscale: 0.60, esbflicker: 9.0, esbflickamt: 0.16,
  esbglow: 1.0, esbglowspeed: 0.5,
  logow: 520, logocell: 10, logox: 0.65, logoy: 0.24, logoglow: 1.3,
  tilt: 0.78, recede: 0.58, rot: 5.0, logovars: 8, logoflicker: 11.0, fadeperiod: 2.6,
  aura: 0.55, rayglow: 0.4, originx: 0.52, originy: 0.66, rayreach: 0.55, conew: 0.62,
};

// ---------- assets / buffers ----------
let video, font, logoImg, track;
let sampleBuf;                 // video downscaled to GW x GH for per-cell sampling
let codeBuf, codeGlow, lightBuf, lightGlow, wgl;
let esbBuf, esbGlow;           // ESB glyphs + their lush cosmos-style color-cycling bloom
let radImg, auraImg;           // static spotlight + halo (screen space)
let radMask;                   // scratch buffer: radiance clipped to a growing disc from origin
let esbMask;                   // baked ESB silhouette + facade coverage field (Float32)
let crestVariants = [];        // pre-warped crest images (glow baked in)
// per-column / field randomness
let randfield, randbin, rspeed, rtail, rperiod, rphase, ractive, noiseField;
let crestCells = [], crestCols = 0, crestRows = 0, crestOutW = 0, crestOutH = 0;
let embQuad = null, ecx = 0, ecy = 0;
let ready = false, started = false;

// ---------- small utils ----------
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a); t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
function hash3(x, y, z) { let h = (Math.imul(x | 0, 374761393) + Math.imul(y | 0, 668265263) + Math.imul(z | 0, 2246822519)) | 0; h = Math.imul(h ^ h >>> 13, 1274126177); return ((h ^ h >>> 16) >>> 0) / 4294967296; }
const clamp = (v, a, b) => v < a ? a : v > b ? b : v;

// half-width of the ESB silhouette (fraction of base) at height yn (0 tip .. 1 base)
function esbHalf(yn) {
  yn = yn < 0 ? 0 : yn > 1 ? 1 : yn;
  if (yn <= ESB_EY[0]) return ESB_EW[0];
  for (let i = 1; i < ESB_EY.length; i++) if (yn <= ESB_EY[i]) { const f = (yn - ESB_EY[i - 1]) / (ESB_EY[i] - ESB_EY[i - 1]); return ESB_EW[i - 1] + f * (ESB_EW[i] - ESB_EW[i - 1]); }
  return ESB_EW[ESB_EW.length - 1];
}

// bake the detailed ESB silhouette + a faint window/floor facade into a coverage field
function buildESBMask() {
  const g = createGraphics(ESB_MW, ESB_MH);
  g.pixelDensity(1); g.clear(); g.noStroke(); g.fill(255);
  const cx = ESB_MW / 2;
  for (let y = 0; y < ESB_MH; y++) {
    const hw = esbHalf(y / (ESB_MH - 1)) * (ESB_MW * 0.5);
    g.rect(cx - hw, y, hw * 2, 1);          // anti-aliased silhouette bar per row
  }
  g.loadPixels();
  esbMask = new Float32Array(ESB_MW * ESB_MH);
  for (let y = 0; y < ESB_MH; y++) for (let x = 0; x < ESB_MW; x++) {
    let v = g.pixels[(y * ESB_MW + x) * 4 + 3] / 255;   // alpha coverage of the silhouette
    if (v > 0.02) {
      const yn = y / (ESB_MH - 1), xn = (x - cx) / (ESB_MW * 0.5);
      const floors = 0.90 + 0.10 * Math.cos(yn * 150);  // horizontal floor striations
      const bays = 0.86 + 0.14 * Math.cos(xn * 27);      // vertical window mullions
      v *= floors * bays;
    }
    esbMask[y * ESB_MW + x] = v;
  }
  g.remove();
}

// sample the baked ESB field; xn in [-1,1] across the base half-width, yn 0..1 tip->base
function sampleESB(yn, xn) {
  if (xn < -1.04 || xn > 1.04) return 0;
  const fx = (xn * 0.5 + 0.5) * (ESB_MW - 1), fy = clamp(yn, 0, 1) * (ESB_MH - 1);
  let x0 = Math.floor(fx), y0 = Math.floor(fy);
  if (x0 < 0) x0 = 0; if (x0 > ESB_MW - 2) x0 = ESB_MW - 2;
  if (y0 < 0) y0 = 0; if (y0 > ESB_MH - 2) y0 = ESB_MH - 2;
  const tx = fx - x0, ty = fy - y0, i0 = y0 * ESB_MW + x0, i1 = i0 + ESB_MW;
  const a = esbMask[i0], b = esbMask[i0 + 1], c = esbMask[i1], d = esbMask[i1 + 1];
  return (a * (1 - tx) + b * tx) * (1 - ty) + (c * (1 - tx) + d * tx) * ty;
}

function preload() {
  video = createVideo(['../media/og-video.mp4']);
  font = loadFont('../work/fonts/CourierPrime-Regular.ttf');
  logoImg = loadImage('../work/knicks_logo.png');
  track = loadJSON('spire_track.json');
}

function setup() {
  pixelDensity(1);
  const c = createCanvas(W, H);
  c.parent('holder');
  video.elt.muted = true; video.elt.setAttribute('playsinline', '');
  video.volume(0); video.loop(); video.hide();
  // start playback on first interaction (autoplay policies); also try immediately
  video.elt.play().catch(() => { });
  window.addEventListener('pointerdown', () => video.elt.play().catch(() => { }), { once: false });

  sampleBuf = createGraphics(GW, GH);
  codeBuf = createGraphics(W, H);  lightBuf = createGraphics(W, H);
  codeGlow = createGraphics(W, H); lightGlow = createGraphics(W, H);
  esbBuf = createGraphics(W, H);   esbGlow = createGraphics(W, H);
  radMask = createGraphics(W, H);
  for (const g of [codeBuf, lightBuf, codeGlow, lightGlow, esbBuf, esbGlow]) { g.textFont(font); g.textAlign(CENTER, CENTER); g.noStroke(); }
  codeBuf.textSize(CELL + 2); lightBuf.textSize(CELL + 2); esbBuf.textSize(CELL + 2);
  wgl = createGraphics(W, H, WEBGL); wgl.textureMode(NORMAL); wgl.noStroke();

  buildFields();
  buildESBMask();
  buildCrestCells();
  rebuildSignal();   // crest variants + aura + radiance
  buildPanel();
  ready = true;
}

// per-column rain params + scrolling random fields + perlin-ish noise grid
function buildFields() {
  const r = mulberry32(7);
  randfield = new Uint8Array(RF * GW); for (let i = 0; i < randfield.length; i++) randfield[i] = (r() * CH.length) | 0;
  randbin = new Uint8Array(RF * GW); for (let i = 0; i < randbin.length; i++) randbin[i] = r() < 0.5 ? 0 : 1;
  const rr = mulberry32(11);
  rspeed = new Float32Array(GW); rtail = new Float32Array(GW); rperiod = new Float32Array(GW); rphase = new Float32Array(GW); ractive = new Float32Array(GW);
  for (let c = 0; c < GW; c++) {
    rspeed[c] = 7 + rr() * 11; rtail[c] = 5 + rr() * 9;
    rperiod[c] = GH + rtail[c] + (40 + rr() * 120); rphase[c] = rr() * rperiod[c];
    ractive[c] = rr() < P.raindensity ? 1 : 0;
  }
  // smooth value noise: low-res random bilinearly upsampled to (RF x GW)
  const nh = Math.max(2, (RF / 6) | 0), nw = Math.max(2, (GW / 4) | 0);
  const nr = mulberry32(33), lr = new Float32Array(nh * nw);
  for (let i = 0; i < lr.length; i++) lr[i] = nr();
  noiseField = new Float32Array(RF * GW);
  for (let y = 0; y < RF; y++) for (let x = 0; x < GW; x++) {
    const fy = y / RF * (nh - 1), fx = x / GW * (nw - 1);
    const y0 = fy | 0, x0 = fx | 0, y1 = Math.min(nh - 1, y0 + 1), x1 = Math.min(nw - 1, x0 + 1);
    const ty = fy - y0, tx = fx - x0;
    const a = lr[y0 * nw + x0], b = lr[y0 * nw + x1], cc = lr[y1 * nw + x0], d = lr[y1 * nw + x1];
    noiseField[y * GW + x] = (a * (1 - tx) + b * tx) * (1 - ty) + (cc * (1 - tx) + d * tx) * ty;
  }
}

// classify the knicks logo into a grid of coloured cells (orange/blue/gray)
function buildCrestCells() {
  const src = logoImg; src.loadPixels();
  const iw = src.width, ih = src.height;
  // bounding box of non-white, opaque pixels
  let x0 = iw, y0 = ih, x1 = 0, y1 = 0;
  for (let y = 0; y < ih; y++) for (let x = 0; x < iw; x++) {
    const i = (y * iw + x) * 4, R = src.pixels[i], G = src.pixels[i + 1], B = src.pixels[i + 2], A = src.pixels[i + 3];
    if (A > 40 && Math.min(R, G, B) < 232) { if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; }
  }
  const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
  crestCols = Math.max(1, (P.logow / P.logocell) | 0);
  crestRows = Math.max(1, Math.round(crestCols * bh / bw));
  crestOutW = crestCols * P.logocell; crestOutH = crestRows * P.logocell;
  crestCells = [];
  for (let r = 0; r < crestRows; r++) for (let c = 0; c < crestCols; c++) {
    // average the source box region for this cell
    const sx0 = x0 + (c / crestCols) * bw, sx1 = x0 + ((c + 1) / crestCols) * bw;
    const sy0 = y0 + (r / crestRows) * bh, sy1 = y0 + ((r + 1) / crestRows) * bh;
    let R = 0, G = 0, B = 0, cov = 0, n = 0;
    for (let y = sy0 | 0; y < (sy1 | 0); y += 2) for (let x = sx0 | 0; x < (sx1 | 0); x += 2) {
      const i = (y * iw + x) * 4, A = src.pixels[i + 3];
      const on = (A > 40 && Math.min(src.pixels[i], src.pixels[i + 1], src.pixels[i + 2]) < 232) ? 1 : 0;
      if (on) { R += src.pixels[i]; G += src.pixels[i + 1]; B += src.pixels[i + 2]; }
      cov += on; n++;
    }
    if (n === 0 || cov / n < 0.38) continue;
    R /= cov; G /= cov; B /= cov;
    const col = R > B + 18 ? LOGO_ORANGE : B > R + 18 ? LOGO_BLUE : LOGO_GRAY;
    crestCells.push({ r, c, col });
  }
}

// ---------- perspective warp (subdivided textured mesh in a WEBGL buffer) ----------
function bilerp(q, u, v) {
  const tx0 = lerp(q[0][0], q[1][0], u), ty0 = lerp(q[0][1], q[1][1], u);
  const bx0 = lerp(q[3][0], q[2][0], u), by0 = lerp(q[3][1], q[2][1], u);
  return [lerp(tx0, bx0, v), lerp(ty0, by0, v)];
}
function warpToImage(src, quad) {
  wgl.clear(); wgl.push(); wgl.translate(-W / 2, -H / 2); wgl.texture(src);
  const N = 16;
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    const u0 = i / N, u1 = (i + 1) / N, v0 = j / N, v1 = (j + 1) / N;
    const a = bilerp(quad, u0, v0), b = bilerp(quad, u1, v0), c = bilerp(quad, u1, v1), d = bilerp(quad, u0, v1);
    wgl.beginShape();
    wgl.vertex(a[0], a[1], 0, u0, v0); wgl.vertex(b[0], b[1], 0, u1, v0);
    wgl.vertex(c[0], c[1], 0, u1, v1); wgl.vertex(d[0], d[1], 0, u0, v1);
    wgl.endShape(CLOSE);
  }
  wgl.pop();
  return wgl.get();
}

// build the flat crest (glyphs + baked glow) for one flicker seed
function buildCrestFlat(seed) {
  const g = createGraphics(crestOutW, crestOutH);
  g.clear(); g.textFont(font); g.textSize(P.logocell + 1); g.textAlign(LEFT, TOP); g.noStroke();
  const rnd = mulberry32(seed);
  for (const cell of crestCells) {
    g.fill(cell.col[0], cell.col[1], cell.col[2]);
    g.text(CH[(rnd() * CH.length) | 0], cell.c * P.logocell, cell.r * P.logocell);
  }
  // bake a bright glow in flat space: crisp + a few additive blurred passes
  const out = createGraphics(crestOutW, crestOutH); out.clear(); out.blendMode(ADD);
  const lg = P.logoglow;
  out.image(g, 0, 0); out.tint(255, 255, 255, 76); out.image(g, 0, 0); out.noTint(); // crisp ~1.3
  const passes = [[4, 1.5], [11, 1.0], [26, 0.6]];
  for (const [rad, wgt] of passes) {
    let amt = wgt * lg;
    while (amt > 0) {
      const a = Math.min(1, amt);
      out.push(); out.drawingContext.filter = `blur(${rad}px)`; out.tint(255, 255, 255, a * 255); out.image(g, 0, 0); out.pop();
      amt -= 1;
    }
  }
  out.blendMode(BLEND);
  return out;
}

// rebuild crest variants + aura + radiance (called on relevant slider changes)
function rebuildSignal() {
  // perspective quad (matches python: left edge full, right edge receded, slanted)
  ecx = P.logox * W; ecy = P.logoy * H;
  const eW = crestOutW, eH = crestOutH;
  const Wp = P.logow, Hp = Wp * (eH / eW) * P.tilt, rf = P.recede;
  const rot = radians(P.rot), cr = Math.cos(rot), sr = Math.sin(rot);
  const local = [[-Wp / 2, -Hp / 2], [Wp / 2, -Hp / 2 * rf], [Wp / 2, Hp / 2 * rf], [-Wp / 2, Hp / 2]];
  embQuad = local.map(([x, y]) => [ecx + x * cr - y * sr, ecy + x * sr + y * cr]);

  crestVariants = [];
  let flat0 = null;
  for (let k = 0; k < P.logovars; k++) {
    const flat = buildCrestFlat(10 + k);
    if (k === 0) flat0 = flat;
    crestVariants.push(warpToImage(flat, embQuad));
    if (k !== 0) flat.remove();
  }

  // aura: a soft cyber halo from the warped crest silhouette (wide blur, suppressed centre)
  const warped0 = warpToImage(flat0, embQuad); flat0.remove();
  const a = createGraphics(W, H); a.clear(); a.blendMode(ADD);
  for (const [rad, wgt] of [[40, 1.0], [80, 1.4]]) {
    a.push(); a.drawingContext.filter = `blur(${rad}px)`;
    a.tint(CYBER[0], CYBER[1], CYBER[2], clamp(wgt * P.aura, 0, 1) * 255); a.image(warped0, 0, 0); a.pop();
  }
  a.blendMode(BLEND); auraImg = a.get(); a.remove();

  buildRadiance();
}

// static spotlight fan welling up behind the skyline, ending at the crest
function buildRadiance() {
  const g = createGraphics(W, H); g.loadPixels();
  const ox = P.originx * W, oy = P.originy * H;
  const axx = ecx - ox, ayy = ecy - oy, La = Math.hypot(axx, ayy), axu = axx / La, ayu = ayy / La;
  const horizon = P.basey * H;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const vx = x - ox, vy = y - oy, dist = Math.hypot(vx, vy) || 1;
    const dotp = (vx * axu + vy * ayu) / dist;
    const ang = Math.acos(clamp(dotp, -1, 1));
    const cone = Math.exp(-((ang / P.conew) ** 2));
    const rad = Math.exp(-dist / (P.rayreach * H));
    const rays = clamp(1 + 0.18 * Math.sin(ang * 11), 0, 2);
    const emerge = clamp((horizon + 0.05 * H - y) / (0.12 * H), 0, 1);
    const originhide = clamp(dist / (0.14 * H), 0, 1);
    const along = vx * axu + vy * ayu;
    const tocrest = 0.40 + 0.60 * clamp(along / La, 0, 1);
    const endcap = clamp((La - along) / (0.09 * H) + 1, 0, 1);
    let rv = rad * cone * rays * clamp(dotp, 0, 1) * emerge * originhide * tocrest * endcap;
    rv = clamp(rv * P.rayglow, 0, 1.2);
    const i = (y * W + x) * 4;
    g.pixels[i] = clamp(CYBER[0] * rv, 0, 255); g.pixels[i + 1] = clamp(CYBER[1] * rv, 0, 255);
    g.pixels[i + 2] = clamp(CYBER[2] * rv, 0, 255); g.pixels[i + 3] = 255;
  }
  g.updatePixels(); radImg = g.get(); g.remove();
}

// ============================ DRAW ============================
function draw() {
  if (!ready) return;
  background(0);
  const t = (video && video.elt && video.elt.readyState >= 2) ? video.elt.currentTime : (millis() / 1000);
  const n = clamp(Math.floor(t * TRACK_FPS), 0, track.cx.length - 1);
  const cxN = track.cx[n], tyN = track.ty[n];

  // 1) sample the video at cell resolution
  if (video.elt.readyState >= 2) { sampleBuf.image(video, 0, 0, GW, GH); started = true; }
  sampleBuf.loadPixels();

  // 2) build the code grid + lights grid (single pass over cells)
  codeBuf.clear(); lightBuf.clear(); esbBuf.clear();
  const tscroll6 = Math.floor(t * 6), htick = Math.floor(t * 12), etick = Math.floor(t * 22);
  const bscroll = Math.floor(t * P.flicker * 0.7);
  const colScroll = new Int32Array(GW);
  for (let c = 0; c < GW; c++) colScroll[c] = Math.floor((rspeed[c] + P.flicker) * t);
  const ebuzz = 1 + 0.06 * Math.sin(2 * Math.PI * t * 37) + 0.04 * Math.sin(2 * Math.PI * t * 113);

  // ESB geometry (per-frame scalars)
  const center = cxN / CELL, tip = tyN / CELL, baseRow = P.basey * H / CELL;
  const BH = Math.max(baseRow - tip, 6), front = clamp(t / P.buildsecs, 0, 1);
  const built = t >= P.buildsecs, te0 = t - P.buildsecs;
  const flash = built ? Math.exp(-te0 / 0.35) : 0;
  const beat = 1 + 0.7 * flash + 0.12 * Math.sin(2 * Math.PI * 1.1 * te0);

  for (let r = 0; r < GH; r++) {
    const sIdx = r * GW;
    for (let c = 0; c < GW; c++) {
      const si = (sIdx + c) * 4;
      const pr = sampleBuf.pixels[si], pg = sampleBuf.pixels[si + 1], pb = sampleBuf.pixels[si + 2];
      const lumc = (0.299 * pr + 0.587 * pg + 0.114 * pb) / 255;
      const vb = clamp(Math.pow(clamp((lumc - 0.05) / 0.95, 0, 1), 0.5) * 3, 0, 1);
      // rain
      const head = ((rspeed[c] * t + rphase[c]) % rperiod[c] + rperiod[c]) % rperiod[c];
      const distr = head - r;
      const rainb = distr >= 0 ? clamp(1 - distr / rtail[c], 0, 1) * ractive[c] : 0;
      const headn = clamp(1 - Math.abs(distr), 0, 1) * ractive[c];
      // perlin dither
      const nz = noiseField[(((r + tscroll6) % RF) + RF) % RF * GW + c];
      const sceneGate = clamp((vb * 1.6 + (nz - 0.5) * P.noiseamt - 0.12) / 0.28, 0, 1);
      let I = Math.max(P.scene * vb * sceneGate, P.rain * rainb);
      I = Math.max(I, headn * P.rain);
      let cr0 = GREEN[0] + (GREEN_HEAD[0] - GREEN[0]) * headn,
        cg0 = GREEN[1] + (GREEN_HEAD[1] - GREEN[1]) * headn,
        cb0 = GREEN[2] + (GREEN_HEAD[2] - GREEN[2]) * headn;
      // tracers
      const hsel = hash3(2000 + htick, r, c);
      if (rainb > 0.04 && hsel < 0.004) { cr0 = KNICKS_ORANGE[0]; cg0 = KNICKS_ORANGE[1]; cb0 = KNICKS_ORANGE[2]; I = Math.max(I, 0.9); }
      else if (rainb > 0.04 && hsel > 0.996) { cr0 = SPIRE_BLUE[0]; cg0 = SPIRE_BLUE[1]; cb0 = SPIRE_BLUE[2]; I = Math.max(I, 0.9); }
      // electric sparks
      if (I > 0.03 && hash3(5000 + etick, r, c) < P.spark) { cr0 = ELECTRIC[0]; cg0 = ELECTRIC[1]; cb0 = ELECTRIC[2]; I = Math.max(I, 1.15); }
      // ESB — sampled from the detailed Art Deco silhouette + facade mask
      let esbCell = false;
      const yn = (r - tip) / BH;
      if (yn >= 0 && yn <= 1 && yn <= front) {
        const xn = (c + 0.5 - center) / Math.max(ESB_HWCELL * P.wscale, 0.001);
        const cov = sampleESB(yn, xn);
        if (cov > 0.06) {
          const shimmer = 0.80 + 0.20 * Math.sin(2 * Math.PI * (t * 1.4) - yn * 6 + c * 0.25);
          const edge = front < 1 ? clamp(1 - Math.abs(yn - front) / 0.05, 0, 1) : 0;
          let Ib = clamp((0.95 * shimmer + 0.6 * edge) * cov, 0, 1.3);
          // bands
          let bcol;
          if (yn >= 0.42) { const st = clamp((yn - 0.42) / 0.58, 0, 1); bcol = [SHAFT_BLUE[0] * (1 - 0.6 * st), SHAFT_BLUE[1] * (1 - 0.6 * st), SHAFT_BLUE[2] * (1 - 0.6 * st)]; }
          else if (yn >= 0.24) bcol = KNICKS_ORANGE.slice();
          else bcol = SPIRE_BLUE.slice();
          bcol = [bcol[0] * (1 - edge) + EDGE[0] * edge, bcol[1] * (1 - edge) + EDGE[1] * edge, bcol[2] * (1 - edge) + EDGE[2] * edge];
          if (built) { Ib *= beat; bcol = [bcol[0] * (1 - 0.5 * flash) + EDGE[0] * 0.5 * flash, bcol[1] * (1 - 0.5 * flash) + EDGE[1] * 0.5 * flash, bcol[2] * (1 - 0.5 * flash) + EDGE[2] * 0.5 * flash]; }
          // per-cell flicker: building glyphs blink in and out like unstable ascii
          const fk = hash3(7000 + Math.floor(t * P.esbflicker), r, c);
          if (fk < P.esbflickamt) Ib = 0;                              // flickered out
          else if (fk < P.esbflickamt + 0.12) Ib *= 0.4;              // mid transition
          if (Ib > I) { I = Ib; cr0 = bcol[0]; cg0 = bcol[1]; cb0 = bcol[2]; esbCell = true; }
        }
      }
      // draw the LGK glyph
      if (I > 0.02) {
        const m = I * ebuzz;
        const fr = clamp(cr0 * m, 0, 255), fg = clamp(cg0 * m, 0, 255), fb = clamp(cb0 * m, 0, 255);
        codeBuf.fill(fr, fg, fb);
        const gi = randfield[(((r + colScroll[c]) % RF) + RF) % RF * GW + c];
        const gx = c * CELL + CELL / 2, gy = r * CELL + CELL / 2;
        codeBuf.text(CH[gi], gx, gy);
        if (esbCell) { esbBuf.fill(fr, fg, fb); esbBuf.text(CH[gi], gx, gy); }   // copy into the ESB bloom buffer
      }
      // city/car lights as faint native-colour binary
      const litw = Math.pow(clamp((lumc - P.lightthr + (nz - 0.5) * P.noiseamt * 0.5) / 0.34, 0, 1), 1.3);
      if (litw > 0.02) {
        const mx = Math.max(pr, pg, pb) + 1e-3, s = litw * P.lights;
        lightBuf.fill(clamp(pr / mx * s * 255, 0, 255), clamp(pg / mx * s * 255, 0, 255), clamp(pb / mx * s * 255, 0, 255));
        const bi = randbin[(((r + bscroll) % RF) + RF) % RF * GW + c];
        lightBuf.text(bi ? "1" : "0", c * CELL + CELL / 2, r * CELL + CELL / 2);
      }
    }
  }

  // glow buffers (canvas blur)
  codeGlow.clear(); codeGlow.push(); codeGlow.drawingContext.filter = `blur(${P.glowradius}px)`; codeGlow.image(codeBuf, 0, 0); codeGlow.pop();
  lightGlow.clear(); lightGlow.push(); lightGlow.drawingContext.filter = `blur(${P.glowradius * 0.7}px)`; lightGlow.image(lightBuf, 0, 0); lightGlow.pop();

  // 3) composite onto the main canvas
  // base video, lifted + green-graded (two passes ~= frame*[0.83,1.24,0.92])
  if (started) {
    push(); tint(255 * 0.83, 255, 255 * 0.92); image(video, 0, 0, W, H);
    blendMode(ADD); tint(0, 255 * 0.24, 0); image(video, 0, 0, W, H); pop();
  }
  blendMode(ADD);
  image(codeBuf, 0, 0);
  tint(255, 255, 255, clamp(P.glow, 0, 1) * 255); image(codeGlow, 0, 0); noTint();   // glow (glow<=1 here)
  image(lightBuf, 0, 0);
  tint(255, 255, 255, clamp(P.lightglow, 0, 1) * 255); image(lightGlow, 0, 0); noTint();

  // cosmos-style bloom on the ESB ascii: bright twinkling cores wrapped in a soft,
  // wide halo. Tinted white so the glow keeps the building's own prominent Knicks
  // orange + blue (the halo glows orange where it's orange, blue where it's blue).
  if (P.esbglow > 0) {
    const pulse = 1 + 0.2 * Math.sin(2 * Math.PI * t * P.esbglowspeed * 0.5);   // gentle breathing
    for (const [rad, wgt] of [[5, 1.1], [13, 0.85], [30, 0.6], [62, 0.42]]) {
      esbGlow.clear(); esbGlow.push(); esbGlow.drawingContext.filter = `blur(${rad}px)`; esbGlow.image(esbBuf, 0, 0); esbGlow.pop();
      tint(255, 255, 255, clamp(wgt * P.esbglow * pulse, 0, 1) * 255); image(esbGlow, 0, 0);
    }
    noTint();
  }

  // 4) signal: spotlight + crest fade in/out, locked to the building's drift
  const sigstart = P.buildsecs + 0.30, te = t - sigstart;
  if (te >= 0) {
    let on = clamp(te / 0.60, 0, 1); on = on * on * (3 - 2 * on);
    const swell = 1 + 0.20 * Math.exp(-te / 0.5);
    const breathe = 1 + 0.05 * Math.sin(2 * Math.PI * 0.40 * te);
    const dx = Math.round(cxN - track.mean_cx), dy = Math.round(tyN - track.mean_ty);

    // beam grows from the origin point outward on each pulse, retracts back into it as it fades
    const pulse = 0.5 - 0.5 * Math.cos(2 * Math.PI * te / P.fadeperiod);  // 0..1..0
    let grow = on * pulse; grow = grow * grow * (3 - 2 * grow);           // eased reveal envelope
    if (grow > 0.002) {
      const ox = P.originx * W + dx, oy = P.originy * H + dy;
      const maxR = Math.hypot(ecx - P.originx * W, ecy - P.originy * H) * 1.25;
      const front = grow * maxR, soft = Math.max(maxR * 0.18, 1);
      radMask.clear();
      radMask.image(radImg, dx, dy);
      // keep only the disc that has grown out from the origin (soft leading edge)
      const ctx = radMask.drawingContext;
      ctx.save(); ctx.globalCompositeOperation = 'destination-in';
      const gr = ctx.createRadialGradient(ox, oy, 0, ox, oy, front + soft);
      const inner = clamp(front / (front + soft), 0, 1);
      gr.addColorStop(0, 'rgba(255,255,255,1)');
      gr.addColorStop(clamp(inner * 0.9, 0, 1), 'rgba(255,255,255,1)');
      gr.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.fillStyle = gr; ctx.fillRect(0, 0, W, H); ctx.restore();
      tint(255, 255, 255, clamp(grow * swell, 0, 1) * 255); image(radMask, 0, 0); noTint();
    }
    const fade = on * (0.5 - 0.5 * Math.cos(2 * Math.PI * te / P.fadeperiod));
    if (fade > 0.01) {
      const a = clamp(fade * breathe, 0, 1) * 255;
      tint(255, 255, 255, a); image(auraImg, dx, dy);
      const vk = (Math.floor(te * P.logoflicker) * 5) % crestVariants.length;
      image(crestVariants[vk], dx, dy); noTint();
    }

    // first-appearance ECHO: as the crest resolves it pings out a few expanding,
    // fading copies of itself, like a ripple radiating from the emblem
    const echoWindow = 1.8;                 // only during the very first appearance
    if (te < echoWindow) {
      const env = clamp(1 - te / echoWindow, 0, 1);
      const NECHO = 3, echoDur = 0.9, stagger = 0.28, expand = 0.62;
      const img = crestVariants[(Math.floor(te * P.logoflicker) * 5) % crestVariants.length];
      const cxS = ecx + dx, cyS = ecy + dy;
      for (let k = 0; k < NECHO; k++) {
        const lt = te - k * stagger;
        if (lt <= 0 || lt >= echoDur) continue;
        const p = lt / echoDur;             // 0..1 expansion progress
        const s = 1 + expand * p;           // scales outward from the crest centre
        const ea = (1 - p) * (1 - p) * 0.55 * env * on;   // bright at birth, fades as it grows
        if (ea < 0.01) continue;
        push();
        translate(cxS, cyS); scale(s); translate(-cxS, -cyS);
        tint(255, 255, 255, clamp(ea, 0, 1) * 255); image(img, dx, dy);
        pop();
      }
      noTint();
    }
  }
  blendMode(BLEND);

  // hud
  if (frameCount % 15 === 0) select('#hud').html(`${nf(frameRate(), 2, 0)} fps · t=${t.toFixed(2)}s`);
}

// ---------- live params panel ----------
function buildPanel() {
  const panel = select('#panel');
  const rebuildKeys = new Set(['logow', 'logocell', 'logox', 'logoy', 'logoglow', 'tilt', 'recede', 'rot', 'logovars', 'aura', 'rayglow', 'originx', 'originy', 'rayreach', 'conew']);
  const fieldKeys = new Set(['raindensity']);
  const specs = [
    ['scene', 0, 1, 0.01], ['rain', 0, 1.5, 0.01], ['raindensity', 0.05, 1, 0.01], ['flicker', 0, 30, 1],
    ['noiseamt', 0, 1, 0.01], ['spark', 0, 0.03, 0.0005], ['glow', 0, 1, 0.02], ['glowradius', 4, 30, 1],
    ['lights', 0, 1.5, 0.01], ['lightglow', 0, 1, 0.02], ['lightthr', 0.05, 0.5, 0.01],
    ['logoglow', 0.5, 4, 0.1], ['rayglow', 0, 1.2, 0.02], ['aura', 0, 1.5, 0.02],
    ['rot', -40, 40, 1], ['recede', 0.2, 1, 0.02], ['tilt', 0.3, 1, 0.02],
    ['logox', 0.2, 0.9, 0.01], ['logoy', 0.1, 0.6, 0.01], ['logow', 120, 600, 10], ['fadeperiod', 0.8, 5, 0.1],
    ['esbflicker', 0, 20, 1], ['esbflickamt', 0, 0.6, 0.02],
    ['esbglow', 0, 2, 0.05], ['esbglowspeed', 0, 2, 0.05],
  ];
  for (const [key, mn, mx, st] of specs) {
    const lab = createElement('label'); lab.parent(panel);
    const name = createSpan(key); name.parent(lab);
    const val = createSpan(P[key]); val.parent(lab); val.class('v');
    const sl = createSlider(mn, mx, P[key], st); sl.parent(lab);
    sl.input(() => {
      P[key] = sl.value(); val.html(P[key]);
      if (fieldKeys.has(key)) buildFields();
      if (rebuildKeys.has(key)) { buildCrestCells(); rebuildSignal(); }
    });
  }
  const hint = createDiv('drives off the video clock · loops with the footage'); hint.parent(panel); hint.class('hint');
}
