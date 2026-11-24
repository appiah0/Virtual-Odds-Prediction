// scanner.js — deterministic OCR + star counting via UMD global Tesseract (no bundler, no AI).
export const CFG = {
  MAX_W: 1400,       // max canvas width before OCR (upscales small screenshots for sharper glyphs)
  MIN_OCR_W: 1100,   // upscale narrow screenshots up to this width so small odds stay legible
  BIN: 105,          // OCR threshold: pixels brighter than this become "ink"
  STAR_DY: 1.55,     // star row centre = team-code centre + STAR_DY * codeHeight
  STAR_HALF: 0.6,    // star band half-height (x codeHeight)
  STAR_PITCH: 0.95,  // star spacing (x codeHeight), fallback when grey stars aren't seen
  AMBER_MIN_PX: 3,   // lit star pixels needed inside one star slot to count it lit
  CHECK_TAB: true,   // refuse the scan if the 1X2 tab is visibly unselected
};

const TESSERACT_SRC = 'https://cdn.jsdelivr.net/npm/tesseract.js@5/dist/tesseract.min.js';
const LEAGUE_WORDS = {
  england: 'England',
  spain: 'Spain',
  germany: 'Germany',
  italy: 'Italy',
  france: 'France',
  champions: 'Champions',
};
const ODDS_RE = /^(?:\d{1,2}[.,]\d{1,2}|[1-9]\d{2})$/;
const CODE_RE = /^[A-Z]{3,4}$/;
const WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:+>-';

let tesseractReady = null;

function parseOddToken(raw) {
  const s = String(raw ?? '').trim().replace(',', '.');
  if (/^\d{1,2}\.\d{1,2}$/.test(s)) {
    return parseFloat(s);
  }
  if (/^[1-9]\d{2}$/.test(s)) {
    return parseInt(s, 10) / 100;
  }
  return NaN;
}

function ensureTesseract() {
  if (window.Tesseract && typeof window.Tesseract.createWorker === 'function') {
    return Promise.resolve(window.Tesseract);
  }
  if (!tesseractReady) {
    tesseractReady = new Promise((resolve, reject) => {
      const existing = document.querySelector(`script[src="${TESSERACT_SRC}"]`);
      if (existing) {
        existing.addEventListener('load', () => resolve(window.Tesseract), { once: true });
        existing.addEventListener('error', () => reject(new Error('Failed to load OCR engine.')), { once: true });
        return;
      }
      const s = document.createElement('script');
      s.src = TESSERACT_SRC;
      s.async = true;
      s.onload = () => {
        if (window.Tesseract && typeof window.Tesseract.createWorker === 'function') {
          resolve(window.Tesseract);
        } else {
          reject(new Error('OCR engine loaded without createWorker.'));
        }
      };
      s.onerror = () => reject(new Error('Failed to load OCR engine from CDN.'));
      document.head.appendChild(s);
    });
  }
  return tesseractReady;
}

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};

function freeCanvas(c) {
  if (c) {
    c.width = 0;
    c.height = 0;
  }
}

async function loadCanvas(source) {
  const bmp = typeof source === 'string'
    ? await createImageBitmap(await (await fetch(source)).blob())
    : await createImageBitmap(source);
  let s = Math.min(1, CFG.MAX_W / bmp.width);
  if (bmp.width * s < CFG.MIN_OCR_W) {
    s = Math.min(CFG.MAX_W / bmp.width, CFG.MIN_OCR_W / bmp.width);
  }
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bmp, 0, 0, c.width, c.height);
  if (bmp.close) bmp.close();
  return c;
}

function binarize(src, threshold = CFG.BIN) {
  const c = document.createElement('canvas');
  c.width = src.width;
  c.height = src.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  const img = src.getContext('2d').getImageData(0, 0, src.width, src.height);
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.max(d[i], d[i + 1], d[i + 2]) > threshold ? 0 : 255;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c;
}

function flattenWords(data) {
  if (Array.isArray(data.words) && data.words.length) return data.words;
  const out = [];
  for (const b of data.blocks || [])
    for (const p of b.paragraphs || [])
      for (const l of p.lines || [])
        for (const w of l.words || []) out.push(w);
  return out;
}

async function ocrWords(canvas, onProgress) {
  const T = await ensureTesseract();
  let worker;
  try {
    worker = await T.createWorker('eng', 1, {
      logger: (m) => {
        if (onProgress && m.status === 'recognizing text') onProgress(m.progress);
      },
    });
    await worker.setParameters({
      tessedit_pageseg_mode: '11',
      tessedit_char_whitelist: WHITELIST,
    });
    const { data } = await worker.recognize(canvas, {}, { blocks: true });
    return flattenWords(data)
      .map((k) => {
        const b = k.bbox || {};
        return { t: String(k.text || '').trim(), x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 };
      })
      .filter((k) => k.t && Number.isFinite(k.x0) && Number.isFinite(k.y1))
      .map((k) => ({
        ...k,
        cx: (k.x0 + k.x1) / 2,
        cy: (k.y0 + k.y1) / 2,
        h: k.y1 - k.y0,
        w: k.x1 - k.x0,
      }));
  } finally {
    if (worker) await worker.terminate();
  }
}

function tabInfo(ctx, words, W, H) {
  const t = words
    .filter((k) => /^1X2$/i.test(k.t) && k.cy < H * 0.28)
    .sort((a, b) => a.x0 - b.x0)[0];
  if (!t) return { y: H * 0.12, selected: null };
  const x0 = Math.max(0, Math.floor(t.x0 - t.w * 0.6));
  const x1 = Math.min(W, Math.floor(t.x1 + t.w * 0.6));
  const y0 = Math.max(0, Math.floor(t.y1 + t.h * 0.4));
  const y1 = Math.min(H, Math.floor(t.y1 + t.h * 3));
  if (x1 - x0 < 4 || y1 - y0 < 2) return { y: t.cy, selected: null };
  const d = ctx.getImageData(x0, y0, x1 - x0, y1 - y0).data;
  let n = 0;
  for (let i = 0; i < d.length; i += 4) {
    if (d[i + 1] > 125 && d[i] < 100 && d[i + 2] < 135 && d[i + 1] - d[i] > 60) n++;
  }
  return { y: t.cy, selected: n >= 35 };
}

function starCount(ctx, x0, y0, x1, y1, codeH) {
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  const w = Math.floor(x1) - x0;
  const h = Math.floor(y1) - y0;
  if (w < 10 || h < 4) return null;
  const d = ctx.getImageData(x0, y0, w, h).data;
  const lit = new Array(w).fill(0);
  const any = new Array(w).fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = d[i], g = d[i + 1], b = d[i + 2];
      const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
      const isLit = r > 155 && g > 95 && b < 125 && r - b > 65;
      const isSlot = !isLit && mx - mn < 40 && mx > 65 && mx < 210;
      if (isLit) {
        lit[x]++;
        any[x]++;
      } else if (isSlot) {
        any[x]++;
      }
    }
  }
  let lo = -1, hi = -1;
  for (let x = 0; x < w; x++) if (any[x] >= 2) { if (lo < 0) lo = x; hi = x; }
  if (lo < 0) return null;
  let span = hi - lo + 1;
  const pitch = codeH * CFG.STAR_PITCH;
  if (span < pitch * 5 * 0.75) {
    const firstLit = lit.findIndex((v) => v >= 1);
    if (firstLit < 0) return 0;
    lo = firstLit;
    span = pitch * 5;
  }
  const slot = span / 5;
  let n = 0;
  for (let i = 0; i < 5; i++) {
    const a = Math.max(0, Math.floor(lo + i * slot + slot * 0.2));
    const b = Math.min(w, Math.floor(lo + (i + 1) * slot - slot * 0.2));
    let sum = 0;
    for (let x = a; x < b; x++) sum += lit[x];
    if (sum >= CFG.AMBER_MIN_PX) n++;
  }
  return n;
}

function parseWords(words, color) {
  const W = color.width, H = color.height;
  const ctx = color.getContext('2d', { willReadFrequently: true });
  const tab = tabInfo(ctx, words, W, H);
  if (CFG.CHECK_TAB && tab.selected === false) {
    throw new Error('Open the 1X2 market in the lobby and take the screenshot again. Other markets are not supported.');
  }

  const odds = words
    .filter((k) => ODDS_RE.test(k.t) && k.cy > tab.y)
    .map((k) => ({ ...k, v: parseOddToken(k.t) }))
    .filter((k) => k.v > 1 && k.v < 100)
    .sort((a, b) => a.cy - b.cy);
  const tol = Math.max(9, median(odds.map((k) => k.h)) * 0.75);
  const clusters = [];
  for (const k of odds) {
    const c = clusters[clusters.length - 1];
    if (c && Math.abs(k.cy - c.cy) <= tol) {
      c.items.push(k);
      c.cy = c.items.reduce((s, i) => s + i.cy, 0) / c.items.length;
    } else clusters.push({ cy: k.cy, items: [k] });
  }
  const rows = clusters
    .filter((c) => c.items.length >= 3)
    .map((c) => ({ cy: c.cy, items: c.items.sort((a, b) => a.cx - b.cx).slice(-3) }));
  if (!rows.length) {
    throw new Error('Could not read any 1X2 odds. Use a sharper, uncropped screenshot of the 1X2 view.');
  }
  const diffs = [];
  for (let i = 1; i < rows.length; i++) diffs.push(rows[i].cy - rows[i - 1].cy);
  const rowH = diffs.length ? median(diffs) : W * 0.15;

  const heads = words
    .filter((k) => k.x0 < W * 0.45 && k.cy > tab.y && LEAGUE_WORDS[k.t.toLowerCase()])
    .map((k) => ({ cy: k.cy, name: LEAGUE_WORDS[k.t.toLowerCase()] }))
    .sort((a, b) => a.cy - b.cy);

  const out = [];
  const seen = new Set();
  for (const row of rows) {
    const firstX = row.items[0].x0;
    const inBand = words.filter(
      (k) => k.cy >= row.cy - rowH * 0.4 && k.cy <= row.cy + rowH * 0.15 && k.x1 < firstX - 4
    );
    const vs = inBand.find((k) => k.t.toLowerCase() === 'vs');
    const codes = inBand
      .filter((k) => CODE_RE.test(k.t.toUpperCase()) && k.t.toLowerCase() !== 'vs')
      .sort((a, b) => a.x0 - b.x0);
    let hc, ac;
    if (vs) {
      hc = [...codes].reverse().find((k) => k.cx < vs.cx);
      ac = codes.find((k) => k.cx > vs.cx);
    } else if (codes.length >= 2) {
      hc = codes[0];
      ac = codes[codes.length - 1];
    }
    if (!hc || !ac) continue;

    let league = null;
    for (const hd of heads) if (hd.cy < row.cy) league = hd.name;

    const midX = vs ? vs.cx : (hc.x1 + ac.x0) / 2;
    const cy = hc.cy + CFG.STAR_DY * hc.h;
    const y0 = cy - CFG.STAR_HALF * hc.h;
    const y1 = cy + CFG.STAR_HALF * hc.h;
    const hs = starCount(ctx, 0, y0, midX - 3, y1, hc.h);
    const awayStars = starCount(ctx, midX + 3, y0, firstX - 4, y1, hc.h);

    const home = hc.t.toUpperCase();
    const away = ac.t.toUpperCase();
    const key = `${league || ''}|${home}|${away}`;
    if (home === away || seen.has(key)) continue;
    seen.add(key);

    out.push({
      league,
      home,
      away,
      home_stars: hs,
      away_stars: awayStars,
      odds: { home: row.items[0].v, draw: row.items[1].v, away: row.items[2].v },
      over_under: null,
    });
  }
  if (!out.length) {
    throw new Error('Could not read fixture rows from that screenshot. Try a clearer 1X2 lobby capture.');
  }
  return {
    market: tab.selected === false ? null : '1X2',
    matches: out,
  };
}

export async function scanScreenshot(source, { onProgress } = {}) {
  let color = null, ocr = null;
  try {
    color = await loadCanvas(source);
    ocr = binarize(color, CFG.BIN);
    const words = await ocrWords(ocr, onProgress);
    freeCanvas(ocr);
    ocr = null;
    const parsed = parseWords(words, color);
    if (!Array.isArray(parsed.matches) || parsed.matches.length === 0) {
      throw new Error('No fixtures detected in screenshot.');
    }
    return parsed;
  } finally {
    freeCanvas(ocr);
    freeCanvas(color);
  }
}
