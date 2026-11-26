// scanner.js v2 — Two-pass Tesseract OCR + margin checksum + star rating reader.
import { createWorker } from 'https://esm.sh/tesseract.js@5.1.1';

export const CFG = {
  MAX_W: 1000,
  BIN: 110,
  CROP_SCALE: 3,
  ODDS_TOL: 0.012,
  STAR_DY: 1.55,
  STAR_HALF: 0.6,
  STAR_PITCH: 0.95,
  AMBER_MIN_PX: 4,
  CHECK_TAB: true,
};

const LEAGUE_WORDS = {
  england: 'England',
  spain: 'Spain',
  germany: 'Germany',
  italy: 'Italy',
  france: 'France',
  champions: 'Champions',
};
const ODDS_RE = /^\d{1,2}[.,]\d{2}$/;
const PAGE_WHITELIST = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,:+>-';

const median = (a) => {
  const s = [...a].sort((x, y) => x - y);
  return s.length ? s[Math.floor(s.length / 2)] : 0;
};
const freeCanvas = (c) => {
  if (c) {
    c.width = 0;
    c.height = 0;
  }
};

async function loadCanvas(file) {
  const bmp = await createImageBitmap(file);
  const s = Math.min(1, CFG.MAX_W / bmp.width);
  const c = document.createElement('canvas');
  c.width = Math.round(bmp.width * s);
  c.height = Math.round(bmp.height * s);
  c.getContext('2d', { willReadFrequently: true }).drawImage(bmp, 0, 0, c.width, c.height);
  if (bmp.close) bmp.close();
  return c;
}

function inkify(img) {
  const d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const v = Math.max(d[i], d[i + 1], d[i + 2]) > CFG.BIN ? 0 : 255;
    d[i] = d[i + 1] = d[i + 2] = v;
    d[i + 3] = 255;
  }
  return img;
}

function binarize(src) {
  const c = document.createElement('canvas');
  c.width = src.width;
  c.height = src.height;
  const ctx = c.getContext('2d', { willReadFrequently: true });
  ctx.putImageData(inkify(src.getContext('2d').getImageData(0, 0, src.width, src.height)), 0, 0);
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

async function readPage(worker, canvas) {
  await worker.setParameters({
    tessedit_pageseg_mode: '11',
    tessedit_char_whitelist: PAGE_WHITELIST,
  });
  const { data } = await worker.recognize(canvas, {}, { blocks: true });
  return flattenWords(data)
    .map((k) => {
      const b = k.bbox || {};
      return { t: String(k.text || '').trim(), x0: b.x0, y0: b.y0, x1: b.x1, y1: b.y1 };
    })
    .filter((k) => k.t && Number.isFinite(k.x0) && Number.isFinite(k.y1))
    .map((k) => ({ ...k, cx: (k.x0 + k.x1) / 2, cy: (k.y0 + k.y1) / 2, h: k.y1 - k.y0, w: k.x1 - k.x0 }));
}

async function readCrops(worker, color, boxes, kind) {
  await worker.setParameters({
    tessedit_pageseg_mode: '7',
    tessedit_char_whitelist: kind === 'code' ? 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' : '0123456789.',
  });
  const S = CFG.CROP_SCALE,
    M = 12,
    pad = 4;
  const out = [];
  for (const b of boxes) {
    const x0 = Math.max(0, Math.floor(b.x0 - pad));
    const y0 = Math.max(0, Math.floor(b.y0 - pad));
    const x1 = Math.min(color.width, Math.ceil(b.x1 + pad));
    const y1 = Math.min(color.height, Math.ceil(b.y1 + pad));
    const w = x1 - x0,
      h = y1 - y0;
    if (w < 4 || h < 4) {
      out.push(null);
      continue;
    }
    const tmp = document.createElement('canvas');
    tmp.width = w * S;
    tmp.height = h * S;
    const tctx = tmp.getContext('2d', { willReadFrequently: true });
    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(color, x0, y0, w, h, 0, 0, w * S, h * S);
    tctx.putImageData(inkify(tctx.getImageData(0, 0, tmp.width, tmp.height)), 0, 0);
    const c = document.createElement('canvas');
    c.width = tmp.width + M * 2;
    c.height = tmp.height + M * 2;
    const ctx = c.getContext('2d');
    ctx.fillStyle = '#fff';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.drawImage(tmp, M, M);
    try {
      const { data } = await worker.recognize(c);
      out.push(String(data.text || '').replace(/\s+/g, '').trim());
    } catch (_) {
      out.push(null);
    }
    freeCanvas(tmp);
    freeCanvas(c);
  }
  return out;
}

function tabInfo(ctx, words, W, H) {
  const t = words
    .filter((k) => /^1X2$/i.test(k.t) && k.cy < H * 0.25)
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
    if (d[i + 1] > 130 && d[i] < 90 && d[i + 2] < 130 && d[i + 1] - d[i] > 70) n++;
  }
  return { y: t.cy, selected: n >= 40 };
}

function starCount(ctx, x0, y0, x1, y1, codeH) {
  x0 = Math.max(0, Math.floor(x0));
  y0 = Math.max(0, Math.floor(y0));
  const w = Math.floor(x1) - x0;
  const h = Math.floor(y1) - y0;
  if (w < 10 || h < 4) return null;
  const d = ctx.getImageData(x0, y0, w, h).data;
  const amber = new Array(w).fill(0);
  const any = new Array(w).fill(0);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = (y * w + x) * 4;
      const r = d[i],
        g = d[i + 1],
        b = d[i + 2];
      const mx = Math.max(r, g, b),
        mn = Math.min(r, g, b);
      const isA = r > 170 && g > 100 && b < 110 && r - b > 90;
      const isG = !isA && mx - mn < 35 && mx > 70 && mx < 200;
      if (isA) {
        amber[x]++;
        any[x]++;
      } else if (isG) any[x]++;
    }
  }
  let lo = -1,
    hi = -1;
  for (let x = 0; x < w; x++) {
    if (any[x] >= 2) {
      if (lo < 0) lo = x;
      hi = x;
    }
  }
  if (lo < 0) return null;
  let span = hi - lo + 1;
  const pitch = codeH * CFG.STAR_PITCH;
  if (span < pitch * 5 * 0.75) {
    const firstA = amber.findIndex((v) => v >= 1);
    if (firstA < 0) return 0;
    lo = firstA;
    span = pitch * 5;
  }
  const slot = span / 5;
  let n = 0;
  for (let i = 0; i < 5; i++) {
    const a = Math.max(0, Math.floor(lo + i * slot + slot * 0.2));
    const b = Math.min(w, Math.floor(lo + (i + 1) * slot - slot * 0.2));
    let sum = 0;
    for (let x = a; x < b; x++) sum += amber[x];
    if (sum >= CFG.AMBER_MIN_PX) n++;
  }
  return n;
}

function parseLayout(words, color) {
  const W = color.width,
    H = color.height;
  const ctx = color.getContext('2d', { willReadFrequently: true });
  const tab = tabInfo(ctx, words, W, H);
  if (CFG.CHECK_TAB && tab.selected === false) {
    throw new Error(
      'Open the 1X2 market in the lobby and take the screenshot again. Other markets are not supported.'
    );
  }

  const odds = words
    .filter((k) => ODDS_RE.test(k.t) && k.cy > tab.y)
    .map((k) => ({ ...k, v: parseFloat(k.t.replace(',', '.')) }))
    .filter((k) => k.v > 1 && k.v < 100)
    .sort((a, b) => a.cy - b.cy);
  const tol = Math.max(6, median(odds.map((k) => k.h)) * 0.6);
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
    .filter((k) => k.x0 < W * 0.15 && k.cy > tab.y && LEAGUE_WORDS[k.t.toLowerCase()])
    .map((k) => ({ cy: k.cy, name: LEAGUE_WORDS[k.t.toLowerCase()] }))
    .sort((a, b) => a.cy - b.cy);

  const out = [];
  for (const row of rows) {
    const firstX = row.items[0].x0;
    const inBand = words.filter(
      (k) => k.cy >= row.cy - rowH * 0.4 && k.cy <= row.cy + rowH * 0.15 && k.x1 < firstX - 4
    );
    const vs = inBand.find((k) => k.t.toLowerCase() === 'vs');
    const codes = inBand
      .filter((k) => k.t.toLowerCase() !== 'vs' && k.t.length >= 2 && /^[A-Za-z]+$/.test(k.t))
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
    out.push({
      league,
      hc,
      ac,
      items: row.items,
      hs: starCount(ctx, 0, y0, midX - 3, y1, hc.h),
      as: starCount(ctx, midX + 3, y0, firstX - 4, y1, hc.h),
      starBoxes: [
        [0, y0, midX - 3, y1],
        [midX + 3, y0, firstX - 4, y1],
      ],
    });
  }
  if (!out.length) throw new Error('Found odds but could not read team codes. Try a clearer screenshot.');
  return { market: tab.selected ? '1X2' : null, rows: out, rowsFound: rows.length };
}

const cleanCode = (s) => String(s || '').toUpperCase().replace(/[^A-Z]/g, '');
const oddsNum = (s) => {
  const t = String(s || '').replace(',', '.');
  if (!/^\d{1,2}\.\d{2}$/.test(t)) return null;
  const v = parseFloat(t);
  return v > 1 && v < 100 ? v : null;
};

function pickOdds(cands, M) {
  let best = null,
    bestDev = Infinity;
  for (const a of cands[0])
    for (const b of cands[1])
      for (const c of cands[2]) {
        const dev = M == null ? 0 : Math.abs(1 / a + 1 / b + 1 / c - M);
        if (dev < bestDev - 0.0005) {
          best = [a, b, c];
          bestDev = dev;
        }
      }
  return { odds: best, dev: bestDev };
}

export async function scanScreenshot(file, { onProgress, debug = false } = {}) {
  let color = null,
    ocr = null,
    worker = null;
  try {
    color = await loadCanvas(file);
    ocr = binarize(color);
    worker = await createWorker('eng', 1, {
      logger: (m) => {
        if (onProgress && m.status === 'recognizing text') onProgress(m.progress * 0.5);
      },
    });

    const words = await readPage(worker, ocr);
    freeCanvas(ocr);
    ocr = null;
    const layout = parseLayout(words, color);
    const rows = layout.rows;

    const codeBoxes = rows.flatMap((r) => [r.hc, r.ac]);
    const codeReads = await readCrops(worker, color, codeBoxes, 'code');
    if (onProgress) onProgress(0.7);
    const oddsBoxes = rows.flatMap((r) => r.items);
    const oddsReads = await readCrops(worker, color, oddsBoxes, 'odds');
    if (onProgress) onProgress(0.95);

    const firstSums = rows.map((r) => r.items.reduce((s, k) => s + 1 / k.v, 0));
    const M = rows.length >= 3 ? median(firstSums) : null;

    const warnings = [];
    const matches = [];
    const seen = new Set();
    rows.forEach((r, i) => {
      const pick = (second, first) => {
        const s = cleanCode(second);
        return s.length >= 3 && s.length <= 4 ? s : cleanCode(first);
      };
      const home = pick(codeReads[2 * i], r.hc.t);
      const away = pick(codeReads[2 * i + 1], r.ac.t);

      const cands = r.items.map((k, j) => {
        const second = oddsNum(oddsReads[3 * i + j]);
        return [...new Set([second, k.v].filter((v) => v !== null))];
      });
      const { odds, dev } = pickOdds(cands, M);
      const oddsOk = M == null ? true : dev <= CFG.ODDS_TOL;
      if (!oddsOk) warnings.push(`${home} vs ${away}: odds failed the margin check, row skipped`);

      const key = `${r.league}|${home}|${away}`;
      if (!home || !away || home === away || seen.has(key)) return;
      seen.add(key);
      matches.push({
        league: r.league,
        home,
        away,
        home_stars: r.hs,
        away_stars: r.as,
        odds: { home: odds[0], draw: odds[1], away: odds[2] },
        odds_ok: oddsOk,
        over_under: null,
      });
    });
    if (!matches.length) throw new Error('Could not read any fixtures. Try a clearer screenshot.');

    const res = {
      market: layout.market,
      matches,
      rows_found: layout.rowsFound,
      rows_ok: matches.length,
      warnings,
    };

    if (debug) {
      const ctx = color.getContext('2d');
      ctx.lineWidth = 2;
      ctx.strokeStyle = '#ff00ff';
      ctx.fillStyle = '#ff00ff';
      ctx.font = 'bold 16px sans-serif';
      rows.forEach((r, i) => {
        for (const [a, b, c, d] of r.starBoxes) ctx.strokeRect(a, b, c - a, d - b);
        ctx.fillText(
          `${r.hs}/${r.as}  ${matches[i] ? matches[i].home + '-' + matches[i].away : ''}`,
          4,
          r.starBoxes[0][1] - 2
        );
      });
      res.debugCanvas = color;
    }
    if (onProgress) onProgress(1);
    return res;
  } finally {
    if (worker) await worker.terminate();
    freeCanvas(ocr);
    if (!debug) freeCanvas(color);
  }
}

export async function sendToScanOdds(supabase, tier, scan) {
  const { data, error } = await supabase.functions.invoke('scan-odds', {
    body: { action: 'scan', tier, market: scan.market, matches: scan.matches },
  });
  if (error) {
    let msg = error.message || 'Scan failed.';
    try {
      const j = await error.context.json();
      if (j && j.error) msg = j.error;
    } catch (_) {
      /* keep default */
    }
    throw new Error(msg);
  }
  return data;
}

export async function scanAndRead(supabase, file, tier, opts) {
  const scan = await scanScreenshot(file, opts);
  return sendToScanOdds(supabase, tier, scan);
}

export function showDebug(canvas) {
  canvas.style.cssText =
    'position:fixed;inset:0;width:100%;height:100%;object-fit:contain;background:#000;z-index:99999';
  canvas.onclick = () => canvas.remove();
  document.body.appendChild(canvas);
}
