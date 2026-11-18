"""
SportyBet virtual football scraper v4 (zero-config, regex + structure).

Env (required): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Env (optional): SPORTYBET_URL, HEADLESS, SCRAPE_RESULTS

Fixes in v4
 1. Odds: tokens with 1-2 decimals are read in on-screen order. After the
    three 1X2 odds, a one-decimal token like "2.5" is treated as the
    Over/Under LINE (skipped, never stored as odds); the next two are
    Over/Under; the next two (if present) are GG/NG.
 2. Stars: counts <svg> icons / rating-container children next to each team
    name (nearest-name assignment, filled vs empty by colour/class).
 3. Results: only results whose event ID belongs to a virtual fixture we
    already stored are accepted (real-world sports results are dropped),
    and they UPDATE the fixture row by event_id - no standalone rows.
"""
import asyncio
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

from playwright.async_api import async_playwright
from supabase import create_client

URL = os.environ.get("SPORTYBET_URL", "https://www.sportybet.com/ng/sport/vFootball")
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"
SCRAPE_RESULTS = os.environ.get("SCRAPE_RESULTS", "true").lower() != "false"
LEAGUES = ["Germany", "Spain"]          # target virtual leagues
TABLE = "scraped_fixtures"

LEAGUE_ALIASES = [
    ("Germany", r"germany|bundesliga"), ("Spain", r"spain|la ?liga"),
    ("England", r"england|premier"), ("Italy", r"italy|serie ?a"),
    ("France", r"france|ligue ?1"),
]
LEAGUE_HEADER_RE = "|".join(p for _, p in LEAGUE_ALIASES)

ODD_TOKEN = re.compile(r"^\d{1,3}\.\d{1,2}$")            # 1 or 2 decimals
ODD_PAT_ANY = r"(?<![\d.])\d{1,3}\.\d{1,2}(?![\d.])"
SCORE_PAT = r"^\s*(\d{1,2})\s*[-\u2013:]\s*(\d{1,2})\s*$"
ID_RE = re.compile(r"ID\s*[:#]?\s*(\d+)", re.I)
ROUND_RE = re.compile(r"(?:round|rd\.?|matchday|md)\s*[:#.]?\s*(\d{1,4})", re.I)
# "ID <tab> Home <tab> 2:1 <tab> Away" as rendered in result tables
RESULT_LINE = re.compile(
    r"^\s*(\d{4,8})\s*\t+\s*(.+?)\s*\t+\s*(\d{1,2})\s*[:\-\u2013]\s*(\d{1,2})"
    r"\s*(?:\([^)]*\))?\s*\t+\s*(.+?)\s*$")
NOISE = re.compile(
    r"^(1|x|2|over|under|gg|ng|home|draw|away|1x2|o/u|btts|vs\.?|v|ft|ht|"
    r"round.*|rd.*|matchday.*|id.*|[+\-]?\d+(\.\d+)?)$", re.I)

DIAG = []


def log(msg):
    print(msg, file=sys.stderr)
    DIAG.append(msg)


# ---------- in-browser row finder ----------
FIND_ROWS_JS = r"""
(args) => {
  const ODD = new RegExp(args.oddPat, 'g');
  const NOISE = new RegExp(args.noise, 'i');
  const HDR = new RegExp(args.leagueRe, 'i');
  const lines = t => (t || '').split('\n').map(s => s.trim()).filter(Boolean);
  const nameLines = L => L.filter(l => /[A-Za-z]{2,}/.test(l) && !NOISE.test(l));
  const teamNames = L => {
    const n = nameLines(L);
    if (!n.length) return [];
    const m = n[0].match(/^(.+?)\s+(?:vs\.?|v)\s+(.+)$/i);
    return m ? [m[1], m[2]] : n.slice(0, 2);
  };
  const ok = el => {
    const t = el.innerText;
    if (!t || t.length > 1500) return false;
    if (teamNames(lines(t)).length < 2) return false;
    return (t.match(ODD) || []).length >= 3;
  };
  const cands = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (el.children.length < 2 && el.tagName !== 'LI') continue;
    if (el.closest('svg')) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (ok(el)) cands.push(el);
  }
  const leaf = cands.filter(el => !cands.some(o => o !== el && el.contains(o)));

  // non-interactive short headings that look like league names
  const headers = [];
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let tn;
  while ((tn = tw.nextNode())) {
    const t = tn.textContent.trim();
    if (t.length < 3 || t.length > 40 || !HDR.test(t)) continue;
    const p = tn.parentElement;
    if (!p || p.closest('button,a,[role=tab],[role=button],nav')) continue;
    headers.push({node: tn, text: t});
  }

  const BAD = /empty|inactive|\boff\b|outline|gr[ae]y|disabled|unfilled|blank|unselected/i;
  const rgb = s => { const m = (s || '').match(/\d+(\.\d+)?/g); return m ? m.slice(0, 3).map(Number) : null; };
  const isGray = c => !c || (Math.max(...c) - Math.min(...c) < 20);
  const isEmptyIcon = el => {
    const cls = (el.getAttribute('class') || '') + ' ' + (el.id || '') + ' ' + (el.getAttribute('aria-label') || '');
    if (BAD.test(cls)) return true;
    const cs = getComputedStyle(el);
    if (parseFloat(cs.opacity) < 0.5) return true;
    const shape = el.querySelector('path,polygon,use,circle') || el;
    const scs = getComputedStyle(shape);
    if (el.tagName.toLowerCase() === 'svg' || shape !== el) {
      if (scs.fill === 'none' || scs.fill === 'rgba(0, 0, 0, 0)') return true;
      return isGray(rgb(scs.fill));
    }
    return isGray(rgb(cs.backgroundColor)) && isGray(rgb(cs.color));
  };
  const centre = e => { const r = e.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; };

  return leaf.map((row, idx) => {
    const L = lines(row.innerText);
    const names = teamNames(L);
    // team-name elements
    const nameEls = [null, null];
    const w = document.createTreeWalker(row, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      const t = n.textContent.trim();
      if (names[0] && t === names[0] && !nameEls[0]) nameEls[0] = n.parentElement;
      else if (names[1] && t === names[1] && !nameEls[1]) nameEls[1] = n.parentElement;
    }
    const nearest = el => {
      const [x, y] = centre(el);
      let best = 0, bd = Infinity;
      nameEls.forEach((ne, i) => {
        if (!ne) return;
        const [nx, ny] = centre(ne);
        const d = Math.hypot(nx - x, (ny - y) * 1.5);
        if (d < bd) { bd = d; best = i; }
      });
      return best;
    };

    const filled = [0, 0], total = [0, 0], dbg = [];
    const done = new Set();
    // (a) rating containers: class/id mentions star|rating, 1-5 icon children, no text
    for (const c of row.querySelectorAll('[class*=star i],[class*=rating i],[id*=star i],[class*=rate i]')) {
      if (c.closest('svg') || c.innerText.trim()) continue;
      const kids = [...c.children];
      if (kids.length < 1 || kids.length > 5) continue;
      const team = nearest(c);
      if (done.has(c) || [...done].some(d => d.contains(c) || c.contains(d))) continue;
      done.add(c);
      total[team] += kids.length;
      kids.forEach(k => { if (!isEmptyIcon(k)) filled[team]++; });
      dbg.push({via: 'container', team, kids: kids.length, cls: (c.getAttribute('class') || '').slice(0, 60)});
    }
    // (b) bare small SVG icons not inside a counted container
    for (const s of row.querySelectorAll('svg')) {
      if ([...done].some(d => d.contains(s))) continue;
      const r = s.getBoundingClientRect();
      if (r.width < 4 || r.width > 40 || r.height < 4 || r.height > 40) continue;
      const par = s.parentElement;
      if (par && ((par.innerText || '').match(ODD) || []).length) continue;
      const team = nearest(s);
      total[team]++;
      if (!isEmptyIcon(s)) filled[team]++;
      if (dbg.length < 12) dbg.push({via: 'svg', team, w: Math.round(r.width),
        fill: getComputedStyle(s.querySelector('path,polygon,use') || s).fill,
        cls: (s.getAttribute('class') || '').slice(0, 40)});
    }

    let hint = null;
    for (const h of headers) {
      if (h.node.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING) hint = h.text;
    }
    return {
      text: row.innerText, lines: L, names,
      filled, total, dbg, hint,
      html: idx < 1 ? row.outerHTML.slice(0, 1800) : null,
    };
  });
}
"""


# ---------- helpers ----------
def clean_name(raw):
    s = ID_RE.sub("", raw or "")
    s = re.sub(r"\b\d+(\.\d+)?\b", "", s)
    s = re.sub(r"[\u2605\u2606\u2B50]", "", s)
    return re.sub(r"\s+", " ", s).strip(" -:|()[]")


def norm_league(raw):
    for name, pat in LEAGUE_ALIASES:
        if raw and re.search(pat, raw, re.I):
            return name
    return None


def make_key(event_id, league, round_no, home, away, day):
    raw = (f"id|{event_id}" if event_id
           else f"{league}|{day}|{round_no}|{home}|{away}")
    return hashlib.sha1(raw.encode()).hexdigest()


def map_odds(lines):
    """Sequential odds mapping with Over/Under line detection.
    Returns (fields dict, ou_line, n_tokens_used)."""
    toks = [l for l in lines if ODD_TOKEN.match(l)]
    out, line = {}, None
    if len(toks) < 3:
        return out, line, len(toks)
    f = lambda t: (float(t) if float(t) >= 1.01 else None)
    out["odds_home"], out["odds_draw"], out["odds_away"] = (f(t) for t in toks[:3])
    rest = toks[3:]
    # a one-decimal x.5 token (or an odd-length remainder starting x.50) is the LINE
    if rest and (re.fullmatch(r"\d\.5", rest[0]) or
                 (re.fullmatch(r"\d\.50", rest[0]) and len(rest) % 2 == 1)):
        line = float(rest.pop(0))
    if len(rest) >= 2:
        if line is None or line == 2.5:
            out["odds_over_25"], out["odds_under_25"] = f(rest[0]), f(rest[1])
        rest = rest[2:]
    if len(rest) >= 2:
        out["odds_gg"], out["odds_ng"] = f(rest[0]), f(rest[1])
    return out, line, len(toks)


def star_value(filled, total):
    """Return 1-5 or None."""
    if total == 0:
        return None
    if filled == 0:
        return None        # all icons look empty/same colour -> ambiguous
    return filled if 1 <= filled <= 5 else None


async def best_frame(page):
    best, best_n = page.main_frame, -1
    for fr in page.frames:
        try:
            txt = await fr.evaluate("document.body ? document.body.innerText : ''")
            n = len(re.findall(ODD_PAT_ANY, txt))
            if n > best_n:
                best, best_n = fr, n
        except Exception:
            continue
    return best, best_n


async def click_label(page, pattern):
    for fr in page.frames:
        try:
            loc = fr.get_by_text(re.compile(pattern, re.I)).first
            if await loc.count() and await loc.is_visible():
                await loc.click(timeout=5000)
                await asyncio.sleep(2)
                return True
        except Exception:
            continue
    return False


async def find_rows(frame):
    return await frame.evaluate(FIND_ROWS_JS, {
        "oddPat": ODD_PAT_ANY, "noise": NOISE.pattern, "leagueRe": LEAGUE_HEADER_RE})


async def all_text_lines(page):
    lines = []
    for fr in page.frames:
        try:
            txt = await fr.evaluate("document.body ? document.body.innerText : ''")
            lines += txt.split("\n")
        except Exception:
            continue
    return lines


# ---------- parsing ----------
def parse_fixture(row, tab_league, day):
    home, away = (clean_name(n) for n in (row["names"] + ["", ""])[:2])
    if not home or not away:
        return None
    odds, line, n_odds = map_odds(row["lines"])
    m = ID_RE.search(row["text"])
    event_id = m.group(1) if m else None
    rm = ROUND_RE.search(row["text"])
    round_no = int(rm.group(1)) if rm else None
    hint = norm_league(row.get("hint"))
    league = hint or tab_league or "Unknown"
    rec = {"league": league, "home_team": home, "away_team": away,
           "event_id": event_id, "round_no": round_no,
           "home_stars": star_value(row["filled"][0], row["total"][0]),
           "away_stars": star_value(row["filled"][1], row["total"][1]),
           "_hint": hint, "_line": line, "_n_odds": n_odds, **odds}
    rec["fixture_key"] = make_key(event_id, league, round_no, home, away, day)
    return rec


def parse_results(lines, known_ids):
    """Accept only result lines whose ID is a known virtual fixture ID."""
    out, dropped = {}, 0
    for l in lines:
        m = RESULT_LINE.match(l)
        if not m:
            continue
        eid = m.group(1)
        if eid not in known_ids:
            dropped += 1
            continue
        hg, ag = int(m.group(3)), int(m.group(4))
        out[eid] = {"event_id": eid, "home_score": hg, "away_score": ag,
                    "outcome": "H" if hg > ag else "A" if hg < ag else "D"}
    return list(out.values()), dropped


async def scrape_results(page, known_ids):
    """Try each visible 'Results' control; keep the one that shows our IDs."""
    for fr in list(page.frames):
        try:
            loc = fr.get_by_text(re.compile(r"^\s*results?\s*$", re.I))
            count = await loc.count()
        except Exception:
            continue
        for i in range(min(count, 4)):
            try:
                el = loc.nth(i)
                if not await el.is_visible():
                    continue
                await el.click(timeout=5000)
                await asyncio.sleep(3)
            except Exception:
                continue
            res, dropped = parse_results(await all_text_lines(page), known_ids)
            log(f"[diag] results view #{i}: {len(res)} virtual results kept, "
                f"{dropped} non-virtual dropped; url={page.url}")
            if res:
                return res
            if "vFootball" not in page.url:      # navigated to real-sports results
                await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
                await asyncio.sleep(6)
    log("[warn] no Results view contained our virtual event IDs")
    return []


# ---------- orchestration ----------
async def scrape(db_known_ids):
    fixtures = {}
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900}, locale="en-NG",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        page = await ctx.new_page()
        try:
            await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            n = 0
            for _ in range(15):
                await asyncio.sleep(3)
                frame, n = await best_frame(page)
                if n >= 3:
                    break
            log(f"[diag] title={await page.title()!r}; frames={len(page.frames)}; odds-like numbers={n}")
            if n < 3:
                raise RuntimeError("No odds found (blocked / geo-restricted / not loaded?)")

            views = [None] + LEAGUES          # current view first, then each league tab
            for tab in views:
                tab_league = None
                if tab:
                    if await click_label(page, rf"^\s*{tab}\b"):
                        tab_league = tab
                    else:
                        log(f"[warn] no clickable '{tab}' tab")
                        continue
                frame, _ = await best_frame(page)
                rows = await find_rows(frame)
                log(f"[diag] view={tab or 'default'}: {len(rows)} fixture rows")
                for r in rows:
                    rec = parse_fixture(r, tab_league, day)
                    if rec:
                        fixtures.setdefault(rec["fixture_key"], rec)
                if rows and tab is None:
                    first = rows[0]
                    log(f"[diag] sample row lines: {first['lines']}")
                    log(f"[diag] sample star debug (filled={first['filled']}, total={first['total']}): {first['dbg']}")
                    log(f"[diag] sample header hint: {first['hint']!r}")
                    if first.get("html"):
                        log(f"[diag] sample row HTML:\n{first['html']}")

            recs = list(fixtures.values())
            if SCRAPE_RESULTS:
                known = set(db_known_ids) | {f["event_id"] for f in recs if f["event_id"]}
                results = await scrape_results(page, known)
            else:
                results = []
        except Exception as e:
            log(f"[error] {e}")
            await page.screenshot(path="debug.png", full_page=True)
            with open("debug.html", "w", encoding="utf-8") as fh:
                fh.write(await page.content())
            raise
        finally:
            await browser.close()
    return recs, results


def main():
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    try:
        rows = client.table(TABLE).select("event_id").not_.is_("event_id", "null").limit(5000).execute().data
        db_ids = {r["event_id"] for r in rows}
    except Exception as e:
        log(f"[warn] could not load known event IDs: {e}")
        db_ids = set()

    fixtures, results = asyncio.run(scrape(db_ids))

    # ---- league scoping for fixtures ----
    dist = {}
    for f in fixtures:
        dist[f["league"]] = dist.get(f["league"], 0) + 1
    log(f"[diag] league labels found: {dist}")
    kept = [f for f in fixtures if f["_hint"] is None or f["league"] in LEAGUES]
    if len(kept) != len(fixtures):
        log(f"[diag] skipped {len(fixtures) - len(kept)} fixtures from non-target leagues "
            f"(add them to LEAGUES if you want them)")
    fixtures = kept

    n = len(fixtures)
    if n:
        log(f"[diag] fixtures={n}; stars missing={sum(1 for f in fixtures if f['home_stars'] is None)}; "
            f"round missing={sum(1 for f in fixtures if f['round_no'] is None)}; "
            f"GG/NG missing={sum(1 for f in fixtures if 'odds_gg' not in f)}")
        counts, lines_seen = {}, {}
        for f in fixtures:
            counts[f["_n_odds"]] = counts.get(f["_n_odds"], 0) + 1
            lines_seen[f["_line"]] = lines_seen.get(f["_line"], 0) + 1
        log(f"[diag] odds tokens per row: {counts}; O/U lines seen: {lines_seen}")

    print("=== DIAGNOSTICS ===")
    print("\n".join(DIAG))
    print("===================")
    print(f"Scraped {n} fixtures, {len(results)} virtual results")
    if not fixtures and not results:
        sys.exit("Nothing scraped - read DIAGNOSTICS above")

    now = datetime.now(timezone.utc).isoformat()
    clean = []
    for f in fixtures:
        r = {k: v for k, v in f.items() if v is not None and not k.startswith("_")}
        r["scraped_at"] = now
        clean.append(r)
    groups = {}
    for r in clean:
        groups.setdefault(tuple(sorted(r)), []).append(r)
    for batch in groups.values():
        for i in range(0, len(batch), 200):
            client.table(TABLE).upsert(batch[i:i + 200], on_conflict="fixture_key").execute()
    print(f"Upserted {len(clean)} fixtures")

    updated = 0
    for r in results:
        eid = r.pop("event_id")
        r["result_scraped_at"] = now
        res = client.table(TABLE).update(r).eq("event_id", eid).execute()
        updated += len(res.data or [])
    print(f"Updated {updated} fixture rows with results")


if __name__ == "__main__":
    main()
