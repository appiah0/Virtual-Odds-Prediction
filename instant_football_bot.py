#!/usr/bin/env python3
"""
SportyBet Instant Football scraper + bet bot (Playwright sync API) - v2.

Per bot cycle ("round #N", numbered sequentially and shared by both leagues):
  1. Scrape Spain (10 fixtures) and Germany (9 fixtures): stars, 1X2, O/U (all six goal
     lines via the "Near" dropdown), Double Chance, GG/NG  -> Supabase `scraped_fixtures`.
  2. Pick exactly ONE match from ONE league (Spain OR Germany): most stars, then lowest odds.
     Stake GHS 0.10, place bet, kick off.
  3. Scrape final scores for all 19 fixtures and update the same rows (H/D/A outcome),
     never overwriting the pre-match odds.

Env vars:
  SPORTYBET_PHONE, SPORTYBET_PASSWORD        (required)
  SUPABASE_URL, SUPABASE_KEY                 (service_role key; optional, SQLite is always written)
  HEADLESS [1]  DRY_RUN [0]  DAILY_LOSS_LIMIT [5.00]  MAX_BETS_PER_DAY [100]
  MIN_STARS [0]  MAX_CONSEC_FAILS [3]  LEAGUES [Spain,Germany]
"""
import os
import re
import sys
import json
import time
import random
import sqlite3
import logging
import functools
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from playwright.sync_api import Page, sync_playwright, Error as PWError, TimeoutError as PWTimeout

# ----------------------------------------------------------------- CONFIG
BASE_URL = "https://www.sportybet.com/gh/"
INSTANT_URL = os.getenv("INSTANT_URL") or (
    "https://www.sportybet.com/gh/m/instant-virtuals/quickgame"
    "?sportId=sr:sport:1&leagueId=201127111256lea000000001")
PHONE = os.environ.get("SPORTYBET_PHONE")
PASSWORD = os.environ.get("SPORTYBET_PASSWORD")
SUPABASE_URL = os.getenv("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")

HEADLESS = os.getenv("HEADLESS", "1") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
LEAGUES = [s.strip() for s in os.getenv("LEAGUES", "Spain,Germany").split(",") if s.strip()]
FIXTURES_PER_LEAGUE = {"Germany": 9, "Spain": 10}
KNOWN_LEAGUES = ["England", "Spain", "Germany", "Italy", "Champions", "Euros", "Club World Cup"]

STAKE = 0.10
STAKE_STR = "0.10"
MIN_STARS = int(os.getenv("MIN_STARS", "0"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "5.00"))
MAX_BETS_PER_DAY = int(os.getenv("MAX_BETS_PER_DAY", "100"))
MAX_CONSEC_FAILS = int(os.getenv("MAX_CONSEC_FAILS", "3"))
CYCLE_MIN_SEC, CYCLE_MAX_SEC = 180, 300  # 3-5 minutes between rounds

MAX_SCROLL_STEPS = 12
SCROLL_FRACTION = 0.8
RESULT_TIMEOUT_MS = 120_000
DEFAULT_TIMEOUT_MS = 20_000
USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/159.0.0.0 Safari/537.36")

DB_FILE = "instant_football.db"
STATE_FILE = "sportybet_state.json"
DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

# Market tabs scraped as "n odds per row" -> column names
MARKETS = [
    ("1X2", 3, ["odds_home", "odds_draw", "odds_away"]),
    ("Double Chance", 3, ["odds_1x", "odds_12", "odds_2x"]),
    ("GG/NG", 2, ["odds_gg", "odds_ng"]),
]
OU_LINES = ["05", "15", "25", "35", "45", "55"]
ODDS_COLS = (["odds_home", "odds_draw", "odds_away"]
             + [f"odds_{side}_{ln}" for ln in OU_LINES for side in ("over", "under")]
             + ["odds_1x", "odds_12", "odds_2x", "odds_gg", "odds_ng"])
BASE_COLS = ["fixture_key", "round_no", "league", "home_team", "away_team", "home_stars", "away_stars"]
RESULT_COLS = ["home_score", "away_score", "outcome"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("bot.log", encoding="utf-8")],
)
log = logging.getLogger("bot")

# ----------------------------------------------------------------- JS HELPERS
# Finds fixture rows = ancestor holding exactly `n` odds cells and >=2 team codes.
# Counts filled stars for each side, works out the league from the section header above the
# row, and tags odds cells (data-bot) / the O-U goal-line dropdown (data-bot-dd) for clicking.
EXTRACT_JS = r"""
({names, n, ou, active}) => {
  document.querySelectorAll('[data-bot],[data-bot-dd]').forEach(e => {
    e.removeAttribute('data-bot'); e.removeAttribute('data-bot-dd'); });
  const vis = e => e.getClientRects().length > 0;
  const rawTxt = e => ((e.innerText !== undefined ? e.innerText : e.textContent) || '').trim();
  const SKIP = '.recommended-selections, .betslip-option, .beetslip-content, [data-op="iv-betslip-selection"]';
  const all = [...document.body.querySelectorAll('*')].filter(e => vis(e) && !e.closest(SKIP));
  const leaves = all.filter(e => e.children.length === 0);
  const T = new Map(), P = new Map();
  for (const e of all) {
    const r = e.getBoundingClientRect();
    P.set(e, {x: r.left + r.width / 2, y: r.top + r.height / 2});
  }
  for (const e of leaves) T.set(e, rawTxt(e));

  // prefer SportyBet's own outcome cells (ignores slip totals like 0.10); generic fallback otherwise
  const outs = leaves.filter(e => e.closest('[data-op="iv-outcome"]'));
  const odds = (outs.length ? outs : leaves).filter(e => /^\d{1,3}\.\d{2}$/.test(T.get(e)));
  const codes = leaves.filter(e => /^[A-Z]{3,4}$/.test(T.get(e)));

  // Layout-independent: a fixture = two team codes on the same line. Everything else
  // (odds, stars, goal-line dropdown) belongs to the fixture line it sits closest to.
  const lines = [];
  for (const c of codes.slice().sort((a, b) => P.get(a).y - P.get(b).y)) {
    const y = P.get(c).y;
    const ln = lines.find(l => Math.abs(l.y - y) < 5);
    if (ln) ln.items.push(c); else lines.push({y, items: [c]});
  }
  const pairs = lines.filter(l => l.items.length >= 2).map(l => {
    const it = l.items.sort((a, b) => P.get(a).x - P.get(b).x);
    return {y: l.y, home: it[0], away: it[1], mid: (P.get(it[0]).x + P.get(it[1]).x) / 2,
            odds: [], hs: 0, ast: 0, seen: 0, dd: null};
  }).sort((a, b) => a.y - b.y);
  if (!pairs.length) return [];

  const gaps = pairs.slice(1).map((p, i) => p.y - pairs[i].y).filter(g => g > 10).sort((a, b) => a - b);
  const rowH = gaps.length ? gaps[Math.floor(gaps.length / 2)] : 100;
  const tol = rowH * 0.6;
  const nearest = y => {
    let best = null, bd = 1e9;
    for (const p of pairs) { const d = Math.abs(p.y - y); if (d < bd) { bd = d; best = p; } }
    return bd <= tol ? best : null;
  };

  for (const o of odds) { const p = nearest(P.get(o).y); if (p) p.odds.push(o); }
  if (ou) {
    const re = /^\d\.5$/, tc = e => (e.textContent || '').trim();
    const cand = all.filter(e => re.test(tc(e)));
    for (const e of cand.filter(e => ![...e.children].some(c => re.test(tc(c))))) {
      const p = nearest(P.get(e).y);
      if (p && !p.dd) p.dd = e;
    }
  }

  // SportyBet draws 5 <svg class="star-rating"> per team; lit ones also have "star-rating-lightened"
  const starEls = all.filter(e => e.classList && e.classList.contains('star-rating'));
  for (const s of starEls) {
    const p = nearest(P.get(s).y);
    if (!p) continue;
    p.seen++;
    if (s.classList.contains('star-rating-lightened')) { if (P.get(s).x < p.mid) p.hs++; else p.ast++; }
  }

  let headers = leaves.filter(e => names.includes(T.get(e)));
  // league tab strip = 3+ league names on one line; real section headers are alone on theirs
  headers = headers.filter(h => headers.filter(o => Math.abs(P.get(o).y - P.get(h).y) < 4).length < 3);
  const leagueFor = y => {
    let lg = active, by = -1e9;
    for (const h of headers) { const hy = P.get(h).y; if (hy <= y && hy > by) { by = hy; lg = T.get(h); } }
    return lg;
  };

  const out = [];
  for (const p of pairs) {
    if (p.odds.length !== n) continue;
    const home = T.get(p.home), away = T.get(p.away), league = leagueFor(p.y);
    const oc = p.odds.sort((a, b) => P.get(a).x - P.get(b).x);
    oc.forEach((el, i) => el.setAttribute('data-bot', [league, home, away, i].join('|')));
    if (p.dd) p.dd.setAttribute('data-bot-dd', [league, home, away].join('|'));
    out.push({league, home, away, hs: p.hs, ast: p.ast, stars_seen: p.seen,
              odds: oc.map(el => parseFloat(T.get(el)))});
  }
  return out;
}
"""

# After a goal-line dropdown is open: read the inline table (0.5 ... 5.5 -> [over, under]).
POPUP_JS = r"""
() => {
  const vis = e => e.getClientRects().length > 0;
  const tc = e => (e.textContent || '').trim();
  const all = [...document.body.querySelectorAll('*')].filter(vis);
  const deepest = re => all.filter(e => re.test(tc(e))).filter(e => ![...e.children].some(c => re.test(tc(c))));
  const cx = e => { const r = e.getBoundingClientRect(); return r.left + r.width / 2; };
  const cy = e => { const r = e.getBoundingClientRect(); return r.top + r.height / 2; };
  const want = ['0.5', '1.5', '2.5', '3.5', '4.5', '5.5'];
  const labels = deepest(/^[0-5]\.5$/);
  const oddsEls = deepest(/^\d{1,3}\.\d{2}$/);
  const start = labels.find(e => tc(e) === '0.5');
  if (!start) return null;
  let a = start.parentElement;
  while (a && a !== document.body) {
    const labs = labels.filter(x => a.contains(x));
    const ods = oddsEls.filter(x => a.contains(x));
    if (want.every(w => labs.some(l => tc(l) === w)) && ods.length >= 10) {
      const rows = {};
      for (const w of want) {
        for (const lab of labs.filter(x => tc(x) === w)) {
          const r = ods.filter(x => Math.abs(cy(x) - cy(lab)) < 12).sort((p, q) => cx(p) - cx(q));
          if (r.length >= 1 && r.length <= 2) { rows[w] = r; break; }
        }
      }
      const twos = Object.values(rows).filter(r => r.length === 2);
      const overX = twos.length ? twos.reduce((s, r) => s + cx(r[0]), 0) / twos.length : null;
      const underX = twos.length ? twos.reduce((s, r) => s + cx(r[1]), 0) / twos.length : null;
      const out = {};
      for (const w of want) {
        const r = rows[w];
        if (!r) continue;
        if (r.length === 2) out[w] = [parseFloat(tc(r[0])), parseFloat(tc(r[1]))];
        else if (overX === null) continue;
        else {
          const v = parseFloat(tc(r[0]));
          out[w] = Math.abs(cx(r[0]) - overX) <= Math.abs(cx(r[0]) - underX) ? [v, null] : [null, v];
        }
      }
      return out;
    }
    a = a.parentElement;
  }
  return null;
}
"""

SCROLL_JS = r"""
({dy, anchor}) => {
  let el = null;
  if (anchor === 'odds') el = document.querySelector('[data-bot]');
  else el = [...document.body.querySelectorAll('*')].find(e => e.children.length === 0 &&
        e.getClientRects().length && /^\d+\s*-\s*\d+$/.test((e.textContent || '').trim()));
  let sc = null;
  for (let p = el && el.parentElement; p; p = p.parentElement) {
    if (p.scrollHeight > p.clientHeight + 4 && /(auto|scroll|overlay)/.test(getComputedStyle(p).overflowY)) { sc = p; break; }
  }
  if (!sc) sc = document.scrollingElement;
  const before = sc.scrollTop;
  sc.scrollTop = dy === 'top' ? 0 : before + dy * sc.clientHeight;
  return {moved: sc.scrollTop !== before};
}
"""

DIAG_JS = r"""
() => [...document.body.querySelectorAll('*')]
  .filter(e => e.children.length === 0 && e.getClientRects().length)
  .slice(0, 400).map(e => { const r = e.getBoundingClientRect();
    return e.tagName + '.' + (e.getAttribute('class') || '').slice(0, 40) + ' | ' +
      ((e.innerText !== undefined ? e.innerText : e.textContent) || '').trim().slice(0, 30) +
      ' | x' + Math.round(r.left) + ' y' + Math.round(r.top); })
"""

SCORE_RE = re.compile(r"\b([A-Z]{3,4}) (\d+)\s?-\s?(\d+) ([A-Z]{3,4})\b")

# ----------------------------------------------------------------- STORAGE (SQLite + Supabase)
def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    cols = ["fixture_key TEXT PRIMARY KEY"] + [c for c in BASE_COLS[1:] + ODDS_COLS + RESULT_COLS]
    with db() as con:
        con.execute(f"CREATE TABLE IF NOT EXISTS fixtures({', '.join(cols)})")
        con.execute(
            "CREATE TABLE IF NOT EXISTS bets(ticket TEXT, round_no INT, ts TEXT, league TEXT, home TEXT, "
            "away TEXT, pick TEXT, odds REAL, stars INT, stake REAL, home_score INT, away_score INT, "
            "won INT, payout REAL)"
        )


def sb(method, path, body=None, extra_headers=None):
    """Tiny PostgREST client (stdlib only). Returns parsed JSON, [] for empty, None on failure."""
    if not (SUPABASE_URL and SUPABASE_KEY):
        return None
    headers = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {SUPABASE_KEY}",
               "Content-Type": "application/json", **(extra_headers or {})}
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(f"{SUPABASE_URL}/rest/v1/{path}", data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            raw = r.read()
            return json.loads(raw) if raw else []
    except urllib.error.HTTPError as e:
        log.error("Supabase %s %s -> HTTP %s %s", method, path.split("?")[0], e.code, e.read().decode()[:300])
    except Exception as e:
        log.error("Supabase %s failed: %s", method, e)
    return None


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def save_row(row):
    """Upsert one fixture. None values are dropped so a partial row never nulls existing odds."""
    row = {k: v for k, v in row.items() if v is not None}
    keys = list(row)
    with db() as con:
        con.execute(
            f"INSERT INTO fixtures({','.join(keys)}) VALUES({','.join('?' * len(keys))}) "
            f"ON CONFLICT(fixture_key) DO UPDATE SET "
            + ",".join(f"{k}=excluded.{k}" for k in keys if k != "fixture_key"),
            [row[k] for k in keys],
        )
    sb("POST", "scraped_fixtures?on_conflict=fixture_key", [{**row, "updated_at": now_iso()}],
       {"Prefer": "resolution=merge-duplicates,return=minimal"})


def update_result(fixture_key, home_score, away_score):
    outcome = "H" if home_score > away_score else "A" if away_score > home_score else "D"
    fields = {"home_score": home_score, "away_score": away_score, "outcome": outcome}
    with db() as con:
        con.execute("UPDATE fixtures SET home_score=?, away_score=?, outcome=? WHERE fixture_key=?",
                    (home_score, away_score, outcome, fixture_key))
    # PATCH touches only these columns, so pre-match odds are never overwritten
    sb("PATCH", f"scraped_fixtures?fixture_key=eq.{urllib.parse.quote(fixture_key, safe='')}",
       {**fields, "updated_at": now_iso()}, {"Prefer": "return=minimal"})
    return outcome


def next_round_no():
    """Continue numbering from the highest round already stored (Supabase first, then local)."""
    best = 0
    res = sb("GET", "scraped_fixtures?select=round_no&order=round_no.desc&limit=1")
    if res:
        best = max(best, int(res[0]["round_no"]))
    with db() as con:
        r = con.execute("SELECT MAX(round_no) m FROM fixtures").fetchone()
        best = max(best, r["m"] or 0)
    return best + 1


def today_stats():
    with db() as con:
        r = con.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(stake),0) s, COALESCE(SUM(payout),0) p "
            "FROM bets WHERE date(ts)=date('now','localtime')"
        ).fetchone()
    return r["n"], r["s"] - r["p"]  # bets today, net loss today (this run's database)


# ----------------------------------------------------------------- UTILITIES
def retry(times=3, delay=3):
    def deco(fn):
        @functools.wraps(fn)
        def wrap(*a, **k):
            for i in range(1, times + 1):
                try:
                    return fn(*a, **k)
                except (PWError, PWTimeout, RuntimeError) as e:
                    log.warning("%s failed (%d/%d): %s", fn.__name__, i, times, str(e).splitlines()[0])
                    if i == times:
                        raise
                    time.sleep(delay)
        return wrap
    return deco


def page_of(ctx):
    return ctx if isinstance(ctx, Page) else ctx.page


def debug_dump(page, tag):
    """Screenshot + HTML + a text report of every frame (url + visible text) into debug/."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        page.screenshot(path=str(DEBUG_DIR / f"{stamp}_{tag}.png"))
    except Exception as e:
        log.warning("screenshot failed: %s", e)
    try:
        (DEBUG_DIR / f"{stamp}_{tag}.html").write_text(page.content(), encoding="utf-8")
        lines = [f"PAGE URL: {page.url}"]
        for fr in page.frames:
            try:
                body = fr.locator("body").inner_text(timeout=3000)[:2000]
            except Exception:
                body = "<unreadable>"
            lines.append(f"\n--- FRAME {fr.url}\n{body}")
        try:
            diag = page.evaluate(DIAG_JS)
            lines.append("\n--- VISIBLE LEAF ELEMENTS (tag.class | text | x y)\n" + "\n".join(diag))
        except Exception as e:
            lines.append(f"\n(diag failed: {e})")
        (DEBUG_DIR / f"{stamp}_{tag}.txt").write_text("\n".join(lines), encoding="utf-8")
        log.info("debug saved: debug/%s_%s.(png|html|txt)", stamp, tag)
    except Exception as e:
        log.warning("debug dump failed: %s", e)


def find_ctx(page, wait_s=3):
    """The game widget may be in an iframe that loads late: poll the page and all frames."""
    deadline = time.time() + wait_s
    while True:
        for fr in [page] + [f for f in page.frames if f != page.main_frame]:
            try:
                if fr.get_by_text("Betslip", exact=True).count() or fr.get_by_text("Next Round", exact=False).count():
                    return fr
            except PWError:
                pass
        if time.time() >= deadline:
            return page
        page.wait_for_timeout(500)


def safe_click(loc, timeout=6000):
    """Normal click; if another element (banner, fixed header) intercepts it, fire the click event directly."""
    try:
        loc.click(timeout=timeout)
    except PWTimeout:
        loc.dispatch_event("click")


def click_text(ctx, text, exact=True, timeout=DEFAULT_TIMEOUT_MS, last=False):
    loc = ctx.get_by_text(text, exact=exact)
    loc = loc.last if last else loc.first
    loc.wait_for(state="visible", timeout=timeout)
    safe_click(loc, min(timeout, 6000))


def read_balance(page):
    try:
        m = re.search(r"GHS\s*([\d,]+\.\d{2})", page.locator("body").inner_text(timeout=5000))
        return float(m.group(1).replace(",", "")) if m else None
    except Exception:
        return None


# ----------------------------------------------------------------- LOGIN
def logged_in(page):
    try:
        return page.get_by_text("My Account").first.is_visible(timeout=3000)
    except Exception:
        return False


@retry(times=2, delay=5)
def ensure_login(page):
    page.set_viewport_size({"width": 1100, "height": 900})
    page.goto(BASE_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(2500)
    if logged_in(page):
        return
    if not PHONE or not PASSWORD:
        raise RuntimeError("Not logged in and SPORTYBET_PHONE / SPORTYBET_PASSWORD are not set")
    log.info("Logging in...")
    page.locator("input[name='phone'], input[type='tel']").first.fill(PHONE)
    page.locator("input[name='psd'], input[type='password']").first.fill(PASSWORD)
    page.locator("button[name='logIn'], button:has-text('Login')").first.click()
    page.wait_for_timeout(5000)
    if not logged_in(page):
        debug_dump(page, "login_failed")
        raise RuntimeError("Login failed (captcha/OTP/wrong selectors?)")
    page.context.storage_state(path=STATE_FILE)
    log.info("Logged in.")


# ----------------------------------------------------------------- SCRAPING
def extract(ctx, n=3, ou=False, active=""):
    return ctx.evaluate(EXTRACT_JS, {"names": KNOWN_LEAGUES, "n": n, "ou": ou, "active": active})


def scroll(ctx, dy, anchor="odds"):
    return ctx.evaluate(SCROLL_JS, {"dy": dy, "anchor": anchor})["moved"]


def open_league(ctx, league):
    click_text(ctx, league, exact=True)
    ctx.wait_for_timeout(700)


def select_market(ctx, name):
    click_text(ctx, name, exact=True)
    ctx.wait_for_timeout(600)


def wait_ready(page, timeout_s=60):
    """Return the context (page or iframe) that shows the 1X2 odds view."""
    deadline = time.time() + timeout_s
    ctx = page
    while time.time() < deadline:
        ctx = find_ctx(page, 3)
        try:
            if ctx.get_by_text("1X2", exact=True).first.is_visible():
                return ctx
            # still on last round's result screen -> move on
            if ctx.get_by_text("Total Won", exact=False).count():
                ctx.get_by_text("Next Round", exact=False).first.click(timeout=3000)
        except PWError:
            pass
        try:
            if "40X" in page.locator("body").inner_text(timeout=2000) and "doesn't exist" in page.locator("body").inner_text(timeout=2000):
                raise RuntimeError(f"404 page at {page.url} - set the INSTANT_URL env var to the real Instant Football address")
        except PWError:
            pass
        page.wait_for_timeout(1500)
    raise RuntimeError("Instant Football widget never showed the 1X2 odds view")


def collect(ctx, league, n, ou=False, on_row=None):
    """Scroll down through the league list and back up. Returns {(home, away): extracted_row}."""
    expected = FIXTURES_PER_LEAGUE.get(league)
    seen = {}

    def grab():
        for m in extract(ctx, n, ou, league):
            if m["league"] != league:
                continue
            k = (m["home"], m["away"])
            fresh = k not in seen
            seen[k] = m
            if fresh and on_row:
                on_row(m)

    grab()
    scroll(ctx, "top")
    ctx.wait_for_timeout(300)
    for _ in range(MAX_SCROLL_STEPS):
        grab()
        if expected and len(seen) >= expected:
            break
        if not scroll(ctx, SCROLL_FRACTION):
            break
        ctx.wait_for_timeout(350)
    grab()
    scroll(ctx, "top")
    if expected and len(seen) != expected:
        log.warning("%s (n=%d): expected %d fixtures, got %d", league, n, expected, len(seen))
    return seen


_OU_DUMPED = []


def read_ou_lines(ctx, m):
    """Open the 'Near' goal-line dropdown of one row, read all six lines, close it again."""
    tag = "|".join([m["league"], m["home"], m["away"]])
    loc = ctx.locator(f'[data-bot-dd="{tag}"]')
    if not loc.count():
        if "nodd" not in _OU_DUMPED:
            _OU_DUMPED.append("nodd")
            log.warning("O/U: no goal-line dropdown found for %s (saving debug snapshot)", tag)
            debug_dump(page_of(ctx), "ou_tab_no_dropdown")
        return None
    dd = loc.first
    data = None
    try:
        dd.scroll_into_view_if_needed(timeout=5000)
        safe_click(dd, 4000)
        for _ in range(3):
            ctx.wait_for_timeout(350)
            data = ctx.evaluate(POPUP_JS)
            if data and len(data) == 6:
                break
        if not data and "popup" not in _OU_DUMPED:
            _OU_DUMPED.append("popup")
            log.warning("O/U: dropdown opened but lines not readable for %s (saving debug snapshot)", tag)
            debug_dump(page_of(ctx), "ou_popup_not_read")
    except (PWError, PWTimeout) as e:
        log.warning("O/U dropdown failed for %s: %s", tag, str(e).splitlines()[0])
    finally:
        try:
            safe_click(dd, 3000)  # toggle closed
            ctx.wait_for_timeout(250)
            if ctx.evaluate(POPUP_JS):
                page_of(ctx).keyboard.press("Escape")
                ctx.wait_for_timeout(200)
        except (PWError, PWTimeout):
            pass
    if not data:
        return None
    cols = {}
    for line, (over, under) in data.items():
        k = line.replace(".", "")
        cols[f"odds_over_{k}"], cols[f"odds_under_{k}"] = over, under  # None = locked
    if len(data) != 6:
        log.warning("%s: only %d/6 O/U lines read", tag, len(data))
    return cols


def scrape_league(ctx, league, round_no):
    """Full pre-match scrape for one league. Saves rows as it goes. Returns list of row dicts."""
    rows = {}

    def row_for(m):
        k = (m["home"], m["away"])
        if k not in rows:
            rows[k] = {"fixture_key": f"{round_no}_{league}_{m['home']}_{m['away']}",
                       "round_no": round_no, "league": league, "home_team": m["home"],
                       "away_team": m["away"], "home_stars": m["hs"], "away_stars": m["ast"]}
        return rows[k]

    open_league(ctx, league)
    for market, n, cols in MARKETS:
        select_market(ctx, market)
        got = collect(ctx, league, n)
        if market == "1X2" and not got:
            raise RuntimeError(f"{league}: no fixtures found on the 1X2 tab (see debug/ files)")
        for m in got.values():
            r = row_for(m)
            if len(m["odds"]) == n:
                r.update(zip(cols, m["odds"]))

    select_market(ctx, "O/U")

    def on_row(m):
        lines = read_ou_lines(ctx, m)
        if lines:
            row_for(m).update(lines)

    collect(ctx, league, 2, ou=True, on_row=on_row)

    out = list(rows.values())
    bad = [r["fixture_key"] for r in out if not (1 <= (r.get("home_stars") or 0) <= 5 and 1 <= (r.get("away_stars") or 0) <= 5)]
    if bad:
        log.warning("%s: star count outside 1-5 for %d fixtures (selectors may need tuning)", league, len(bad))
    for r in out:
        save_row(r)
    log.info("%s: scraped and saved %d fixtures (round #%d)", league, len(out), round_no)
    select_market(ctx, "1X2")
    return out


def scrape_results(ctx, leagues):
    out = {}
    for league in leagues:
        try:
            open_league(ctx, league)
        except (PWError, PWTimeout):
            log.warning("results tab %s not found", league)
            continue
        expected = FIXTURES_PER_LEAGUE.get(league)
        res = {}
        scroll(ctx, "top", anchor="score")
        for _ in range(MAX_SCROLL_STEPS):
            text = re.sub(r"\s+", " ", ctx.locator("body").inner_text())
            for h, hg, ag, a in SCORE_RE.findall(text):
                res[(h, a)] = (int(hg), int(ag))
            if expected and len(res) >= expected:
                break
            if not scroll(ctx, SCROLL_FRACTION, anchor="score"):
                break
            ctx.wait_for_timeout(300)
        out[league] = res
        log.info("%s: %d results scraped", league, len(res))
    return out


# ----------------------------------------------------------------- STRATEGY
def choose_selection(by_league):
    """ONE match, ONE league. Highest star rating first, lowest 1X2 odds as tie-break."""
    cands = []
    for league, rows in by_league.items():
        for r in rows:
            for pick, team, stars, odds, idx in (
                ("Home", r["home_team"], r.get("home_stars"), r.get("odds_home"), 0),
                ("Away", r["away_team"], r.get("away_stars"), r.get("odds_away"), 2),
            ):
                if stars is not None and odds and odds > 1.0 and stars >= MIN_STARS:
                    cands.append(dict(league=league, home=r["home_team"], away=r["away_team"],
                                      pick=pick, team=team, stars=stars, odds=odds, idx=idx,
                                      fixture_key=r["fixture_key"]))
    if not cands:
        return None
    cands.sort(key=lambda c: (-c["stars"], c["odds"]))
    return cands[0]


# ----------------------------------------------------------------- BETTING
STAKE_SPAN = 'span[data-op="IV_stake_input"]:not(.m-input--quick-bet)'
PLACE_BET_BTN = '[data-op="IV_betslip_placeBet_cta"]:visible'


@retry(times=3, delay=2)
def select_odds(ctx, sel):
    """League tab -> 1X2 tab -> scroll until the tagged odds cell shows up -> click it."""
    open_league(ctx, sel["league"])
    select_market(ctx, "1X2")
    scroll(ctx, "top")
    css = f'[data-bot="{"|".join([sel["league"], sel["home"], sel["away"], str(sel["idx"])])}"]'
    for _ in range(MAX_SCROLL_STEPS):
        extract(ctx, 3, False, sel["league"])  # re-tags the DOM
        loc = ctx.locator(css)
        if loc.count():
            loc.first.scroll_into_view_if_needed()
            safe_click(loc.first)
            return
        if not scroll(ctx, SCROLL_FRACTION):
            break
    raise RuntimeError(f"odds cell not found: {css}")


def ensure_slip(ctx, sel):
    """Click the odds and confirm the slip shows the stake box; re-click if the tap was missed."""
    stake = ctx.locator(STAKE_SPAN).first
    for attempt in range(1, 4):
        select_odds(ctx, sel)
        try:
            stake.wait_for(state="visible", timeout=5000)
            return
        except PWTimeout:
            pass
        try:  # maybe it registered but the slip is collapsed
            safe_click(ctx.locator('[data-op="iv-betslip-button"]').first, 3000)
            stake.wait_for(state="visible", timeout=3000)
            return
        except (PWError, PWTimeout):
            log.warning("selection did not reach the slip (attempt %d/3), retrying", attempt)
    raise RuntimeError("selection never appeared in the bet slip")


def read_stake(ctx):
    try:
        return float(ctx.locator(STAKE_SPAN).first.inner_text().strip())
    except (PWError, PWTimeout, ValueError):
        return None


def set_stake_with_keypad(ctx):
    """Only used if the saved default stake is not 0.10: tap the stake, clear it, type 0.10."""
    safe_click(ctx.locator(STAKE_SPAN).first)
    ctx.locator(".m-keyboard").first.wait_for(state="visible", timeout=5000)
    for _ in range(10):
        safe_click(ctx.locator('.m-keyboard [data-key="delete"]').first, 2000)
    for ch in STAKE_STR:
        safe_click(ctx.locator(f'.m-keyboard [data-key="{ch}"]').first, 2000)
    ctx.wait_for_timeout(300)


def place_bet(ctx):
    """NOT retried: once Confirm is clicked we must never double-place."""
    stake = ctx.locator(STAKE_SPAN).first
    try:
        stake.wait_for(state="visible", timeout=6000)
    except PWTimeout:
        safe_click(ctx.locator('[data-op="iv-betslip-button"]').first, 5000)  # open the slip
        stake.wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)
    if read_stake(ctx) != STAKE:
        set_stake_with_keypad(ctx)
        if read_stake(ctx) != STAKE:
            raise RuntimeError(f"stake shows {read_stake(ctx)!r}, expected {STAKE_STR}")

    if DRY_RUN:
        log.info("DRY_RUN: slip ready (stake %.2f), not placing the bet", STAKE)
        return None

    safe_click(ctx.locator(PLACE_BET_BTN).first)
    click_text(ctx, "Confirm", exact=True)
    ticket = None
    try:
        ctx.get_by_text("Kick Off", exact=True).wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)
        m = re.search(r"Ticket ID:\s*(\d+)", ctx.locator("body").inner_text())
        ticket = m.group(1) if m else "unknown"
        try:
            click_text(ctx, "Skip", exact=True, timeout=3000)
        except (PWError, PWTimeout):
            pass
        click_text(ctx, "Kick Off", exact=True)
    except (PWError, PWTimeout) as e:
        raise RuntimeError(f"bet placed but kick-off flow failed (ticket {ticket}): {e}")
    return ticket


def wait_for_results(ctx):
    page = page_of(ctx)
    deadline = time.time() + RESULT_TIMEOUT_MS / 1000
    while time.time() < deadline:
        try:
            skip = ctx.get_by_text("Skip to Result", exact=True).first
            if skip.is_visible():
                skip.click()
            if ctx.get_by_text("Total Won", exact=False).count() and \
               ctx.get_by_text("Next Round", exact=False).first.is_visible():
                break
        except PWError:
            pass
        page.wait_for_timeout(1000)
    else:
        raise RuntimeError("timed out waiting for results")
    try:  # close the "YOU WON" popup without advancing to the next round
        if ctx.get_by_text("YOU WON", exact=True).first.is_visible():
            closer = ctx.locator("[class*='close' i]:visible").first
            if closer.count():
                closer.click()
            else:
                page.keyboard.press("Escape")
            page.wait_for_timeout(500)
    except PWError:
        pass


# ----------------------------------------------------------------- CYCLE
def run_cycle(page, round_no):
    """Returns 'ok', 'fail' or 'stop'."""
    n, loss = today_stats()
    if n >= MAX_BETS_PER_DAY or loss >= DAILY_LOSS_LIMIT:
        log.error("Limit hit (bets=%d, net loss=GHS %.2f). Stopping.", n, loss)
        return "stop"

    ctx = page
    try:
        page.set_viewport_size({"width": 430, "height": 932})
        page.goto(INSTANT_URL, wait_until="domcontentloaded")
        ctx = wait_ready(page)
        bal = read_balance(page)
        if bal is not None and bal < STAKE:
            log.error("Balance GHS %.2f is below the GHS %.2f stake. Stopping (deposit to continue).", bal, STAKE)
            return "stop"
        log.info("=== Round #%d | balance=%s ===", round_no, bal)

        by_league = {}
        for league in LEAGUES:
            rows = retry(times=2, delay=2)(scrape_league)(ctx, league, round_no)
            if rows:
                by_league[league] = rows

        if not by_league:
            raise RuntimeError("no league returned any fixtures")
        sel = choose_selection(by_league)
        if not sel:
            log.warning("No qualifying selection this round.")
            return "ok"
        log.info("PICK %s: %s (%s vs %s) stars=%d odds=%.2f", sel["league"], sel["team"],
                 sel["home"], sel["away"], sel["stars"], sel["odds"])

        ensure_slip(ctx, sel)
        ticket = place_bet(ctx)
        if DRY_RUN:
            return "ok"

        wait_for_results(ctx)
        results = scrape_results(ctx, list(by_league))
        outcome_of = {}
        for league, rows in by_league.items():
            for r in rows:
                res = results.get(league, {}).get((r["home_team"], r["away_team"]))
                if res:
                    outcome_of[r["fixture_key"]] = (res, update_result(r["fixture_key"], *res))
        log.info("Results stored for %d/%d fixtures", len(outcome_of), sum(len(v) for v in by_league.values()))

        hs = as_ = won = None
        payout = 0.0
        if sel["fixture_key"] in outcome_of:
            (hs, as_), outcome = outcome_of[sel["fixture_key"]]
            won = int(outcome == ("H" if sel["pick"] == "Home" else "A"))
            payout = round(STAKE * sel["odds"], 2) if won else 0.0
        with db() as con:
            con.execute("INSERT INTO bets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (ticket, round_no, now_iso(), sel["league"], sel["home"], sel["away"], sel["pick"],
                         sel["odds"], sel["stars"], STAKE, hs, as_, won, payout))
        log.info("Ticket %s: score %s-%s won=%s payout=GHS %.2f", ticket, hs, as_, won, payout)
        return "ok"
    except Exception:
        log.exception("Cycle failed")
        debug_dump(page, "cycle_failed")
        return "fail"


def main():
    init_db()
    if not (SUPABASE_URL and SUPABASE_KEY):
        log.warning("SUPABASE_URL / SUPABASE_KEY not set: saving to local SQLite only")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS, args=["--disable-blink-features=AutomationControlled"])
        kw = dict(viewport={"width": 1100, "height": 900}, locale="en-GH", timezone_id="Africa/Accra",
                  user_agent=USER_AGENT)
        if Path(STATE_FILE).exists():
            kw["storage_state"] = STATE_FILE
        context = browser.new_context(**kw)
        context.set_default_timeout(DEFAULT_TIMEOUT_MS)
        page = context.new_page()
        exit_code = 0
        try:
            ensure_login(page)
            round_no = next_round_no()
            fails = 0
            while True:
                started = time.time()
                status = run_cycle(page, round_no)
                if status == "stop":
                    break
                fails = fails + 1 if status == "fail" else 0
                if fails >= MAX_CONSEC_FAILS:
                    log.error("%d failed cycles in a row. Stopping so Actions minutes aren't wasted. "
                              "Check the debug/ folder in the bot-output artifact.", fails)
                    exit_code = 1
                    break
                round_no += 1
                wait = max(random.uniform(CYCLE_MIN_SEC, CYCLE_MAX_SEC) - (time.time() - started), 5)
                log.info("Next round in %.0fs", wait)
                time.sleep(wait)
                if not logged_in(page):
                    ensure_login(page)
        except KeyboardInterrupt:
            log.info("Stopped by user")
        except Exception:
            log.exception("Fatal error")
            debug_dump(page, "fatal")
            exit_code = 1
        finally:
            try:
                context.storage_state(path=STATE_FILE)
            except Exception:
                pass
            browser.close()
        sys.exit(exit_code)


if __name__ == "__main__":
    main()
