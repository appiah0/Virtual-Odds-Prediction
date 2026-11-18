#!/usr/bin/env python3
"""
SportyBet Instant Football scraper + bet bot (Playwright, sync API).

Setup:
    pip install playwright && playwright install chromium
    export SPORTYBET_PHONE="024xxxxxxx"        # local number, no +233
    export SPORTYBET_PASSWORD="your-password"
    python instant_football_bot.py

Optional env vars (defaults in brackets):
    HEADLESS [1]  LEAGUES [Spain,Germany]  DAILY_LOSS_LIMIT [1.00]
    MAX_BETS_PER_DAY [100]  MIN_STARS [0]  DRY_RUN [0]  (1 = scrape + pick, never place)

NOTE: the page DOM was not visible to the author, only screenshots. Selectors are
heuristic (text based). If something fails, check debug/ for the screenshot + HTML
dump and adjust the CONFIG / helper selectors.
"""
import os
import re
import sys
import random
import sqlite3
import logging
import functools
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright, Error as PWError, TimeoutError as PWTimeout

# ----------------------------------------------------------------- CONFIG
BASE_URL = "https://www.sportybet.com/gh/"
INSTANT_URL = os.getenv("INSTANT_URL", "https://www.sportybet.com/gh/instant-virtual")
PHONE = os.environ.get("SPORTYBET_PHONE")
PASSWORD = os.environ.get("SPORTYBET_PASSWORD")

HEADLESS = os.getenv("HEADLESS", "1") == "1"
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
LEAGUES = [s.strip() for s in os.getenv("LEAGUES", "Spain,Germany").split(",") if s.strip()]
FIXTURES_PER_LEAGUE = {"Germany": 9, "Spain": 10}  # others: scroll until list stops growing
KNOWN_LEAGUES = ["England", "Spain", "Germany", "Italy", "Champions", "Europa"]

STAKE = 0.10
STAKE_STR = "0.10"
MIN_STARS = int(os.getenv("MIN_STARS", "0"))
DAILY_LOSS_LIMIT = float(os.getenv("DAILY_LOSS_LIMIT", "1.00"))
MAX_BETS_PER_DAY = int(os.getenv("MAX_BETS_PER_DAY", "100"))
CYCLE_MIN_SEC, CYCLE_MAX_SEC = 180, 300  # 3-5 minutes between rounds

MAX_SCROLL_STEPS = 12
SCROLL_FRACTION = 0.8
RESULT_TIMEOUT_MS = 120_000
DEFAULT_TIMEOUT_MS = 20_000

DB_FILE = "instant_football.db"
STATE_FILE = "sportybet_state.json"
DEBUG_DIR = Path("debug")
DEBUG_DIR.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler("bot.log", encoding="utf-8")],
)
log = logging.getLogger("bot")

# ----------------------------------------------------------------- JS HELPERS
# Finds every fixture row (3 odds cells + 2 team codes), counts filled stars per team,
# tags the 3 odds cells with data-bot="league|home|away|idx" so we can click them later.
EXTRACT_JS = r"""
(leagueNames) => {
  const vis = e => e.getClientRects().length > 0;
  const txt = e => (e.textContent || '').trim();
  const cx = e => { const r = e.getBoundingClientRect(); return r.left + r.width / 2; };
  const all = [...document.body.querySelectorAll('*')];
  const leaves = all.filter(e => e.children.length === 0 && vis(e));
  const odds = leaves.filter(e => /^\d{1,3}\.\d{2}$/.test(txt(e)));
  const codes = leaves.filter(e => /^[A-Z]{3,4}$/.test(txt(e)));
  const headers = leaves.filter(e => leagueNames.includes(txt(e)));

  const rows = new Set();
  for (const o of odds) {
    let a = o.parentElement;
    while (a && a !== document.body) {
      const os = odds.filter(x => a.contains(x));
      if (os.length > 3) break;
      if (os.length === 3 && codes.filter(x => a.contains(x)).length >= 2) { rows.add(a); break; }
      a = a.parentElement;
    }
  }

  const rgb = s => { const m = (s || '').match(/\d+(\.\d+)?/g); return m ? m.slice(0, 3).map(Number) : null; };
  const isFilled = e => {
    const cs = getComputedStyle(e);
    for (const v of [cs.fill, cs.color, cs.backgroundColor]) {
      const c = rgb(v);
      if (!c || v.includes('rgba(0, 0, 0, 0)')) continue;
      if (c[0] > 180 && c[1] > 120 && c[2] < 130) return true;                 // gold/yellow
      if (Math.abs(c[0] - c[1]) < 22 && Math.abs(c[1] - c[2]) < 22) return false; // grey
    }
    const cls = e.getAttribute('class') || '';
    if (/(empty|inactive|off|grey|gray|disabled)/i.test(cls)) return false;
    if (/(active|filled|full|gold|yellow|checked|on)(\s|$|-|_)/i.test(cls)) return true;
    return txt(e) === '\u2605';
  };

  const out = [];
  for (const a of rows) {
    const rc = codes.filter(x => a.contains(x)).sort((p, q) => cx(p) - cx(q));
    const home = txt(rc[0]), away = txt(rc[1]);
    const mid = (cx(rc[0]) + cx(rc[1])) / 2;
    const starEls = [...a.querySelectorAll('*')].filter(e =>
      (/star/i.test(e.getAttribute('class') || '') && !e.querySelector('[class*="star" i]')) ||
      (e.children.length === 0 && /^[\u2605\u2606]$/.test(txt(e))));
    let hs = 0, as = 0;
    for (const s of starEls) { if (isFilled(s)) { if (cx(s) < mid) hs++; else as++; } }
    const oc = odds.filter(x => a.contains(x)).sort((p, q) => cx(p) - cx(q));
    let league = null;
    for (const h of headers) {
      if (h.compareDocumentPosition(a) & Node.DOCUMENT_POSITION_FOLLOWING) league = txt(h);
    }
    oc.forEach((el, i) => el.setAttribute('data-bot', [league, home, away, i].join('|')));
    out.push({league, home, away, hs, as, stars_seen: starEls.length,
              o1: parseFloat(txt(oc[0])), ox: parseFloat(txt(oc[1])), o2: parseFloat(txt(oc[2]))});
  }
  return out;
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

SCORE_RE = re.compile(r"\b([A-Z]{3,4}) (\d+)\s?-\s?(\d+) ([A-Z]{3,4})\b")

# ----------------------------------------------------------------- DB
def db():
    con = sqlite3.connect(DB_FILE)
    con.row_factory = sqlite3.Row
    return con


def init_db():
    with db() as con:
        con.executescript(
            """
            CREATE TABLE IF NOT EXISTS fixtures(
                round_id TEXT, ts TEXT, league TEXT, home TEXT, away TEXT,
                home_stars INT, away_stars INT, odds_home REAL, odds_draw REAL, odds_away REAL,
                home_goals INT, away_goals INT,
                PRIMARY KEY(round_id, league, home, away));
            CREATE TABLE IF NOT EXISTS bets(
                ticket TEXT, round_id TEXT, ts TEXT, league TEXT, home TEXT, away TEXT,
                pick TEXT, odds REAL, stars INT, stake REAL,
                home_goals INT, away_goals INT, won INT, payout REAL);
            """
        )


def save_fixtures(round_id, league, matches):
    ts = datetime.now().isoformat(timespec="seconds")
    with db() as con:
        for m in matches:
            con.execute(
                "INSERT OR REPLACE INTO fixtures(round_id,ts,league,home,away,home_stars,away_stars,"
                "odds_home,odds_draw,odds_away) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (round_id, ts, league, m["home"], m["away"], m["hs"], m["as"], m["o1"], m["ox"], m["o2"]),
            )


def save_results(round_id, league, results):
    with db() as con:
        for (h, a), (hg, ag) in results.items():
            con.execute(
                "UPDATE fixtures SET home_goals=?, away_goals=? WHERE round_id=? AND league=? AND home=? AND away=?",
                (hg, ag, round_id, league, h, a),
            )


def today_stats():
    with db() as con:
        r = con.execute(
            "SELECT COUNT(*) n, COALESCE(SUM(stake),0) s, COALESCE(SUM(payout),0) p "
            "FROM bets WHERE date(ts)=date('now','localtime')"
        ).fetchone()
    return r["n"], r["s"] - r["p"]  # bets today, net loss today


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


def debug_dump(page, ctx, tag):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        page.screenshot(path=str(DEBUG_DIR / f"{stamp}_{tag}.png"), full_page=True)
        (DEBUG_DIR / f"{stamp}_{tag}.html").write_text(ctx.content(), encoding="utf-8")
        log.info("debug saved: %s_%s", stamp, tag)
    except Exception as e:  # never let debugging crash the bot
        log.warning("debug dump failed: %s", e)


def get_ctx(page):
    """The game widget may live in an iframe; fall back to the page itself."""
    for fr in page.frames:
        if fr == page.main_frame:
            continue
        try:
            if fr.get_by_text("Betslip").count() or fr.get_by_text("Next Round").count():
                return fr
        except PWError:
            pass
    return page


def click_text(ctx, text, exact=True, timeout=DEFAULT_TIMEOUT_MS, last=False):
    loc = ctx.get_by_text(text, exact=exact)
    loc = loc.last if last else loc.first
    loc.wait_for(state="visible", timeout=timeout)
    loc.click(timeout=timeout)


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
        debug_dump(page, page, "login_failed")
        raise RuntimeError("Login failed (captcha/OTP/wrong selectors?). Run with HEADLESS=0 and log in once by hand.")
    page.context.storage_state(path=STATE_FILE)


# ----------------------------------------------------------------- SCRAPING
def extract(ctx):
    return ctx.evaluate(EXTRACT_JS, KNOWN_LEAGUES)


def scroll(ctx, dy, anchor="odds"):
    return ctx.evaluate(SCROLL_JS, {"dy": dy, "anchor": anchor})["moved"]


def open_league(ctx, league):
    click_text(ctx, league, exact=True)
    ctx.wait_for_timeout(700)


def scrape_league(ctx, league):
    """Scroll down through the whole list (and back up) so no fixture below the fold is missed."""
    expected = FIXTURES_PER_LEAGUE.get(league)
    open_league(ctx, league)
    seen = {}

    def grab():
        for m in extract(ctx):
            if m["league"] == league:
                seen[(m["home"], m["away"])] = m

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
    scroll(ctx, "top")  # scroll back up
    if expected and len(seen) != expected:
        log.warning("%s: expected %d fixtures, got %d", league, expected, len(seen))
    log.info("%s: scraped %d fixtures", league, len(seen))
    return list(seen.values())


def wait_odds_view(ctx):
    ctx.get_by_text("1X2", exact=True).first.wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)


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
        log.info("%s: %d results", league, len(res))
    return out


# ----------------------------------------------------------------- STRATEGY
def choose_selection(by_league):
    """Highest star rating first, lowest 1X2 odds as tie-break. Home or Away side only."""
    cands = []
    for league, matches in by_league.items():
        for m in matches:
            cands.append(dict(league=league, home=m["home"], away=m["away"], pick="Home",
                              team=m["home"], stars=m["hs"], odds=m["o1"], idx=0))
            cands.append(dict(league=league, home=m["home"], away=m["away"], pick="Away",
                              team=m["away"], stars=m["as"], odds=m["o2"], idx=2))
    cands = [c for c in cands if c["stars"] >= MIN_STARS and c["odds"] and c["odds"] > 1.0]
    if not cands:
        return None
    cands.sort(key=lambda c: (-c["stars"], c["odds"]))
    return cands[0]


# ----------------------------------------------------------------- BETTING
@retry(times=3, delay=2)
def select_odds(ctx, sel):
    """Open the league tab, scroll until the tagged odds cell is found, click it."""
    tag = "|".join([sel["league"], sel["home"], sel["away"], str(sel["idx"])])
    css = f'[data-bot="{tag}"]'
    open_league(ctx, sel["league"])
    scroll(ctx, "top")
    for _ in range(MAX_SCROLL_STEPS):
        extract(ctx)  # re-tags the DOM
        loc = ctx.locator(css)
        if loc.count():
            loc.first.scroll_into_view_if_needed()
            loc.first.click()
            return
        if not scroll(ctx, SCROLL_FRACTION):
            break
    raise RuntimeError(f"odds cell not found for {tag}")


def place_bet(page, ctx):
    """NOT retried: once Confirm is clicked we must never double-place."""
    stake_in = ctx.locator(
        "input[type='text']:visible, input[type='tel']:visible, input[type='number']:visible, input:not([type]):visible"
    ).last
    stake_in.wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)
    stake_in.click()
    stake_in.fill(STAKE_STR)
    if abs(float(stake_in.input_value() or 0) - STAKE) > 1e-9:
        raise RuntimeError(f"stake field shows {stake_in.input_value()!r}, expected {STAKE_STR}")

    if DRY_RUN:
        log.info("DRY_RUN: would place bet now")
        return None

    click_text(ctx, "Place Bet", exact=False)
    click_text(ctx, "Confirm", exact=True)
    ticket = None
    try:
        ctx.get_by_text("Kick Off", exact=True).wait_for(state="visible", timeout=DEFAULT_TIMEOUT_MS)
        m = re.search(r"Ticket ID:\s*(\d+)", ctx.locator("body").inner_text())
        ticket = m.group(1) if m else "unknown"
        try:
            click_text(ctx, "Skip", exact=True, timeout=3000)  # simulation speed: skip
        except (PWError, PWTimeout):
            pass
        click_text(ctx, "Kick Off", exact=True)
    except (PWError, PWTimeout) as e:
        raise RuntimeError(f"bet placed but kick-off flow failed (ticket {ticket}): {e}")
    return ticket


def wait_for_results(page, ctx):
    deadline = time.time() + RESULT_TIMEOUT_MS / 1000
    while time.time() < deadline:
        try:
            if ctx.get_by_text("Skip to Result", exact=True).first.is_visible():
                ctx.get_by_text("Skip to Result", exact=True).first.click()
            if ctx.get_by_text("Next Round", exact=False).first.is_visible() and \
               ctx.get_by_text("Total Won", exact=False).count():
                break
        except PWError:
            pass
        page.wait_for_timeout(1000)
    else:
        raise RuntimeError("timed out waiting for results")
    # close the "YOU WON" popup without advancing to the next round
    try:
        if ctx.get_by_text("YOU WON", exact=True).first.is_visible():
            closer = ctx.locator("[class*='close' i]:visible").first
            if closer.count():
                closer.click()
            else:
                page.keyboard.press("Escape")
            page.wait_for_timeout(500)
    except PWError:
        pass


def settle(sel, results):
    res = results.get(sel["league"], {}).get((sel["home"], sel["away"]))
    if not res:
        return None, None, None, None
    hg, ag = res
    won = int((hg > ag) if sel["pick"] == "Home" else (ag > hg))
    return hg, ag, won, round(STAKE * sel["odds"], 2) if won else 0.0


# ----------------------------------------------------------------- CYCLE
def run_cycle(page):
    n, loss = today_stats()
    if n >= MAX_BETS_PER_DAY or loss >= DAILY_LOSS_LIMIT:
        log.error("Daily limit hit (bets=%d, net loss=GHS %.2f). Stopping.", n, loss)
        return False

    page.goto(INSTANT_URL, wait_until="domcontentloaded")
    page.wait_for_timeout(3000)
    ctx = get_ctx(page)
    bal = read_balance(page)
    if bal is not None and bal < STAKE:
        log.error("Balance GHS %.2f below stake. Stopping.", bal)
        return False

    round_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    try:
        wait_odds_view(ctx)
        by_league = {}
        for league in LEAGUES:
            matches = retry(times=3, delay=2)(scrape_league)(ctx, league)
            if matches:
                by_league[league] = matches
                save_fixtures(round_id, league, matches)
        sel = choose_selection(by_league)
        if not sel:
            log.warning("No qualifying selection this round.")
            return True
        log.info("PICK %s %s (%s vs %s) stars=%d odds=%.2f", sel["league"], sel["team"],
                 sel["home"], sel["away"], sel["stars"], sel["odds"])

        select_odds(ctx, sel)
        ticket = place_bet(page, ctx)
        if DRY_RUN:
            return True
        wait_for_results(page, ctx)
        results = scrape_results(ctx, list(by_league))
        for lg, res in results.items():
            save_results(round_id, lg, res)
        hg, ag, won, payout = settle(sel, results)
        with db() as con:
            con.execute(
                "INSERT INTO bets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (ticket, round_id, datetime.now().isoformat(timespec="seconds"), sel["league"], sel["home"],
                 sel["away"], sel["pick"], sel["odds"], sel["stars"], STAKE, hg, ag, won,
                 payout if payout is not None else 0.0),
            )
        log.info("Ticket %s settled: score %s-%s won=%s payout=%s", ticket, hg, ag, won, payout)
    except Exception:
        log.exception("Cycle failed")
        debug_dump(page, ctx, "cycle_failed")
    return True


def main():
    init_db()
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=HEADLESS)
        kw = dict(viewport={"width": 1100, "height": 900}, locale="en-GH")
        if Path(STATE_FILE).exists():
            kw["storage_state"] = STATE_FILE
        context = browser.new_context(**kw)
        context.set_default_timeout(DEFAULT_TIMEOUT_MS)
        page = context.new_page()
        try:
            ensure_login(page)
            while True:
                started = time.time()
                if not run_cycle(page):
                    break
                wait = random.uniform(CYCLE_MIN_SEC, CYCLE_MAX_SEC) - (time.time() - started)
                log.info("Next round in %.0fs", max(wait, 5))
                time.sleep(max(wait, 5))
                if not logged_in(page):
                    ensure_login(page)
        except KeyboardInterrupt:
            log.info("Stopped by user")
        finally:
            context.storage_state(path=STATE_FILE)
            browser.close()


if __name__ == "__main__":
    main()
