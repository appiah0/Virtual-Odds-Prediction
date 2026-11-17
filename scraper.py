"""
SportyBet virtual football scraper v3 - ZERO CONFIG (no CSS selectors).

Finds fixture rows by *content* (a block with >=3 decimal odds AND >=2
team-name lines), finds tabs by their visible text, and parses everything
with regex. Env: SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Optional env: SPORTYBET_URL, HEADLESS, SCRAPE_RESULTS

Phone-friendly debugging: every run prints a "DIAGNOSTICS" block (sample
rows, what was/wasn't found) to the GitHub Actions log. Read it in the
GitHub mobile app or browser - no DevTools needed.
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
LEAGUES = ["Germany", "Spain"]
TABLE = "scraped_fixtures"

# Odds are mapped sequentially in on-screen order.
ODD_FIELDS = ["odds_home", "odds_draw", "odds_away", "odds_over_25",
              "odds_under_25", "odds_gg", "odds_ng"]

ODD_PAT = r"(?<![\d.])\d{1,3}\.\d{2}(?![\d.])"
SCORE_PAT = r"^\s*(\d{1,2})\s*[-\u2013:]\s*(\d{1,2})\s*$"   # score on its own line
ID_RE = re.compile(r"ID\s*[:#]?\s*(\d+)", re.I)
ROUND_RE = re.compile(r"(?:round|rd\.?|matchday|md)\s*[:#.]?\s*(\d{1,4})", re.I)
NOISE = re.compile(
    r"^(1|x|2|over|under|gg|ng|home|draw|away|1x2|o/u|btts|vs\.?|v|ft|ht|"
    r"round.*|rd.*|matchday.*|id.*|\d+(\.\d+)?)$", re.I)

DIAG = []


def log(msg):
    print(msg, file=sys.stderr)
    DIAG.append(msg)


# ---------- in-browser row finder (pure structure + regex) ----------
FIND_ROWS_JS = r"""
(args) => {
  const ODD = new RegExp(args.oddPat, 'g');
  const SCORE = new RegExp(args.scorePat);
  const NOISE = new RegExp(args.noise, 'i');
  const lines = t => (t || '').split('\n').map(s => s.trim()).filter(Boolean);
  const nameLines = L => L.filter(l => /[A-Za-z]{3,}/.test(l) && !NOISE.test(l));
  const teamNames = L => {
    const n = nameLines(L);
    if (!n.length) return [];
    const m = n[0].match(/^(.+?)\s+(?:vs\.?|v)\s+(.+)$/i);
    return m ? [m[1], m[2]] : n.slice(0, 2);
  };
  const ok = el => {
    const t = el.innerText;
    if (!t || t.length > 1500) return false;
    const L = lines(t);
    if (teamNames(L).length < 2) return false;
    if (args.mode === 'fixtures') return (t.match(ODD) || []).length >= 3;
    return L.some(l => SCORE.test(l)) && (t.match(ODD) || []).length < 3;
  };
  const cands = [];
  for (const el of document.body.querySelectorAll('*')) {
    if (el.children.length < 2 && el.tagName !== 'LI') continue;
    if (el.closest('svg')) continue;
    const r = el.getBoundingClientRect();
    if (r.width === 0 || r.height === 0) continue;
    if (ok(el)) cands.push(el);
  }
  const set = new Set(cands);
  const leaf = cands.filter(el => !cands.some(o => o !== el && el.contains(o)));

  const BAD = /empty|inactive|\boff\b|outline|gr[ae]y|disabled|unfilled|blank/i;
  const attrs = el => [el.getAttribute('class'), el.id, el.getAttribute('aria-label'),
                       el.getAttribute('title'), el.getAttribute('alt'),
                       el.getAttribute('src'), el.getAttribute('data-icon')].filter(Boolean).join(' ');

  return leaf.map(row => {
    const L = lines(row.innerText);
    const names = teamNames(L);
    const stars = [0, 0];
    let explicit = [null, null], cur = 0;
    const w = document.createTreeWalker(row, NodeFilter.SHOW_ELEMENT | NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
      if (n.nodeType === 3) {
        const t = n.textContent.trim();
        if (!t) continue;
        if (names[0] && t === names[0]) cur = 0;
        else if (names[1] && t === names[1]) cur = 1;
        const f = (t.match(/\u2605|\u2B50/g) || []).length;   // filled star glyphs
        if (f) stars[cur] += f;
        continue;
      }
      const el = n;
      if (el.closest('svg') && el.tagName.toLowerCase() !== 'svg') continue;
      const a = attrs(el);
      const em = a.match(/(\d)\s*(?:out of\s*\d\s*)?stars?\b/i);
      if (em && +em[1] >= 1 && +em[1] <= 5) { explicit[cur] = +em[1]; continue; }
      const tag = el.tagName.toLowerCase();
      const isIcon = tag === 'svg' || tag === 'img' || tag === 'i' || el.children.length === 0;
      let hay = a;
      if (tag === 'svg') hay += ' ' + el.outerHTML.slice(0, 400);
      if (isIcon && /star/i.test(hay) && !BAD.test(a)) stars[cur] += 1;
    }
    return {
      text: row.innerText,
      lines: L,
      names,
      stars: [explicit[0] !== null ? explicit[0] : stars[0],
              explicit[1] !== null ? explicit[1] : stars[1]],
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


def star_val(n):
    return n if isinstance(n, int) and 1 <= n <= 5 else None


def make_key(league, round_no, home, away, event_id, day):
    raw = (f"{league}|{event_id}" if event_id
           else f"{league}|{day}|{round_no}|{home}|{away}")
    return hashlib.sha1(raw.encode()).hexdigest()


async def best_frame(page):
    """Frame (or main page) with the most decimal odds in its text."""
    best, best_n = page.main_frame, -1
    for fr in page.frames:
        try:
            txt = await fr.evaluate("document.body ? document.body.innerText : ''")
            n = len(re.findall(ODD_PAT.replace("(?<![\\d.])", "(?<![\\d.])"), txt))
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


async def find_rows(frame, mode):
    return await frame.evaluate(FIND_ROWS_JS, {
        "mode": mode, "oddPat": ODD_PAT, "scorePat": SCORE_PAT, "noise": NOISE.pattern})


async def page_text(frame):
    try:
        return await frame.evaluate("document.body.innerText")
    except Exception:
        return ""


def detect_league(text, default):
    for lg in LEAGUES:
        if re.search(lg, text, re.I):
            return lg
    return default


# ---------- parsing ----------
def parse_fixture(row, league, round_no, day):
    home, away = (clean_name(n) for n in (row["names"] + ["", ""])[:2])
    if not home or not away:
        return None
    odds = [float(o) for o in re.findall(ODD_PAT, row["text"]) if float(o) >= 1.01]
    rec = {"league": league, "home_team": home, "away_team": away}
    m = ID_RE.search(row["text"])
    rec["event_id"] = m.group(1) if m else None
    rm = ROUND_RE.search(row["text"])
    rec["round_no"] = int(rm.group(1)) if rm else round_no
    rec["home_stars"] = star_val(row["stars"][0])
    rec["away_stars"] = star_val(row["stars"][1])
    for field, val in zip(ODD_FIELDS, odds):
        rec[field] = val
    rec["_n_odds"] = len(odds)
    rec["fixture_key"] = make_key(league, rec["round_no"], home, away, rec["event_id"], day)
    return rec


def parse_result(row, league, round_no, day):
    home, away = (clean_name(n) for n in (row["names"] + ["", ""])[:2])
    sc = next((re.match(SCORE_PAT, l) for l in row["lines"] if re.match(SCORE_PAT, l)), None)
    if not home or not away or not sc:
        return None
    hg, ag = int(sc.group(1)), int(sc.group(2))
    m = ID_RE.search(row["text"])
    rm = ROUND_RE.search(row["text"])
    rn = int(rm.group(1)) if rm else round_no
    eid = m.group(1) if m else None
    return {"league": league, "home_team": home, "away_team": away, "event_id": eid,
            "round_no": rn, "home_score": hg, "away_score": ag,
            "outcome": "H" if hg > ag else "A" if hg < ag else "D",
            "fixture_key": make_key(league, rn, home, away, eid, day)}


# ---------- scraping ----------
async def scrape():
    fixtures, results = {}, {}
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
            frame, n = None, 0
            for _ in range(15):                      # wait up to ~45s for odds to render
                await asyncio.sleep(3)
                frame, n = await best_frame(page)
                if n >= 3:
                    break
            log(f"[diag] page title: {await page.title()!r}; url: {page.url}")
            log(f"[diag] frames: {len(page.frames)}; odds-like numbers in best frame: {n}")
            if n < 3:
                raise RuntimeError("No decimal odds found on page (blocked / geo-restricted / not loaded?)")

            for league in LEAGUES:
                clicked = await click_label(page, rf"^\s*{league}\b")
                if not clicked:
                    log(f"[warn] could not click a '{league}' tab; will detect league from page text")
                frame, _ = await best_frame(page)
                txt = await page_text(frame)
                rm = ROUND_RE.search(txt)
                round_no = int(rm.group(1)) if rm else None
                if round_no is None:
                    log(f"[warn] {league}: no 'Round N' text found on page; round_no left null")
                lg = league if clicked else detect_league(txt, "Unknown")

                rows = await find_rows(frame, "fixtures")
                log(f"[diag] {lg}: {len(rows)} fixture rows found")
                for r in rows[:2]:
                    log(f"[diag] sample row lines: {r['lines'][:14]} stars={r['stars']}")
                for r in rows:
                    rec = parse_fixture(r, lg, round_no, day)
                    if rec:
                        fixtures[rec["fixture_key"]] = rec

                if SCRAPE_RESULTS:
                    if await click_label(page, r"^\s*(results?|history|settled)\s*$"):
                        frame, _ = await best_frame(page)
                        rrows = await find_rows(frame, "results")
                        log(f"[diag] {lg}: {len(rrows)} result rows found")
                        for r in rrows[:2]:
                            log(f"[diag] sample result lines: {r['lines'][:10]}")
                        for r in rrows:
                            rec = parse_result(r, lg, round_no, day)
                            if rec:
                                results[rec["fixture_key"]] = rec
                        await click_label(page, r"^\s*(fixtures?|upcoming|matches|games|schedule)\s*$")
                    else:
                        log("[warn] no Results/History tab found; skipping results")
        except Exception as e:
            log(f"[error] {e}")
            await page.screenshot(path="debug.png", full_page=True)
            with open("debug.html", "w", encoding="utf-8") as fh:
                fh.write(await page.content())
            try:
                fr, _ = await best_frame(page)
                log("[diag] page text excerpt:\n" + (await page_text(fr))[:1500])
            except Exception:
                pass
            raise
        finally:
            await browser.close()
    return list(fixtures.values()), list(results.values())


# ---------- database ----------
def clean_for_db(rows):
    out = []
    for r in rows:
        r = {k: v for k, v in r.items() if v is not None and not k.startswith("_")}
        out.append(r)   # None-valued keys dropped so upserts never overwrite with NULL
    return out


def batch_upsert(client, rows, label):
    groups = {}
    for r in rows:
        groups.setdefault(tuple(sorted(r)), []).append(r)
    for batch in groups.values():
        for i in range(0, len(batch), 200):
            client.table(TABLE).upsert(batch[i:i + 200], on_conflict="fixture_key").execute()
    print(f"Upserted {len(rows)} {label}")


def link_results(client, results):
    """Results without an event ID: attach to the newest un-scored fixture with same teams."""
    for r in results:
        if r.get("event_id"):
            continue
        try:
            q = (client.table(TABLE).select("fixture_key")
                 .eq("home_team", r["home_team"]).eq("away_team", r["away_team"])
                 .is_("home_score", "null").order("scraped_at", desc=True).limit(1).execute())
            if q.data:
                r["fixture_key"] = q.data[0]["fixture_key"]
            else:
                log(f"[warn] no matching fixture for result {r['home_team']} v {r['away_team']}; saving standalone")
        except Exception as e:
            log(f"[warn] result linking failed: {e}")


def main():
    fixtures, results = asyncio.run(scrape())

    # ---- diagnostics summary ----
    n = len(fixtures)
    if n:
        log(f"[diag] fixtures: {n}; missing stars: {sum(1 for f in fixtures if f['home_stars'] is None)}; "
            f"missing round: {sum(1 for f in fixtures if f['round_no'] is None)}; "
            f"missing event_id: {sum(1 for f in fixtures if not f['event_id'])}")
        counts = {}
        for f in fixtures:
            counts[f["_n_odds"]] = counts.get(f["_n_odds"], 0) + 1
        log(f"[diag] odds-per-row distribution (expect 7 for full 1X2+OU+GG/NG): {counts}")
        if any(c in counts for c in (4, 6)) or any(c > 7 for c in counts):
            log("[warn] unusual odds counts - mapping may be misaligned; check sample rows above")
    print("=== DIAGNOSTICS ===")
    print("\n".join(DIAG))
    print("===================")
    print(f"Scraped {n} fixtures, {len(results)} results")

    if not fixtures and not results:
        sys.exit("Nothing scraped - read DIAGNOSTICS above")

    now = datetime.now(timezone.utc).isoformat()
    for f in fixtures:
        f["scraped_at"] = now
    for r in results:
        r["result_scraped_at"] = now

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    if fixtures:
        batch_upsert(client, clean_for_db(fixtures), "fixtures")
    if results:
        link_results(client, results)
        batch_upsert(client, clean_for_db(results), "results")


if __name__ == "__main__":
    main()
