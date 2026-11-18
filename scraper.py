"""
SportyBet virtual football scraper v5 - built from the real page DOM.

Real structure (from your log):
  .match-row
    .clock-time "10:08"      .game-id "ID: 32654"
    .home-team / .away-team
    .m-market[data-op="...event_sr:match:<SRID>_market_1"]    -> 1X2 (3 odds)
    .m-market[data-op="...<SRID>_market_18"]                  -> Over/Under, line in
                                                                 .af-select-input (dropdown!)
    .market-size "+45"       -> opens the full market list
  League headers look like "England Virtual".

What this version does
  * Reads rows by those classes (no guessing).
  * Over/Under: reads the dropdown line; if it is not 2.5 it opens the dropdown
    and picks 2.5. The line and the displayed odds are always stored too
    (ou_line, odds_over, odds_under) so a wrong line can never be mislabeled.
  * GG/NG: opens "+45" per match and reads market 29 (best effort, capped,
    stops after 3 consecutive failures). The first probe logs what it saw.
  * Results: opens SportyBet's results page, tries to switch it to Virtual,
    keeps only lines whose ID is one of OUR stored virtual fixtures, and
    updates those rows by event_id.
  * Stars / round: NOT present in the lobby row DOM. They stay null; the
    "+45" probe logs whether the detail view has anything star-like.

Env (required): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Env (optional): SPORTYBET_URL, HEADLESS, SCRAPE_RESULTS, ENRICH_DETAIL,
                ENRICH_MAX (45), LEAGUES (comma list; empty = all found)
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
BASE = URL.split("/ng/")[0]
RESULTS_URL = os.environ.get("RESULTS_URL", BASE + "/ng/liveResult/")
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"
SCRAPE_RESULTS = os.environ.get("SCRAPE_RESULTS", "true").lower() != "false"
ENRICH_DETAIL = os.environ.get("ENRICH_DETAIL", "true").lower() != "false"
ENRICH_MAX = int(os.environ.get("ENRICH_MAX", "45"))
LEAGUES = [x.strip() for x in os.environ.get("LEAGUES", "").split(",") if x.strip()]
TABLE = "scraped_fixtures"

RESULT_LINE = re.compile(
    r"^\s*(\d{4,8})\s*\t+\s*(.+?)\s*\t+\s*(\d{1,2})\s*[:\-\u2013]\s*(\d{1,2})"
    r"\s*(?:\([^)]*\))?\s*\t+\s*(.+?)\s*$")
DIAG = []


def log(msg):
    print(msg, file=sys.stderr)
    DIAG.append(msg)


# ---------- in-browser extraction (real selectors) ----------
ROWS_JS = r"""
() => {
  const HDR = /^[A-Za-z][A-Za-z .&'-]{1,30}\s+Virtual$/i;
  const headers = [];
  const tw = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  let tn;
  while ((tn = tw.nextNode())) {
    const t = tn.textContent.trim();
    if (t.length < 5 || t.length > 40 || !HDR.test(t)) continue;
    const p = tn.parentElement;
    if (!p || p.closest('button,a,[role=tab],[role=button],nav')) continue;
    headers.push({node: tn, text: t});
  }
  return [...document.querySelectorAll('.match-row')].map(row => {
    const t = s => { const e = row.querySelector(s); return e ? e.textContent.trim() : null; };
    const opEl = row.querySelector('[data-op*="event_sr:match:"]');
    const markets = [...row.querySelectorAll('.m-market')].map(m => {
      const mid = ((m.getAttribute('data-op') || '').match(/_market_(\d+)/) || [])[1] || null;
      const sel = m.querySelector('.af-select-input');
      return {id: mid,
              line: sel ? sel.textContent.replace('^', '').trim() : null,
              odds: [...m.querySelectorAll('.m-outcome-odds')].map(e => e.textContent.trim())};
    });
    let hint = null;
    for (const h of headers)
      if (h.node.compareDocumentPosition(row) & Node.DOCUMENT_POSITION_FOLLOWING) hint = h.text;
    return {home: t('.home-team'), away: t('.away-team'), gameId: t('.game-id'),
            time: t('.clock-time'), op: opEl ? opEl.getAttribute('data-op') : null,
            hint, markets};
  });
}
"""


# ---------- helpers ----------
def num(s):
    m = re.search(r"\d+(?:\.\d+)?", s or "")
    return float(m.group()) if m else None


def odd(s):
    v = num(s)
    return v if v is not None and v >= 1.01 else None


def clean_name(raw):
    s = re.sub(r"ID\s*[:#]?\s*\d+", "", raw or "", flags=re.I)
    s = re.sub(r"\b\d+(\.\d+)?\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" -:|()[]")


def league_name(hint):
    if not hint:
        return "Unknown"
    return re.sub(r"\s*virtual\s*", "", hint, flags=re.I).strip().title() or "Unknown"


def make_key(sr_id, event_id, league, home, away, day):
    raw = (f"sr|{sr_id}" if sr_id else f"id|{event_id}" if event_id
           else f"{league}|{day}|{home}|{away}")
    return hashlib.sha1(raw.encode()).hexdigest()


def parse_row(r, day):
    home, away = clean_name(r.get("home")), clean_name(r.get("away"))
    if not home or not away:
        return None
    event_id = (re.search(r"\d+", r.get("gameId") or "") or [None])[0]
    sr_m = re.search(r"sr:match:(\d+)", r.get("op") or "")
    sr_id = sr_m.group(1) if sr_m else None
    league = league_name(r.get("hint"))
    rec = {"league": league, "home_team": home, "away_team": away,
           "event_id": event_id, "sr_match_id": sr_id,
           "kickoff_time": re.sub(r"\s+", " ", (r.get("time") or "").replace("\xa0", " ")).strip() or None}
    by_id = {m["id"]: m for m in r.get("markets", []) if m.get("id")}
    m1 = by_id.get("1")
    if m1 and len(m1["odds"]) >= 3:
        rec["odds_home"], rec["odds_draw"], rec["odds_away"] = (odd(x) for x in m1["odds"][:3])
    m18 = by_id.get("18")
    if m18 and len(m18["odds"]) >= 2:
        line = num(m18.get("line"))
        rec["ou_line"] = line
        rec["odds_over"], rec["odds_under"] = odd(m18["odds"][0]), odd(m18["odds"][1])
        if line == 2.5:
            rec["odds_over_25"], rec["odds_under_25"] = rec["odds_over"], rec["odds_under"]
    m29 = by_id.get("29")
    if m29 and len(m29["odds"]) >= 2:
        rec["odds_gg"], rec["odds_ng"] = odd(m29["odds"][0]), odd(m29["odds"][1])
    rec["fixture_key"] = make_key(sr_id, event_id, league, home, away, day)
    return rec


def parse_results(lines, known_ids):
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


async def all_text_lines(page):
    lines = []
    for fr in page.frames:
        try:
            lines += (await fr.evaluate("document.body ? document.body.innerText : ''")).split("\n")
        except Exception:
            continue
    return lines


def row_locator(page, event_id):
    return page.locator(".match-row", has_text=re.compile(rf"ID:\s*{event_id}\b")).first


async def wait_rows(page, timeout=30000):
    await page.wait_for_selector(".match-row", timeout=timeout)
    await asyncio.sleep(1)


# ---------- Over/Under 2.5 via the line dropdown ----------
async def pick_line_25(page, rec, probe):
    row = row_locator(page, rec["event_id"])
    mk = row.locator("[data-op$='_market_18']").first
    await mk.locator(".af-select-title").first.click(timeout=4000)
    await asyncio.sleep(0.5)
    if probe:
        log(f"[diag] O/U dropdown HTML after opening:\n{(await mk.inner_html())[:1200]}")
    opt = None
    for scope in (mk, page.locator("[class*='af-select']")):
        cand = scope.get_by_text(re.compile(r"^\s*2\.5\s*$")).first
        if await cand.count() and await cand.is_visible():
            opt = cand
            break
    if not opt:
        await page.keyboard.press("Escape")
        return False
    await opt.click(timeout=4000)
    await asyncio.sleep(0.6)
    texts = await mk.locator(".m-outcome-odds").all_inner_texts()
    line = num(await mk.locator(".af-select-input").first.inner_text())
    if line == 2.5 and len(texts) >= 2 and odd(texts[0]) and odd(texts[1]):
        rec["ou_line"] = 2.5
        rec["odds_over"] = rec["odds_over_25"] = odd(texts[0])
        rec["odds_under"] = rec["odds_under_25"] = odd(texts[1])
        return True
    return False


# ---------- GG/NG (and probe for stars) via "+45" ----------
async def enrich_detail(page, rec, probe):
    before = page.url
    row = row_locator(page, rec["event_id"])
    await row.locator(".market-size").first.click(timeout=4000)
    await asyncio.sleep(2.5)
    sr = rec.get("sr_match_id")
    ops = await page.eval_on_selector_all(
        "[data-op*='_market_']", "els => els.map(e => e.getAttribute('data-op'))")
    mids = sorted({m for o in ops if sr and sr in (o or "") for m in re.findall(r"_market_(\d+)", o)},
                  key=int)
    if probe:
        star_like = await page.evaluate(
            "() => document.querySelectorAll(\"[class*='star' i],[class*='rating' i]\").length")
        txt = (await page.evaluate("document.body.innerText"))[:700]
        log(f"[diag] +45 probe: url_changed={page.url != before}; market ids for this match={mids}; "
            f"star/rating-like elements on page={star_like}\n[diag] +45 page text start:\n{txt}")
    ok = False
    if sr:
        mk = page.locator(f"[data-op*='{sr}'][data-op$='_market_29']").first
        if await mk.count():
            odds = await mk.locator(".m-outcome-odds").all_inner_texts()
            if len(odds) >= 2 and odd(odds[0]) and odd(odds[1]):
                rec["odds_gg"], rec["odds_ng"] = odd(odds[0]), odd(odds[1])
                ok = True
    # return to the list
    if page.url != before:
        await page.go_back()
        await wait_rows(page)
    else:
        await page.keyboard.press("Escape")
        await asyncio.sleep(0.5)
    return ok


# ---------- results ----------
async def scrape_results(page, known_ids):
    await page.goto(RESULTS_URL, wait_until="domcontentloaded", timeout=60000)
    await asyncio.sleep(6)

    async def attempt(label):
        lines = await all_text_lines(page)
        res, dropped = parse_results(lines, known_ids)
        log(f"[diag] results ({label}): {len(res)} virtual kept, {dropped} non-virtual dropped; url={page.url}")
        return res, lines

    res, lines = await attempt("initial")
    if res:
        return res
    log("[diag] results page top text:\n" + "\n".join(l.strip() for l in lines if l.strip())[:600])
    labels = await page.evaluate("""() => [...new Set([...document.querySelectorAll(
        "li,[role=option],[role=tab],button,[class*=tab],[class*=option],[class*=select]")]
        .filter(e => e.offsetParent !== null).map(e => e.innerText.trim().split('\\n')[0])
        .filter(t => t && t.length < 30))].slice(0, 40)""")
    log(f"[diag] clickable labels on results page: {labels}")

    tried, opened = set(), False
    for _ in range(5):
        clicked = False
        cands = page.get_by_text(re.compile(r"virtual", re.I))
        for i in range(await cands.count()):
            el = cands.nth(i)
            try:
                if not await el.is_visible():
                    continue
                label = (await el.inner_text()).strip()[:40]
                if label in tried:
                    continue
                tried.add(label)
                log(f"[diag] clicking results control {label!r}")
                await el.click(timeout=4000)
                await asyncio.sleep(3)
                clicked = True
                break
            except Exception:
                continue
        if clicked:
            res, _ = await attempt(f"after clicking {sorted(tried)[-1]!r}")
            if res:
                return res
        elif not opened:        # open a sport dropdown, then look for "Virtual" again
            opened = True
            try:
                dd = page.get_by_text(re.compile(r"^\s*(football|soccer|all sports?)\s*$", re.I)).first
                await dd.click(timeout=4000)
                await asyncio.sleep(1.5)
            except Exception:
                break
        else:
            break
    log("[warn] could not reach a virtual results view; see the labels above and tell me which to click")
    return []


# ---------- orchestration ----------
async def scrape(db_known_ids):
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 900}, locale="en-NG",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        page = await ctx.new_page()
        recs, results = [], []
        try:
            await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            await wait_rows(page, 60000)
            raw = await page.evaluate(ROWS_JS)
            log(f"[diag] title={await page.title()!r}; .match-row count={len(raw)}")
            seen = {}
            for r in raw:
                rec = parse_row(r, day)
                if rec:
                    seen.setdefault(rec["fixture_key"], rec)
            recs = list(seen.values())

            # --- Over/Under 2.5 ---
            need = [r for r in recs if r.get("ou_line") is not None and r["ou_line"] != 2.5 and r["event_id"]]
            good = bad = 0
            for i, rec in enumerate(need):
                try:
                    if await pick_line_25(page, rec, probe=(i == 0)):
                        good += 1
                    else:
                        bad += 1
                except Exception as e:
                    bad += 1
                    log(f"[warn] line select failed for {rec['event_id']}: {str(e)[:80]}")
            log(f"[diag] O/U 2.5 selected for {good}/{len(need)} rows ({bad} failed)")

            # --- GG/NG via +45 ---
            if ENRICH_DETAIL:
                fails, done, got = 0, 0, 0
                for rec in recs[:ENRICH_MAX]:
                    if not rec["event_id"]:
                        continue
                    try:
                        ok = await enrich_detail(page, rec, probe=(done == 0))
                        got += ok
                        fails = 0 if ok else fails + 1
                    except Exception as e:
                        fails += 1
                        log(f"[warn] +45 failed for {rec['event_id']}: {str(e)[:80]}")
                        try:
                            await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
                            await wait_rows(page)
                        except Exception:
                            break
                    done += 1
                    if fails >= 3:
                        log("[warn] 3 consecutive GG/NG failures - stopping detail scrape")
                        break
                log(f"[diag] GG/NG captured for {got}/{done} matches opened")

            if SCRAPE_RESULTS:
                known = set(db_known_ids) | {r["event_id"] for r in recs if r["event_id"]}
                results = await scrape_results(page, known)
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

    dist = {}
    for f in fixtures:
        dist[f["league"]] = dist.get(f["league"], 0) + 1
    log(f"[diag] leagues found: {dist}")
    if LEAGUES:
        want = {x.lower() for x in LEAGUES}
        kept = [f for f in fixtures if f["league"].lower() in want]
        log(f"[diag] LEAGUES filter {LEAGUES}: kept {len(kept)} of {len(fixtures)}")
        fixtures = kept
    if fixtures:
        cnt = lambda k: sum(1 for f in fixtures if f.get(k) is not None)
        log(f"[diag] filled: 1X2={cnt('odds_home')} O/U2.5={cnt('odds_over_25')} "
            f"GG/NG={cnt('odds_gg')} kickoff={cnt('kickoff_time')} of {len(fixtures)}")

    print("=== DIAGNOSTICS ===")
    print("\n".join(DIAG))
    print("===================")
    print(f"Scraped {len(fixtures)} fixtures, {len(results)} virtual results")
    if not fixtures and not results:
        sys.exit("Nothing scraped - read DIAGNOSTICS above")

    now = datetime.now(timezone.utc).isoformat()
    clean = []
    for f in fixtures:
        r = {k: v for k, v in f.items() if v is not None}
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
        res = (client.table(TABLE).update(r).eq("event_id", eid)
               .is_("home_score", "null").execute())
        updated += len(res.data or [])
    print(f"Updated {updated} fixture rows with results")


if __name__ == "__main__":
    main()
