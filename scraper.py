"""
SportyBet virtual football scraper v2 -> Supabase (ML feature set).

Env (required): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Env (optional): SPORTYBET_URL, HEADLESS ("true"), SCRAPE_RESULTS ("true")

!! All CSS selectors / tab labels in CONFIG are PLACEHOLDERS. Inspect the
live lobby in Chrome DevTools and fill them in. On any failure the script
writes debug.png + debug.html (the workflow uploads them as an artifact).
"""
import asyncio
import hashlib
import os
import re
import sys
from datetime import datetime, timezone

from playwright.async_api import async_playwright, TimeoutError as PWTimeout
from supabase import create_client

URL = os.environ.get("SPORTYBET_URL", "https://www.sportybet.com/ng/sport/vFootball")
HEADLESS = os.environ.get("HEADLESS", "true").lower() != "false"
SCRAPE_RESULTS = os.environ.get("SCRAPE_RESULTS", "true").lower() != "false"
LEAGUES = ["Germany", "Spain"]
TABLE = "scraped_fixtures"

CONFIG = {
    "iframe": None,                                  # e.g. "iframe[src*='virtual']"
    "league_tab": "[class*='league'], [class*='tab']",
    "results_tab_text": "Results",                   # label of the results/history tab
    "fixtures_tab_text": "Fixtures",                 # label to return to upcoming games
    "round": "[class*='round'], [class*='matchday']",  # element containing round number
    # --- upcoming fixture rows ---
    "row": "[class*='match-row'], [class*='event-row']",
    "home_box": "[class*='home']",                   # team container inside a row
    "away_box": "[class*='away']",
    "team_name": "[class*='name'], span",            # name element inside a team box
    "star": "[class*='star'][class*='active'], [class*='star'][class*='filled']",
    "odds_cell": "[class*='odd'] span, [class*='outcome'] span",
    # Market groups. "tab": text of a market tab to click first (None = already
    # visible in the same row). "fields": odds cells in on-screen order.
    "market_groups": [
        {"tab": None, "fields": ["odds_home", "odds_draw", "odds_away",
                                 "odds_over_25", "odds_under_25",
                                 "odds_gg", "odds_ng"]},
        # Example if markets live in separate tabs:
        # {"tab": "1X2", "fields": ["odds_home", "odds_draw", "odds_away"]},
        # {"tab": "Over/Under", "fields": ["odds_over_25", "odds_under_25"]},
        # {"tab": "GG/NG", "fields": ["odds_gg", "odds_ng"]},
    ],
    # --- results rows ---
    "result_row": "[class*='result-row'], [class*='history-row']",
}

ODD_RE = re.compile(r"^\d{1,3}\.\d{2}$")
SCORE_RE = re.compile(r"(\d+)\s*[-:]\s*(\d+)")
ID_RE = re.compile(r"ID\s*[:#]?\s*(\d+)", re.I)


# ---------- helpers ----------
def clean_name(raw: str) -> str:
    """Strip 'ID: 12345', stray digits/odds, and whitespace from a team name."""
    s = ID_RE.sub("", raw)
    s = re.sub(r"\b\d+(\.\d+)?\b", "", s)
    return re.sub(r"\s+", " ", s).strip(" -:|")


def extract_id(raw: str):
    m = ID_RE.search(raw)
    return m.group(1) if m else None


def make_key(league, round_no, home, away, event_id, day):
    if event_id:
        raw = f"{league}|{event_id}"
    else:  # fallback: round numbers repeat, so include the UTC date
        raw = f"{league}|{day}|{round_no}|{home}|{away}"
    return hashlib.sha1(raw.encode()).hexdigest()


async def get_ctx(page):
    if not CONFIG["iframe"]:
        return page
    await page.wait_for_selector(CONFIG["iframe"], timeout=30000)
    handle = await page.query_selector(CONFIG["iframe"])
    return (await handle.content_frame()) or page


async def click_text(ctx, selector, text, timeout=8000):
    loc = ctx.locator(selector, has_text=re.compile(re.escape(text), re.I)).first
    try:
        await loc.click(timeout=timeout)
        await asyncio.sleep(1.2)
        return True
    except PWTimeout:
        print(f"tab '{text}' not found", file=sys.stderr)
        return False


async def read_round(ctx):
    try:
        el = await ctx.query_selector(CONFIG["round"])
        if el:
            m = re.search(r"\d+", await el.inner_text())
            return int(m.group()) if m else None
    except Exception:
        pass
    return None


async def team_info(row, box_sel):
    """Return (clean_name, stars, event_id) for one team box."""
    box = await row.query_selector(box_sel)
    if not box:
        return None, None, None
    raw_full = await box.inner_text()
    name_el = await box.query_selector(CONFIG["team_name"])
    raw_name = (await name_el.inner_text()) if name_el else raw_full
    stars = len(await box.query_selector_all(CONFIG["star"]))
    stars = stars if 1 <= stars <= 5 else None
    return clean_name(raw_name), stars, extract_id(raw_full)


# ---------- fixtures ----------
async def parse_fixture_rows(ctx, league, round_no, fields, day):
    out = {}
    for row in await ctx.query_selector_all(CONFIG["row"]):
        try:
            home, hs, hid = await team_info(row, CONFIG["home_box"])
            away, as_, aid = await team_info(row, CONFIG["away_box"])
            if not home or not away:
                continue
            cells = [(await c.inner_text()).strip()
                     for c in await row.query_selector_all(CONFIG["odds_cell"])]
            odds = [float(c) for c in cells if ODD_RE.match(c)]
            rec = {
                "league": league, "round_no": round_no,
                "home_team": home, "away_team": away,
                "home_stars": hs, "away_stars": as_,
                "event_id": hid or aid,
            }
            for name, val in zip(fields, odds):
                rec[name] = val
            rec["fixture_key"] = make_key(league, round_no, home, away, rec["event_id"], day)
            out[rec["fixture_key"]] = rec
        except Exception as e:
            print(f"fixture row skipped: {e}", file=sys.stderr)
    return out


async def scrape_fixtures(ctx, league, round_no, day):
    merged = {}
    for group in CONFIG["market_groups"]:
        if group["tab"]:
            await click_text(ctx, CONFIG["league_tab"], group["tab"])
        rows = await parse_fixture_rows(ctx, league, round_no, group["fields"], day)
        for key, rec in rows.items():
            merged.setdefault(key, {}).update(rec)
    return list(merged.values())


# ---------- results ----------
async def scrape_results(ctx, league, day):
    out = []
    for row in await ctx.query_selector_all(CONFIG["result_row"]):
        try:
            text = await row.inner_text()
            m = SCORE_RE.search(text)
            if not m:
                continue
            home, _, hid = await team_info(row, CONFIG["home_box"])
            away, _, aid = await team_info(row, CONFIG["away_box"])
            if not home or not away:
                continue
            hg, ag = int(m.group(1)), int(m.group(2))
            rm = re.search(r"(?:round|rd|md|matchday)\s*\.?\s*(\d+)", text, re.I)
            round_no = int(rm.group(1)) if rm else None
            event_id = hid or aid
            out.append({
                "league": league, "round_no": round_no,
                "home_team": home, "away_team": away, "event_id": event_id,
                "home_score": hg, "away_score": ag,
                "outcome": "H" if hg > ag else "A" if hg < ag else "D",
                "fixture_key": make_key(league, round_no, home, away, event_id, day),
            })
        except Exception as e:
            print(f"result row skipped: {e}", file=sys.stderr)
    return out


# ---------- orchestration ----------
async def scrape():
    fixtures, results = [], []
    day = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900}, locale="en-NG",
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"))
        page = await context.new_page()
        try:
            await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            ctx = await get_ctx(page)
            await ctx.wait_for_selector(CONFIG["row"], timeout=45000)

            for league in LEAGUES:
                await click_text(ctx, CONFIG["league_tab"], league)
                round_no = await read_round(ctx)
                fixtures += await scrape_fixtures(ctx, league, round_no, day)

                if SCRAPE_RESULTS and await click_text(ctx, CONFIG["league_tab"], CONFIG["results_tab_text"]):
                    await ctx.wait_for_selector(CONFIG["result_row"], timeout=20000)
                    results += await scrape_results(ctx, league, day)
                    await click_text(ctx, CONFIG["league_tab"], CONFIG["fixtures_tab_text"])
        except Exception as e:
            print(f"Scrape failed: {e}", file=sys.stderr)
            await page.screenshot(path="debug.png", full_page=True)
            with open("debug.html", "w", encoding="utf-8") as fh:
                fh.write(await page.content())
            raise
        finally:
            await browser.close()
    return fixtures, results


def batch_upsert(client, rows, label):
    """Upsert in groups that share identical column sets, so a partial
    payload (e.g. results only) never overwrites other columns with NULL."""
    groups = {}
    for r in rows:
        groups.setdefault(tuple(sorted(r)), []).append(r)
    for batch in groups.values():
        for i in range(0, len(batch), 200):
            client.table(TABLE).upsert(batch[i:i + 200], on_conflict="fixture_key").execute()
    print(f"Upserted {len(rows)} {label}")


def main():
    fixtures, results = asyncio.run(scrape())
    print(f"Scraped {len(fixtures)} fixtures, {len(results)} results")
    if not fixtures and not results:
        sys.exit("Nothing scraped - check CONFIG selectors / debug.html")

    now = datetime.now(timezone.utc).isoformat()
    for f in fixtures:
        f["scraped_at"] = now
    for r in results:
        r["result_scraped_at"] = now

    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    if fixtures:
        batch_upsert(client, fixtures, "fixtures")
    if results:
        batch_upsert(client, results, "results")


if __name__ == "__main__":
    main()
