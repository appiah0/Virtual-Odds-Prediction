"""
SportyBet virtual football scraper -> Supabase.

Env vars (required): SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY
Env vars (optional): SPORTYBET_URL, HEADLESS (default "true")

!! The CSS selectors below are PLACEHOLDERS. Open the virtual lobby in
Chrome DevTools, inspect a fixture row, and update the SELECTORS dict.
On failure the script saves debug.png / debug.html so you can inspect
what the runner actually saw (uploaded as a workflow artifact).
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
LEAGUES = ["Germany", "Spain"]  # leagues to capture (matched case-insensitively)

SELECTORS = {
    # Virtual games often render inside an iframe; set to a CSS selector
    # like "iframe[src*='virtual']" if rows are not found on the main page.
    "iframe": None,
    "league_tab": "[class*='league'], [class*='tab']",  # clickable league tabs
    "row": "[class*='match-row'], [class*='event-row'], [class*='fixture']",
    "star": "[class*='star'][class*='active'], [class*='star'][class*='filled']",
    "odds": "[class*='odd'] span, [class*='outcome'] span",
}

ODD_RE = re.compile(r"^\d{1,3}\.\d{2}$")


async def get_context(page):
    """Return the page or the iframe containing the fixtures."""
    sel = SELECTORS["iframe"]
    if not sel:
        return page
    await page.wait_for_selector(sel, timeout=30000)
    handle = await page.query_selector(sel)
    frame = await handle.content_frame()
    return frame or page


async def parse_rows(ctx, league):
    rows = await ctx.query_selector_all(SELECTORS["row"])
    fixtures = []
    for row in rows:
        try:
            text = (await row.inner_text()).strip()
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            odds = [l for l in lines if ODD_RE.match(l)]
            names = [l for l in lines if not ODD_RE.match(l) and not re.fullmatch(r"[\d:\-\s]+", l)]
            if len(odds) < 3 or len(names) < 2:
                continue
            stars = len(await row.query_selector_all(SELECTORS["star"]))
            stars = max(1, min(5, stars)) if stars else None
            home, away = names[0], names[1]
            fixtures.append({
                "league": league,
                "home_team": home,
                "away_team": away,
                "stars": stars,
                "odds_home": float(odds[0]),
                "odds_draw": float(odds[1]),
                "odds_away": float(odds[2]),
            })
        except Exception as e:  # skip a bad row, keep going
            print(f"row skipped: {e}", file=sys.stderr)
    return fixtures


def add_keys(fixtures):
    """Deterministic key so re-runs in the same hour upsert, not duplicate."""
    hour = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H")
    now = datetime.now(timezone.utc).isoformat()
    for f in fixtures:
        raw = f"{hour}|{f['league']}|{f['home_team']}|{f['away_team']}"
        f["fixture_key"] = hashlib.sha1(raw.encode()).hexdigest()
        f["scraped_at"] = now
    return fixtures


async def scrape():
    all_fixtures = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=HEADLESS)
        context = await browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            locale="en-NG",
        )
        page = await context.new_page()
        try:
            await page.goto(URL, wait_until="domcontentloaded", timeout=60000)
            ctx = await get_context(page)
            await ctx.wait_for_selector(SELECTORS["row"], timeout=45000)

            for league in LEAGUES:
                tab = ctx.locator(SELECTORS["league_tab"], has_text=re.compile(league, re.I)).first
                try:
                    await tab.click(timeout=8000)
                    await page.wait_for_timeout(1500)
                except PWTimeout:
                    print(f"League tab '{league}' not found; parsing current view", file=sys.stderr)
                all_fixtures += await parse_rows(ctx, league)
        except Exception as e:
            print(f"Scrape failed: {e}", file=sys.stderr)
            await page.screenshot(path="debug.png", full_page=True)
            with open("debug.html", "w", encoding="utf-8") as fh:
                fh.write(await page.content())
            raise
        finally:
            await browser.close()
    return add_keys(all_fixtures)


def save(fixtures):
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_ROLE_KEY"])
    client.table("scraped_fixtures").upsert(fixtures, on_conflict="fixture_key").execute()


def main():
    fixtures = asyncio.run(scrape())
    print(f"Scraped {len(fixtures)} fixtures")
    if not fixtures:
        sys.exit("No fixtures found - check selectors / debug.html")
    save(fixtures)
    print("Upserted to Supabase")


if __name__ == "__main__":
    main()
