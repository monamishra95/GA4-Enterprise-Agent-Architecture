"""
GA4 Enterprise Agent Architecture — Traffic Spawner
====================================================
Simulates two distinct traffic profiles using Playwright:

  simulate_human_traffic()  — realistic mouse movement, scroll, reading pauses
  simulate_llm_scraper()    — instant DOM extraction, zero interaction signals

Loops ITERATIONS times, randomly selecting between profiles at a 60/40 ratio.
Run with: python scripts/traffic_generator.py

Dependencies: pip install playwright && playwright install chromium
"""

import asyncio
import random
from datetime import datetime
from playwright.async_api import async_playwright

# ─────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────

# [PRODUCTION] Replace with your staging or production URL
TARGET_URL = "http://localhost:8080/docs/index.html"

ITERATIONS = 50           # Total sessions to simulate
HUMAN_RATIO = 0.60        # 60% human, 40% bot

# Simulated CPC values (for spend-saved reporting)
SIMULATED_CPC_USD = 2.40

# ─────────────────────────────────────────────────────────────
# HUMAN TRAFFIC SIMULATION
# Mimics a real user: natural mouse paths, reading pauses, scroll depth
# ─────────────────────────────────────────────────────────────

async def simulate_human_traffic(page, session_id: str):
    """
    Simulates a real human browsing session with:
    - Full page load wait (networkidle)
    - Random mouse movements across the viewport
    - Gradual scroll down (simulates reading each section)
    - Configurable reading pause
    - Scroll back up (simulates re-reading headline)

    In production: these behavioral signals feed the Edge Score model
    via Server-Side GTM → GA4 Measurement Protocol.
    """
    print(f"  [HUMAN:{session_id}] Loading page...")
    await page.goto(TARGET_URL, wait_until="networkidle")
    await page.wait_for_timeout(1000 + random.randint(0, 800))  # Initial load pause

    viewport = page.viewport_size or {"width": 1280, "height": 800}
    w, h = viewport["width"], viewport["height"]

    # Natural, non-linear mouse movement
    n_moves = random.randint(5, 10)
    print(f"  [HUMAN:{session_id}] Simulating {n_moves} mouse movements...")
    for _ in range(n_moves):
        x = random.randint(80, w - 80)
        y = random.randint(80, h - 80)
        await page.mouse.move(x, y)
        await page.wait_for_timeout(random.randint(150, 500))

    # Gradual scroll — simulates reading each dashboard section
    scroll_steps = random.randint(4, 7)
    print(f"  [HUMAN:{session_id}] Scrolling through {scroll_steps} content sections...")
    for step in range(1, scroll_steps + 1):
        await page.evaluate(f"window.scrollTo({{ top: {step * 320}, behavior: 'smooth' }})")
        await page.wait_for_timeout(random.randint(700, 1500))

    # Reading pause — longest delay, simulating content consumption
    reading_pause = random.randint(2500, 5000)
    print(f"  [HUMAN:{session_id}] Reading pause: {reading_pause}ms")
    await page.wait_for_timeout(reading_pause)

    # Scroll back up (common real-user behavior)
    await page.evaluate("window.scrollTo({ top: 0, behavior: 'smooth' })")
    await page.wait_for_timeout(600)

    # [PRODUCTION] At this point, Server-Side GTM would have captured:
    # - High edge score (>0.7) from Cloud Armor
    # - Multiple scroll_depth events in GA4
    # - session_engaged = true
    # - conversion_value = $150.00
    print(f"  [HUMAN:{session_id}] ✓ Session complete — high edge score, full scroll depth recorded")


# ─────────────────────────────────────────────────────────────
# LLM SCRAPER / BOT SIMULATION
# Mimics AI agents: instant DOM extraction, zero interaction
# ─────────────────────────────────────────────────────────────

async def simulate_llm_scraper(page, session_id: str):
    """
    Simulates an LLM scraper or automated bot:
    - domcontentloaded only (no waiting for JS/images)
    - Immediately extracts full page text
    - Zero mouse movement, zero scroll events
    - Sub-500ms total session duration

    Detection signals in production:
      - Edge Score < 0.4 (Cloud Armor / reCAPTCHA)
      - User-Agent matches known bot signatures
      - Zero mouse_move events in session
      - event_velocity > 15 events/sec (page_view + instant close)
      - session_duration_sec < 5

    Result: GA4 Measurement Protocol fires bot_traffic_detected
    event with conversion_value = $0.00 for VBB.
    """
    print(f"  [BOT:{session_id}]   Loading page (domcontentloaded only)...")
    await page.goto(TARGET_URL, wait_until="domcontentloaded")

    # Bots don't wait — minimal delay before extraction
    await page.wait_for_timeout(random.randint(150, 400))

    # Instant full-DOM text extraction (classic LLM scraper pattern)
    extracted = await page.evaluate("document.body.innerText")
    word_count = len(extracted.split())

    print(f"  [BOT:{session_id}]   Extracted {word_count} words in < 400ms — flagged as LLM scraper")

    # [PRODUCTION] This pattern triggers the server-side detection pipeline:
    #   Cloud Armor assigns edge_score < 0.4
    #   Server-Side GTM fires bot_traffic_detected MP event
    #   conversion_value set to $0.00 → Google Ads ignores click for bidding
    print(f"  [BOT:{session_id}]   Simulated MP event: bot_traffic_detected | conv_value = $0.00 | spend_saved = ${SIMULATED_CPC_USD:.2f}")


# ─────────────────────────────────────────────────────────────
# MAIN SIMULATION LOOP
# ─────────────────────────────────────────────────────────────

async def run_traffic_loop():
    """
    Runs ITERATIONS sessions, alternating between human and bot
    profiles at the configured HUMAN_RATIO.

    headless=False keeps the browser window visible — required for
    live demos and pitch presentations.
    """
    human_count, bot_count, spend_saved = 0, 0, 0.0

    print("\n" + "=" * 60)
    print("  GA4 Enterprise Agent Architecture — Traffic Spawner")
    print(f"  Target URL : {TARGET_URL}")
    print(f"  Iterations : {ITERATIONS}  ({int(HUMAN_RATIO*100)}% human / {int((1-HUMAN_RATIO)*100)}% bot)")
    print("=" * 60 + "\n")

    async with async_playwright() as p:
        # headless=False: browser window visible for live demo
        browser = await p.chromium.launch(headless=False, slow_mo=50)

        for i in range(1, ITERATIONS + 1):
            is_human = random.random() < HUMAN_RATIO
            profile = "HUMAN" if is_human else "BOT"
            sid = f"{i:02d}"

            print(f"[{datetime.now().strftime('%H:%M:%S')}] ── Session {i}/{ITERATIONS} ── {profile} ──")

            context = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent=(
                    # Human UA: realistic Chrome on macOS
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    if is_human else
                    # Bot UA: GPTBot signature — detected by Cloud Armor in production
                    "GPTBot/1.0 (+https://openai.com/gptbot)"
                ),
                # Bots typically don't send referrer headers
                extra_http_headers={} if is_human else {"Referer": ""}
            )
            page = await context.new_page()

            try:
                if is_human:
                    await simulate_human_traffic(page, sid)
                    human_count += 1
                else:
                    await simulate_llm_scraper(page, sid)
                    bot_count += 1
                    spend_saved += SIMULATED_CPC_USD

            except Exception as e:
                print(f"  [ERROR] Session {i} failed: {e}")

            finally:
                await page.wait_for_timeout(400)
                await context.close()

            # Inter-session delay: humans have longer gaps, bots are rapid-fire
            delay = random.uniform(1.5, 3.5) if is_human else random.uniform(0.1, 0.6)
            print(f"  → Next session in {delay:.1f}s\n")
            await asyncio.sleep(delay)

        await browser.close()

    # Final summary
    print("\n" + "=" * 60)
    print("  Simulation Complete — Summary")
    print("=" * 60)
    print(f"  Total Sessions  : {ITERATIONS}")
    print(f"  Human Sessions  : {human_count} ({human_count/ITERATIONS:.0%})")
    print(f"  Bot Sessions    : {bot_count}  ({bot_count/ITERATIONS:.0%})")
    print(f"  Ad Spend Saved  : ${spend_saved:.2f} (algorithmic $0 bidding)")
    print("=" * 60)
    print("\n  → Check your GA4 Command Center dashboard for live updates.")


if __name__ == "__main__":
    asyncio.run(run_traffic_loop())
