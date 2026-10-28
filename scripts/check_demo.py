"""Render the demo in headless Chromium: console errors, overflow, screenshots (desktop + phone, light + dark)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000/"
OUT = Path(__file__).resolve().parents[1] / "data/cache/shots"
OUT.mkdir(parents=True, exist_ok=True)

with sync_playwright() as p:
    b = p.chromium.launch()
    for name, vp, scheme in [("desktop_light", (1440, 1000), "light"), ("desktop_dark", (1440, 1000), "dark"),
                             ("phone_light", (390, 844), "light")]:
        pg = b.new_page(viewport=dict(width=vp[0], height=vp[1]), color_scheme=scheme)
        errs = []
        pg.on("console", lambda m: errs.append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
        pg.on("pageerror", lambda e: errs.append(f"pageerror: {e}"))
        pg.goto(URL, wait_until="networkidle")
        pg.wait_for_timeout(1200)
        over = pg.evaluate("document.documentElement.scrollWidth - document.documentElement.clientWidth")
        pg.screenshot(path=str(OUT / f"{name}_replay.png"), full_page=False)
        if name == "desktop_light":
            pg.click("#play"); pg.wait_for_timeout(2500)
            pg.screenshot(path=str(OUT / f"{name}_playing.png"))
            pg.click(".seg button[data-v=snowy_night]"); pg.click("#end"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}_snowy.png"))
            pg.click("#tab-portfolio"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}_portfolio.png"), full_page=True)
            pg.click("#tab-evidence"); pg.wait_for_timeout(300)
            pg.screenshot(path=str(OUT / f"{name}_evidence.png"), full_page=True)
            pg.click("#tab-replay")
        print(f"{name}: horizontal overflow {over}px; console: {errs or 'clean'}")
    b.close()
