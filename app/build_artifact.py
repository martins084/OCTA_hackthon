"""Bundle the demo into one self-contained page (data embedded) for sharing as a hosted page.

Embeds trips.json (portfolio) and the full detail only for replay trips, so the page stays small.
Usage: python app/build_artifact.py [out.html]
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "static/data"


def main(out: Path):
    trips = json.loads((DATA / "trips.json").read_text())
    details = {t["id"]: json.loads((DATA / f"trip_{t['id']}.json").read_text()) for t in trips if t["tags"]["replay"]}
    page = (HERE / "static/index.html").read_text(encoding="utf-8")
    blob = json.dumps({"trips": trips, "details": details}, separators=(",", ":")).replace("</", r"<\/")
    marker = "<script>\n(() => {"
    assert marker in page, "script marker not found"
    page = page.replace(marker, f"<script>window.__DEMO_DATA__ = {blob};</script>\n{marker}", 1)
    out.write_text(page, encoding="utf-8")
    print(f"{out} ({out.stat().st_size / 1e6:.2f} MB, {len(details)} replay trips, {len(trips)} portfolio trips)")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "dist/octa_drive_score.html")
