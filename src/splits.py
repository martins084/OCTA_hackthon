"""Dev/blind split guard. Every loader goes through here; see docs/PLAN.md section 3.

Blind data is only reachable inside `blind_access()`, which src/blind_eval.py opens after
--confirm. Anything else that touches a blind driver, dataset or path raises BlindDataError.
"""
import json
from contextlib import contextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPLITS = json.loads((ROOT / "data_splits.json").read_text())
_BLIND_OPEN = False


class BlindDataError(RuntimeError):
    pass


@contextmanager
def blind_access():
    global _BLIND_OPEN
    _BLIND_OPEN = True
    try:
        yield
    finally:
        _BLIND_OPEN = False


def _check(is_blind: bool, what: str):
    if is_blind and not _BLIND_OPEN:
        raise BlindDataError(f"{what} is blind test data; only src/blind_eval.py may read it")


def uah_trip_dirs(root: Path, split: str = "dev") -> list[Path]:
    """UAH trip folders for 'dev' or 'blind'. Filters by driver id, so it is safe even when the
    blind driver sits in the same folder (as on Kaggle, where uah/ still contains D3)."""
    blind = set(SPLITS["uah"]["blind_drivers"])
    if split == "blind":
        _check(True, "UAH blind drivers")
        roots = [root, ROOT / SPLITS["uah"]["blind_path"]]
        want = blind
    else:
        roots = [root]
        want = set(SPLITS["uah"]["dev_drivers"])
    dirs = {d.resolve() for r in roots for d in r.glob("D*/*/") if d.parent.name in want}
    return sorted(dirs)


def event_dataset(name: str) -> str:
    """Call before loading an event dataset by name; raises if it is blind."""
    _check(name in SPLITS["events"]["blind_datasets"], f"event dataset '{name}'")
    return name


def assert_path_allowed(path: Path):
    p = Path(path).resolve()
    for key in ("uah", "events", "own_recordings"):
        bp = (ROOT / SPLITS[key]["blind_path"]).resolve()
        _check(bp == p or bp in p.parents, str(p))
