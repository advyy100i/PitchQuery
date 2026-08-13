"""Cached, polite HTTP fetcher for the StatsBomb open-data repo.

Every file is written to disk on first fetch and read from disk after that, so
re-running any script costs zero bandwidth. Never clone the repo: with the 360
files it is multiple GB and we need a slice.
"""
import json
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.config import RAW_DIR, SB_BASE  # noqa: E402

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "pitchquery/0.1 (portfolio project; open data)"


def get_json(rel_path: str, *, cache: bool = True, retries: int = 4):
    """rel_path is relative to the open-data `data/` dir, e.g. 'events/3788741.json'.

    Returns the parsed JSON, or None if the file does not exist upstream (404).
    """
    local = RAW_DIR / rel_path
    if cache and local.exists() and local.stat().st_size > 0:
        with open(local, "r", encoding="utf-8") as f:
            return json.load(f)

    url = f"{SB_BASE}/{rel_path}"
    delay = 1.0
    for attempt in range(retries):
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            print(f"  ! {e} -> retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay *= 2
            continue

        if r.status_code == 404:
            return None
        if r.status_code == 200:
            data = r.json()
            if cache:
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(r.text, encoding="utf-8")
            return data
        if attempt == retries - 1:
            r.raise_for_status()
        print(f"  ! HTTP {r.status_code} -> retry in {delay:.0f}s", file=sys.stderr)
        time.sleep(delay)
        delay *= 2
    return None


def competitions():
    return get_json("competitions.json")


def matches(competition_id, season_id):
    return get_json(f"matches/{competition_id}/{season_id}.json") or []


def events(match_id):
    return get_json(f"events/{match_id}.json")


def three_sixty(match_id):
    return get_json(f"three-sixty/{match_id}.json")


def lineups(match_id):
    return get_json(f"lineups/{match_id}.json")
