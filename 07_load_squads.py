"""
load_squads.py
--------------
Loads squad data from squads/ipl_2026_squads.json and patches the IPL_SCHEDULE
in 07_app.py so every match has the correct player lists.

Editing squads
--------------
  1. Open squads/ipl_2026_squads.json
  2. Edit the "bat" and "bowl" arrays for each team
  3. Player names MUST exactly match player_profiles.pkl keys
     (run the check command above to see valid names)
  4. Restart Flask — squads are loaded at startup

Usage
-----
  # Auto-called by 07_app.py at startup — no manual action needed.
  # To reload squads without restarting Flask: POST /api/reload_squads
"""

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)

SQUADS_FILE = Path(__file__).parent / "input" / "iplsquads" / "ipl_2026_squads.json"


def load_ipl_squads() -> dict:
    """
    Returns { "RCB": {"bat": [...], "bowl": [...]}, ... }
    or empty dict if file not found.
    """
    if not SQUADS_FILE.exists():
        log.warning("ipl_2026_squads.json not found — using hardcoded squads")
        return {}
    try:
        with open(SQUADS_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        # Strip metadata keys starting with _
        squads = {k: v for k, v in data.items() if not k.startswith("_")}
        log.info("Loaded squads for %d teams from %s", len(squads), SQUADS_FILE)
        return squads
    except Exception as exc:
        log.error("Failed to load squads JSON: %s", exc)
        return {}


def patch_schedule_squads(schedule: list[dict], squads: dict) -> list[dict]:
    """
    For each match in schedule, fill in squad data from the squads dict.
    Existing squad data is overwritten so the JSON file is always authoritative.
    """
    if not squads:
        return schedule
    patched = 0
    for match in schedule:
        home = match.get("home", "")
        away = match.get("away", "")
        if home in squads or away in squads:
            match.setdefault("squads", {})
            if home in squads:
                match["squads"][home] = squads[home]
            if away in squads:
                match["squads"][away] = squads[away]
            patched += 1
    log.info("Patched squads for %d/%d matches", patched, len(schedule))
    return schedule


def validate_squads(squads: dict, bat_pool: dict, bowl_pool: dict) -> dict:
    """
    Check which player names in the squads JSON are NOT in the ML model pools.
    Returns { "team": {"bat": [unknown...], "bowl": [unknown...]} }
    Useful for debugging mismatched player names.
    """
    issues = {}
    for team, sq in squads.items():
        unknown_bat  = [p for p in sq.get("bat",  []) if p not in bat_pool]
        unknown_bowl = [p for p in sq.get("bowl", []) if p not in bowl_pool]
        if unknown_bat or unknown_bowl:
            issues[team] = {}
            if unknown_bat:  issues[team]["bat_not_in_model"]  = unknown_bat
            if unknown_bowl: issues[team]["bowl_not_in_model"] = unknown_bowl
    return issues
