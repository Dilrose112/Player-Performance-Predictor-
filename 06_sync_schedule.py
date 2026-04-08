"""
06_sync_schedule.py
--------------------
Auto-syncs the IPL 2026 and T20I match schedule by:

  1. Scraping ESPNcricinfo for completed match results and adding them to
     schedule entries as `result` objects.
  2. Fetching upcoming fixtures not yet in the schedule and appending them.
  3. Persisting the merged schedule to `output/schedule.json` so Flask
     can load it on startup instead of using the hardcoded lists in 06_app.py.

Usage
-----
  # Run once to bootstrap
  python 07_sync_schedule.py

  # Or import and call from Flask for on-demand refresh
  from sync_schedule import sync_all
  sync_all()

The Flask app (06_app.py) should be updated to:
  - Load schedule.json at startup (falling back to hardcoded lists if absent)
  - Expose  GET /api/sync  to trigger a refresh on demand
  - Return  result  data alongside each match in  /api/schedule

ESPN Series IDs (confirmed April 2026)
  IPL 2026  → 1449924
  T20I (India home) → varies — we scrape the team pages directly
"""

import json
import os
import re
import time
import logging
from datetime import datetime, date
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# ── Config ────────────────────────────────────────────────────────────────────

BASE_DIR    = Path(__file__).parent
OUTPUT_DIR  = BASE_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)
SCHEDULE_FILE = OUTPUT_DIR / "schedule.json"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

ESPN_BASE   = "https://www.espncricinfo.com"
IPL_SERIES  = "1449924"   # IPL 2026
CRICKETDATA_BASE = "https://api.cricapi.com/v1"
CRICKETDATA_API_KEY = os.getenv("CRICAPI_KEY") or os.getenv("CRICKETDATA_API_KEY") or "83130e0e-f545-4b7b-8f76-d34c6f4715bc"
CRICKETDATA_IPL_SERIES_ID = "87c62aac-bc3c-4738-ab93-19da0690488f"

# Polite delay between requests (seconds)
SCRAPE_DELAY = 1.5

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


# ── Hardcoded seed schedule (from 06_app.py) ─────────────────────────────────
# These are the fallback entries; synced data is merged on top of them.
# We import via importlib since the filename starts with a digit.

import importlib.util as _ilu
import sys as _sys

def _import_app():
    spec = _ilu.spec_from_file_location(
        "app06", Path(__file__).parent / "06_app.py"
    )
    mod = _ilu.module_from_spec(spec)
    _sys.modules["app06"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass   # app needs Flask/pickle at runtime; we only need the schedule lists
    return mod

_app = _import_app()
IPL_SCHEDULE  = getattr(_app, "IPL_SCHEDULE",  [])
T20I_SCHEDULE = getattr(_app, "T20I_SCHEDULE", [])


# ── ESPN helpers ──────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3) -> BeautifulSoup | None:
    """Fetch a URL and return a BeautifulSoup, or None on failure."""
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=15)
            r.raise_for_status()
            time.sleep(SCRAPE_DELAY)
            return BeautifulSoup(r.text, "html.parser")
        except requests.RequestException as exc:
            log.warning("Attempt %d failed for %s: %s", attempt + 1, url, exc)
            time.sleep(SCRAPE_DELAY * (attempt + 1))
    log.error("All retries exhausted for %s", url)
    return None


def _cricketdata_get(endpoint: str, **params) -> dict | None:
    """Call CricketData/CricAPI and return parsed JSON, or None on failure."""
    if not CRICKETDATA_API_KEY:
        log.warning("CRICAPI_KEY / CRICKETDATA_API_KEY is not set")
        return None

    url = f"{CRICKETDATA_BASE}/{endpoint.lstrip('/')}"
    query = {"apikey": CRICKETDATA_API_KEY, "offset": 0, **params}
    try:
        r = requests.get(url, params=query, timeout=20)
        r.raise_for_status()
        payload = r.json()
    except requests.RequestException as exc:
        log.warning("CricketData request failed for %s: %s", url, exc)
        return None
    except ValueError as exc:
        log.warning("CricketData returned invalid JSON for %s: %s", url, exc)
        return None

    status = str(payload.get("status", "")).lower()
    if status and status not in {"success", "true"}:
        log.warning("CricketData returned non-success for %s: %s", endpoint, payload)
        return None
    return payload


def _parse_date(raw: str) -> str | None:
    """
    Parse messy ESPN date strings into YYYY-MM-DD.
    Examples handled:
      'Sat, 28 Mar 2026'
      'March 28, 2026'
      '28 Mar'  (current year assumed)
    """
    raw = raw.strip()
    if not raw:
        return None
    if "T" in raw:
        raw = raw.replace("Z", "+00:00")
        try:
            return datetime.fromisoformat(raw).strftime("%Y-%m-%d")
        except ValueError:
            pass
    for fmt in (
        "%a, %d %b %Y", "%B %d, %Y", "%d %b %Y",
        "%d %b",        "%b %d, %Y", "%Y-%m-%d",
    ):
        try:
            parsed = datetime.strptime(raw, fmt)
            # If year missing (e.g. "%d %b"), use current year
            if parsed.year == 1900:
                parsed = parsed.replace(year=date.today().year)
            return parsed.strftime("%Y-%m-%d")
        except ValueError:
            continue
    log.debug("Could not parse date: %r", raw)
    return None


def _normalise_team(name: str) -> str:
    """Map ESPN full team names to the short codes used in our schedule."""
    mapping = {
        "Royal Challengers Bengaluru": "RCB",
        "Royal Challengers Bangalore": "RCB",
        "Royal Challengers Bengaluru Women": "RCB",
        "Royal Challengers Bangalore Women": "RCB",
        "RCBW": "RCB",
        "Chennai Super Kings": "CSK",
        "Mumbai Indians": "MI",
        "Mumbai Indians Women": "MI",
        "Kolkata Knight Riders": "KKR",
        "kolkata knight riders": "KKR",
        "Sunrisers Hyderabad": "SRH",
        "Rajasthan Royals": "RR",
        "Gujarat Titans": "GT",
        "Punjab Kings": "PBKS",
        "Delhi Capitals": "DC",
        "Delhi Capitals Women": "DC",
        "Lucknow Super Giants": "LSG",
        # T20I nations — keep as-is
    }
    return mapping.get(name.strip(), name.strip())


def _normalise_venue(name: str) -> str:
    """Normalise venue names to match our player-profile keys."""
    venue_map = {
        "M.Chinnaswamy Stadium":            "M Chinnaswamy Stadium",
        "M Chinnaswamy Stadium, Bengaluru": "M Chinnaswamy Stadium",
        "Wankhede Stadium, Mumbai":         "Wankhede Stadium",
        "Eden Gardens, Kolkata":            "Eden Gardens",
        "Rajiv Gandhi Intl. Cricket Stadium": "Rajiv Gandhi International Stadium",
        "Sawai Mansingh Stadium, Jaipur":   "Sawai Mansingh Stadium",
        "Narendra Modi Stadium, Ahmedabad": "Narendra Modi Stadium",
        "Punjab Cricket Association IS Bindra Stadium": "Punjab Cricket Association Stadium",
        "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium":
            "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "BRSABV Ekana Cricket Stadium":
            "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
        "Melbourne Cricket Ground, Melbourne": "Melbourne Cricket Ground",
    }
    for k, v in venue_map.items():
        if k.lower() in name.lower() or name.lower() in k.lower():
            return v
    return name.strip()


def _iter_candidate_matches(obj):
    """Yield likely match dicts from nested CricketData payloads."""
    if isinstance(obj, list):
        for item in obj:
            yield from _iter_candidate_matches(item)
        return
    if not isinstance(obj, dict):
        return

    keys = set(obj.keys())
    if (
        {"name", "date"}.issubset(keys)
        or {"teams", "date"}.issubset(keys)
        or {"teamInfo", "date"}.issubset(keys)
    ):
        yield obj

    for key in ("data", "matches", "matchList", "match_list", "fixtures"):
        value = obj.get(key)
        if value:
            yield from _iter_candidate_matches(value)


def _pick_first_str(*values: object) -> str:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _cricketdata_team_name(team: object) -> str:
    if isinstance(team, dict):
        return _normalise_team(
            _pick_first_str(
                team.get("shortname"),
                team.get("shortName"),
                team.get("name"),
            )
        )
    if isinstance(team, str):
        return _normalise_team(team)
    return ""


def _cricketdata_match_status(match: dict) -> tuple[str, dict | None]:
    raw_status = _pick_first_str(
        match.get("status"),
        match.get("matchStatus"),
        match.get("state"),
    )
    lowered = raw_status.lower()
    if match.get("matchEnded") or any(word in lowered for word in ("won", "result", "tied", "draw", "abandoned")):
        return "completed", {"summary": raw_status}
    if any(word in lowered for word in ("live", "inning", "stumps", "need", "require")):
        return "live", None
    return "upcoming", None


def _cricketdata_match_to_schedule(match: dict, match_no: int) -> dict | None:
    teams = []
    if isinstance(match.get("teamInfo"), list):
        teams = [_cricketdata_team_name(team) for team in match["teamInfo"][:2]]
    elif isinstance(match.get("teams"), list):
        teams = [_cricketdata_team_name(team) for team in match["teams"][:2]]

    teams = [team for team in teams if team]
    if len(teams) < 2:
        return None

    match_date = _parse_date(
        _pick_first_str(match.get("date"), match.get("dateTimeGMT"), match.get("dateTime"))
    ) or ""
    venue = _normalise_venue(
        _pick_first_str(
            match.get("venue"),
            match.get("matchVenue"),
            match.get("venueName"),
        )
    )
    status, result = _cricketdata_match_status(match)
    if result is not None:
        score_lines = []
        score_value = match.get("score")
        if isinstance(score_value, list):
            for score in score_value:
                if isinstance(score, dict):
                    score_lines.append(
                        _pick_first_str(
                            score.get("r"),
                            score.get("inning"),
                            score.get("score"),
                        )
                    )
        result = {
            "winner": "",
            "margin": "",
            "home_score": score_lines[0] if len(score_lines) > 0 else "",
            "away_score": score_lines[1] if len(score_lines) > 1 else "",
            "summary": result.get("summary", ""),
        }

    return {
        "espn_id": _pick_first_str(match.get("id"), match.get("matchId")),
        "match": match_no,
        "date": match_date,
        "home": teams[0],
        "away": teams[1],
        "venue": venue,
        "format": "ipl",
        "status": status,
        "result": result,
    }


def fetch_ipl_schedule_from_cricketdata(series_id: str = CRICKETDATA_IPL_SERIES_ID) -> list[dict]:
    """Fetch IPL fixtures/results from CricketData/CricAPI using JSON endpoints."""
    matches: list[dict] = []
    seen_ids = set()
    payloads = []

    series_payload = _cricketdata_get("series_info", id=series_id)
    if series_payload:
        payloads.append(series_payload)

    matches_payload = _cricketdata_get("matches")
    if matches_payload:
        payloads.append(matches_payload)

    current_payload = _cricketdata_get("currentMatches")
    if current_payload:
        payloads.append(current_payload)

    series_name_hints = ("indian premier league", "ipl 2026", "indian premier league 2026")

    for payload in payloads:
        for raw_match in _iter_candidate_matches(payload):
            series_name = _pick_first_str(
                raw_match.get("series"),
                raw_match.get("seriesName"),
            ).lower()
            series_info = raw_match.get("seriesInfo")
            if isinstance(series_info, dict):
                series_name = _pick_first_str(
                    series_name,
                    series_info.get("name"),
                    series_info.get("series"),
                ).lower()
                raw_series_id = _pick_first_str(series_info.get("id"))
            else:
                raw_series_id = _pick_first_str(raw_match.get("series_id"), raw_match.get("seriesId"))

            belongs_to_series = raw_series_id == series_id or any(hint in series_name for hint in series_name_hints)
            if not belongs_to_series:
                continue

            parsed = _cricketdata_match_to_schedule(raw_match, len(matches) + 1)
            if not parsed:
                continue
            unique_id = parsed["espn_id"] or f"{parsed['date']}|{parsed['home']}|{parsed['away']}"
            if unique_id in seen_ids:
                continue
            seen_ids.add(unique_id)
            matches.append(parsed)

    matches.sort(key=lambda m: m.get("date", "9999-12-31"))
    for i, match in enumerate(matches, start=1):
        match["match"] = i

    log.info("Found %d IPL matches from CricketData", len(matches))
    return matches


# ── ESPN Schedule / Results scraper ───────────────────────────────────────────

def fetch_ipl_schedule_from_espn(series_id: str = IPL_SERIES) -> list[dict]:
    """
    Scrape IPL match list from the ESPN series schedule page.

    Returns a list of dicts:
      {
        'espn_id': str,          # ESPN match ID
        'match': int,            # match number (1-indexed in ESPN title)
        'date': 'YYYY-MM-DD',
        'home': 'RCB',
        'away': 'SRH',
        'venue': '...',
        'format': 'ipl',
        'status': 'completed' | 'upcoming' | 'live',
        'result': {              # only if completed
            'winner': 'RCB',
            'margin': '5 wickets',
            'home_score': '180/4 (20)',
            'away_score': '175/6 (20)',
        }
      }
    """
    url = f"{ESPN_BASE}/series/{series_id}/match-schedule-fixtures-and-results"
    log.info("Fetching IPL schedule from %s", url)
    soup = _get(url)
    if soup is None:
        return []

    matches = []

    # ESPN renders match cards as <article> or <div> with data-testid attributes
    # Structure varies by ESPN version — we target the most stable selectors
    cards = soup.find_all("div", attrs={"data-testid": "match-summary"})
    if not cards:
        # Fallback: look for anchor tags with /matches/ in href
        cards = soup.find_all("a", href=re.compile(r"/matches/\d+/"))

    match_no = 0
    seen_ids  = set()

    for card in cards:
        try:
            match_no += 1
            entry = _parse_ipl_card(card, match_no, series_id, seen_ids)
            if entry:
                seen_ids.add(entry["espn_id"])
                matches.append(entry)
        except Exception as exc:
            log.debug("Card parse error: %s", exc)
            continue

    log.info("Found %d IPL matches from ESPN", len(matches))
    return matches


def _parse_ipl_card(card, match_no: int, series_id: str, seen_ids: set) -> dict | None:
    """Extract fields from a single ESPN match card element."""
    # Try to find match URL → ESPN ID
    link = card if card.name == "a" else card.find("a", href=re.compile(r"/matches/\d+/"))
    if not link:
        return None

    href    = link.get("href", "")
    id_match = re.search(r"/matches/(\d+)/", href)
    if not id_match:
        return None
    espn_id = id_match.group(1)
    if espn_id in seen_ids:
        return None

    # Date
    date_tag = card.find(attrs={"data-testid": "match-date"}) or \
               card.find(class_=re.compile(r"date|time", re.I))
    raw_date = date_tag.get_text(strip=True) if date_tag else ""
    parsed_date = _parse_date(raw_date) or ""

    # Teams
    team_tags = card.find_all(attrs={"data-testid": "team-name"})
    if len(team_tags) < 2:
        team_tags = card.find_all(class_=re.compile(r"team-name|TeamName", re.I))
    teams = [_normalise_team(t.get_text(strip=True)) for t in team_tags[:2]]
    if len(teams) < 2:
        return None
    home, away = teams[0], teams[1]

    # Venue
    venue_tag = card.find(attrs={"data-testid": "match-venue"}) or \
                card.find(class_=re.compile(r"venue|ground", re.I))
    venue = _normalise_venue(venue_tag.get_text(strip=True) if venue_tag else "")

    # Status + result
    status = "upcoming"
    result = None

    score_tags = card.find_all(attrs={"data-testid": "score"})
    if not score_tags:
        score_tags = card.find_all(class_=re.compile(r"score", re.I))

    result_tag = card.find(attrs={"data-testid": "match-result"}) or \
                 card.find(class_=re.compile(r"result|status", re.I))
    result_text = result_tag.get_text(strip=True) if result_tag else ""

    if result_text and any(w in result_text.lower() for w in ("won", "tied", "no result")):
        status = "completed"
        scores = [s.get_text(strip=True) for s in score_tags[:2]]
        winner_text = ""
        for t in teams:
            if t.lower() in result_text.lower() or \
               _normalise_team(t).lower() in result_text.lower():
                winner_text = t
                break
        result = {
            "winner":     winner_text,
            "margin":     _extract_margin(result_text),
            "home_score": scores[0] if len(scores) > 0 else "",
            "away_score": scores[1] if len(scores) > 1 else "",
            "summary":    result_text,
        }
    elif score_tags:
        status = "live"

    return {
        "espn_id":  espn_id,
        "match":    match_no,
        "date":     parsed_date,
        "home":     home,
        "away":     away,
        "venue":    venue,
        "format":   "ipl",
        "status":   status,
        "result":   result,
    }


def _extract_margin(result_text: str) -> str:
    """Extract win margin from 'RCB won by 5 wickets' → '5 wickets'."""
    m = re.search(r"won by (.+?)(?:\s*\(|$)", result_text, re.I)
    return m.group(1).strip() if m else ""


def fetch_match_result_from_espn(espn_id: str) -> dict | None:
    """
    Scrape a single completed match page for scorecard data.

    Returns:
      {
        'winner': 'RCB',
        'margin': '5 wickets',
        'home_score': '180/4 (20)',
        'away_score': '175/6 (20)',
        'summary': '...',
        'top_performers': [
            {'name': 'V Kohli', 'runs': 72, 'balls': 48, 'role': 'BAT'},
            {'name': 'JJ Bumrah', 'wickets': 3, 'runs': 22, 'role': 'BOWL'},
        ]
      }
    """
    url = f"{ESPN_BASE}/matches/{espn_id}/full-scorecard"
    log.info("Fetching scorecard for match %s", espn_id)
    soup = _get(url)
    if soup is None:
        return None

    result: dict = {
        "winner": "", "margin": "", "home_score": "",
        "away_score": "", "summary": "", "top_performers": [],
    }

    # Result summary (usually in <p class="status-text"> or similar)
    status_el = soup.find(class_=re.compile(r"status-text|result-summary|match-result", re.I))
    if status_el:
        result["summary"] = status_el.get_text(strip=True)
        result["margin"]  = _extract_margin(result["summary"])

    # Innings scores
    innings_headers = soup.find_all(class_=re.compile(r"innings-header|InningsHeader", re.I))
    scores = []
    for ih in innings_headers[:2]:
        score_el = ih.find(class_=re.compile(r"score|runs", re.I))
        if score_el:
            scores.append(score_el.get_text(strip=True))
    if len(scores) >= 2:
        result["home_score"] = scores[0]
        result["away_score"] = scores[1]
    elif len(scores) == 1:
        result["home_score"] = scores[0]

    # Top performers — bat
    bat_rows = soup.find_all("tr", class_=re.compile(r"batsman|batter", re.I))
    for row in bat_rows[:6]:
        cols = row.find_all("td")
        if len(cols) >= 3:
            name = cols[0].get_text(strip=True)
            try:
                runs  = int(cols[1].get_text(strip=True))
                balls = int(cols[2].get_text(strip=True))
                if runs >= 20:
                    result["top_performers"].append(
                        {"name": name, "runs": runs, "balls": balls, "role": "BAT"}
                    )
            except ValueError:
                pass

    # Top performers — bowl
    bowl_rows = soup.find_all("tr", class_=re.compile(r"bowler", re.I))
    for row in bowl_rows[:6]:
        cols = row.find_all("td")
        if len(cols) >= 4:
            name = cols[0].get_text(strip=True)
            try:
                wickets = int(cols[3].get_text(strip=True))
                runs_c  = int(cols[2].get_text(strip=True))
                if wickets >= 1:
                    result["top_performers"].append(
                        {"name": name, "wickets": wickets, "runs": runs_c, "role": "BOWL"}
                    )
            except ValueError:
                pass

    return result


# ── Merge logic ───────────────────────────────────────────────────────────────

def _match_key(m: dict) -> tuple:
    """Canonical key for deduplicating matches."""
    return (m.get("date", ""), m.get("home", ""), m.get("away", ""))


def merge_schedules(seed: list[dict], scraped: list[dict]) -> list[dict]:
    """
    Merge scraped ESPN data into the seed schedule (from 06_app.py).

    Rules:
      - Seed entries without an ESPN counterpart are kept unchanged.
      - When an ESPN match matches a seed entry (by date + teams),
        we enrich the seed entry with:  espn_id, status, result.
      - ESPN entries not in the seed are appended as new entries.
    """
    seed_index = {_match_key(m): i for i, m in enumerate(seed)}
    merged     = [dict(m) for m in seed]   # shallow copy

    for scraped_m in scraped:
        key = _match_key(scraped_m)
        if key in seed_index:
            idx = seed_index[key]
            merged[idx]["espn_id"] = scraped_m.get("espn_id", "")
            merged[idx]["status"]  = scraped_m.get("status", "upcoming")
            if scraped_m.get("result"):
                merged[idx]["result"] = scraped_m["result"]
        else:
            # New match not in seed — append it
            new_entry = {
                "match":   len(merged) + 1,
                "date":    scraped_m["date"],
                "format":  scraped_m["format"],
                "home":    scraped_m["home"],
                "away":    scraped_m["away"],
                "venue":   scraped_m["venue"],
                "espn_id": scraped_m.get("espn_id", ""),
                "status":  scraped_m.get("status", "upcoming"),
                "squads":  {},   # unknown until we have squad data
                "result":  scraped_m.get("result"),
            }
            merged.append(new_entry)
            seed_index[key] = len(merged) - 1

    # Sort by date
    merged.sort(key=lambda m: m.get("date", "9999"))
    # Re-number match indices after sort
    for i, m in enumerate(merged, start=1):
        m["match"] = i

    return merged


def enrich_completed_matches(schedule: list[dict]) -> list[dict]:
    """
    CricketData already returns the status and basic score summary we need,
    so no extra per-match fetch is required here.
    """
    return schedule


# ── Main sync entrypoint ──────────────────────────────────────────────────────

def sync_all(enrich: bool = True) -> dict:
    """
    Full sync:
      1. Scrape IPL schedule from ESPN.
      2. Merge with seed schedule.
      3. Optionally enrich completed matches with scorecard data.
      4. Save to output/schedule.json.
      5. Return the final schedule dict.

    Returns:
      { 'ipl': [...], 't20i': [...] }
    """
    log.info("=== Starting schedule sync ===")

    # ── IPL ──
    scraped_ipl = fetch_ipl_schedule_from_cricketdata(CRICKETDATA_IPL_SERIES_ID)
    merged_ipl  = merge_schedules(list(IPL_SCHEDULE), scraped_ipl)
    if enrich:
        merged_ipl = enrich_completed_matches(merged_ipl)

    # ── T20I ──
    # T20I fixtures are more scattered; for now we use the seed and only
    # mark completed matches based on today's date as a best-effort fallback.
    # A future enhancement can scrape individual series pages.
    merged_t20i = list(T20I_SCHEDULE)
    today = date.today().isoformat()
    for m in merged_t20i:
        if "status" not in m:
            m["status"] = "completed" if m.get("date", "9999") < today else "upcoming"

    final = {"ipl": merged_ipl, "t20i": merged_t20i}

    # ── Persist ──
    with open(SCHEDULE_FILE, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2, default=str)
    log.info("Saved schedule to %s", SCHEDULE_FILE)
    log.info("IPL: %d matches | T20I: %d matches",
             len(merged_ipl), len(merged_t20i))

    return final


def load_schedule() -> dict:
    """
    Load schedule.json if it exists (and is <6 hours old).
    Otherwise fall back to the hardcoded seed lists.

    Call this from Flask instead of referencing IPL_SCHEDULE / T20I_SCHEDULE
    directly so the app always serves the freshest persisted data.
    """
    if SCHEDULE_FILE.exists():
        age_hours = (time.time() - SCHEDULE_FILE.stat().st_mtime) / 3600
        if age_hours < 6:
            with open(SCHEDULE_FILE, encoding="utf-8") as fh:
                data = json.load(fh)
            log.info("Loaded schedule from cache (%.1f h old)", age_hours)
            return data
        log.info("Cache is %.1f h old — will re-sync", age_hours)

    # Fallback to seed data (no ESPN scraping)
    log.info("No cache found — using seed schedule")
    today = date.today().isoformat()
    ipl  = [dict(m, status="completed" if m["date"] < today else "upcoming")
            for m in IPL_SCHEDULE]
    t20i = [dict(m, status="completed" if m["date"] < today else "upcoming")
            for m in T20I_SCHEDULE]
    return {"ipl": ipl, "t20i": t20i}


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Sync cricket match schedule")
    parser.add_argument("--no-enrich", action="store_true",
                        help="Skip per-match scorecard enrichment (faster)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print merged schedule without saving")
    args = parser.parse_args()

    schedule = sync_all(enrich=not args.no_enrich)

    if args.dry_run:
        print(json.dumps(schedule, indent=2, default=str))
    else:
        ipl_done = sum(1 for m in schedule["ipl"] if m.get("status") == "completed")
        ipl_up   = sum(1 for m in schedule["ipl"] if m.get("status") == "upcoming")
        print(f"\n✓ IPL:   {ipl_done} completed, {ipl_up} upcoming")
        t20_done = sum(1 for m in schedule["t20i"] if m.get("status") == "completed")
        t20_up   = sum(1 for m in schedule["t20i"] if m.get("status") == "upcoming")
        print(f"✓ T20I:  {t20_done} completed, {t20_up} upcoming")
        print(f"✓ Saved → {SCHEDULE_FILE}")
