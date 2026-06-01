"""
07_sync_schedule.py
--------------------
Syncs IPL 2026 + T20I schedule via cricapi.com v1 — rate-safe.

API call budget per run
-----------------------
  Phase 1  series search      1 call   (cached 24h in output/meta_cache.json)
  Phase 2  series_info        1 call   (same cache)
  Phase 3  matches (T20I)     1 call   (not paginated — one page is enough)
  Phase 4  scorecards         N calls  — at most --scorecard-limit per run
                                         (default 5), each cached forever in
                                         output/scorecard_cache/<id>.json
                                         so the same match is NEVER fetched twice
  TOTAL first run:  ~8 calls max
  TOTAL warm run:   3 calls  (series already cached, no new scorecards)

  Between every call: MIN_DELAY seconds (default 1.5s)
  On 429: exponential back-off, then skip and move on (never infinite retry)

Usage
-----
  python 07_sync_schedule.py                        # default: enrich up to 5 new scorecards
  python 07_sync_schedule.py --scorecard-limit 2    # only fetch 2 new scorecards this run
  python 07_sync_schedule.py --no-enrich            # skip all scorecard calls entirely
  python 07_sync_schedule.py --dry-run              # print merged JSON, do not save
  python 07_sync_schedule.py --clear-meta-cache     # force re-fetch series id
"""

import json
import logging
import re
import time
import importlib.util
import sys
from datetime import date, datetime
from pathlib import Path

import requests

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR        = Path(__file__).parent
OUTPUT_DIR      = BASE_DIR / "output"
SC_CACHE_DIR    = OUTPUT_DIR / "scorecard_cache"   # one JSON per match id
META_CACHE_FILE = OUTPUT_DIR / "meta_cache.json"   # series id + T20I list
SCHEDULE_FILE   = OUTPUT_DIR / "schedule.json"

for d in (OUTPUT_DIR, SC_CACHE_DIR):
    d.mkdir(exist_ok=True)

API_KEY         = "af880336-99e8-46a0-8f8d-c3293a43cc79"
BASE_URL        = "https://api.cricapi.com/v1"

MIN_DELAY       = 1.5    # seconds between every API call
META_CACHE_TTL  = 24     # hours before re-fetching series id / T20I list
SCHED_CACHE_TTL = 6      # hours before load_schedule() considers schedule.json stale
DEFAULT_SC_LIMIT = 5     # max new scorecard calls per run (CLI: --scorecard-limit)
IPL_MATCHES_CACHE_VERSION = "2026-known-teams-v3"
SERIES_SEARCH_OFFSETS = range(0, 201, 25)
MATCH_SEARCH_OFFSETS = range(0, 126, 25)

# Known fallback series IDs for IPL 2026
# Primary: cricapi.com series ID (try to discover dynamically)
# Fallback: use /v1/currentMatches to find live IPL matches
IPL_SERIES_ID   = "d5a498c8-7596-4b93-8ab0-e0efc3345312"  # may be stale, dynamic search preferred

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Seed schedule
# ─────────────────────────────────────────────────────────────────────────────

def _load_app():
    spec = importlib.util.spec_from_file_location("app06", BASE_DIR / "06_app.py")
    mod  = importlib.util.module_from_spec(spec)
    sys.modules["app06"] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        pass
    return mod

_app          = _load_app()
IPL_SCHEDULE  = getattr(_app, "IPL_SCHEDULE",  [])
T20I_SCHEDULE = getattr(_app, "T20I_SCHEDULE", [])

# ─────────────────────────────────────────────────────────────────────────────
# Rate-limited API helper
# ─────────────────────────────────────────────────────────────────────────────

_last_call_at: float = 0.0   # module-level timestamp of the last HTTP call

def _api(endpoint: str, params: dict | None = None) -> dict | None:
    """
    Single API call with:
      - Mandatory MIN_DELAY gap between calls (enforced globally via _last_call_at)
      - One retry on 429 with a 30-second back-off
      - Returns None on any failure instead of raising
    """
    global _last_call_at

    # Enforce minimum gap
    gap = time.time() - _last_call_at
    if gap < MIN_DELAY:
        time.sleep(MIN_DELAY - gap)

    url = f"{BASE_URL}/{endpoint}"
    p   = {"apikey": API_KEY, **(params or {})}

    for attempt in (1, 2):           # max 2 attempts per call
        try:
            _last_call_at = time.time()
            r = requests.get(url, params=p, timeout=15)

            if r.status_code == 429:
                if attempt == 1:
                    log.warning("429 rate-limit on %s — waiting 30s then retrying once", endpoint)
                    time.sleep(30)
                    continue
                else:
                    log.error("429 on retry for %s — skipping", endpoint)
                    return None

            r.raise_for_status()
            data = r.json()

            if data.get("status") == "failure":
                log.warning("API failure [%s]: %s", endpoint, data.get("reason", "?"))
                return None

            remaining = data.get("info", {}).get("hitsUsed")
            if remaining is not None:
                log.info("Credits used this period: %s", remaining)

            return data

        except requests.RequestException as exc:
            log.warning("Request error [%s] attempt %d: %s", endpoint, attempt, exc)
            if attempt == 1:
                time.sleep(MIN_DELAY * 2)

    return None

# ─────────────────────────────────────────────────────────────────────────────
# Meta cache  (series id + T20I fixture list)
# ─────────────────────────────────────────────────────────────────────────────

def _load_meta_cache() -> dict:
    if META_CACHE_FILE.exists():
        age_h = (time.time() - META_CACHE_FILE.stat().st_mtime) / 3600
        if age_h < META_CACHE_TTL:
            with open(META_CACHE_FILE, encoding="utf-8") as fh:
                return json.load(fh)
    return {}

def _save_meta_cache(data: dict):
    with open(META_CACHE_FILE, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# Scorecard disk cache  (one file per match, never expires — completed = final)
# ─────────────────────────────────────────────────────────────────────────────

def _sc_cached(cricapi_id: str) -> dict | None:
    path = SC_CACHE_DIR / f"{cricapi_id}.json"
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return None

def _sc_save(cricapi_id: str, data: dict):
    path = SC_CACHE_DIR / f"{cricapi_id}.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


def _load_existing_schedule() -> dict:
    """
    Load the current persisted schedule so sync can preserve completed match
    results even when the API returns bare fixtures or unresolved playoff rows.
    """
    if not SCHEDULE_FILE.exists():
        return {}
    try:
        with open(SCHEDULE_FILE, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Could not load existing schedule for preservation: %s", exc)
        return {}

# ─────────────────────────────────────────────────────────────────────────────
# Normalisation
# ─────────────────────────────────────────────────────────────────────────────

_TEAM_MAP = {
    "royal challengers bengaluru": "RCB",
    "royal challengers bangalore": "RCB",
    "chennai super kings":         "CSK",
    "mumbai indians":              "MI",
    "kolkata knight riders":       "KKR",
    "sunrisers hyderabad":         "SRH",
    "rajasthan royals":            "RR",
    "gujarat titans":              "GT",
    "punjab kings":                "PBKS",
    "delhi capitals":              "DC",
    "lucknow super giants":        "LSG",
}

_IPL_CODES = {"RCB", "CSK", "MI", "KKR", "SRH", "RR", "GT", "PBKS", "DC", "LSG"}
_PLACEHOLDER_TEAMS = {
    "",
    "tbd",
    "tbc",
    "to be decided",
    "to be confirmed",
    "winner of qualifier 1",
    "winner of qualifier 2",
    "loser of qualifier 1",
    "winner of eliminator",
}

_PLAYOFF_WORDS = ("qualifier", "eliminator", "final")

_VENUE_MAP = {
    "chinnaswamy":    "M Chinnaswamy Stadium",
    "wankhede":       "Wankhede Stadium",
    "eden gardens":   "Eden Gardens",
    "rajiv gandhi":   "Rajiv Gandhi International Stadium",
    "sawai mansingh": "Sawai Mansingh Stadium",
    "narendra modi":  "Narendra Modi Stadium",
    "is bindra":      "Punjab Cricket Association Stadium",
    "pca stadium":    "Punjab Cricket Association Stadium",
    "ekana":          "Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow",
    "arun jaitley":   "Arun Jaitley Stadium",
    "chidambaram":    "MA Chidambaram Stadium",
    "melbourne cricket ground": "Melbourne Cricket Ground",
}

def _norm_team(raw: str) -> str:
    if _is_placeholder_team(raw):
        return "TBC"
    return _TEAM_MAP.get(raw.strip().lower(), raw.strip())

def _is_placeholder_team(raw: str) -> bool:
    name = (raw or "").strip().lower()
    return (
        name in _PLACEHOLDER_TEAMS
        or "to be decided" in name
        or name.startswith(("winner of ", "loser of "))
    )

def _is_known_ipl_team(raw: str) -> bool:
    return _norm_team(raw) in _IPL_CODES

def _norm_venue(raw: str) -> str:
    low = raw.strip().lower()
    for k, v in _VENUE_MAP.items():
        if k in low:
            return v
    return raw.strip()

def _parse_date(raw: str) -> str:
    if not raw:
        return ""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d", "%d/%m/%Y", "%d %b %Y"):
        try:
            return datetime.strptime(raw[:len(fmt)], fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return raw[:10]

def _infer_status(m: dict) -> str:
    ended   = m.get("matchEnded",   False)
    started = m.get("matchStarted", False)
    status  = m.get("status", "").lower()
    if ended or "won" in status or "no result" in status or "draw" in status:
        return "completed"
    if started:
        return "live"
    return "upcoming"


def _fmt_score(s: dict) -> str:
    r, w, o = s.get("r", ""), s.get("w", ""), s.get("o", "")
    return f"{r}/{w} ({o})" if r != "" else ""


def _score_team(score: dict, teams: list[str]) -> str:
    label = " ".join(
        str(score.get(k, "") or "")
        for k in ("team", "inning", "innings", "name")
    ).lower()
    for team in teams:
        raw = (team or "").strip()
        norm = _norm_team(raw)
        if raw and raw.lower() in label:
            return norm
        if norm and norm.lower() in label.split():
            return norm
    return ""


def _repair_score_order(result: dict | None, home: str, away: str) -> bool:
    """
    Older cached scorecards stored score[0] as home_score and score[1] as
    away_score, but CricAPI score order is innings order. Use the result
    margin to swap those old rows when the card order and batting order differ.
    """
    if not isinstance(result, dict):
        return False
    if result.get("score_order") == "team":
        return False
    home_score = result.get("home_score")
    away_score = result.get("away_score")
    winner = result.get("winner", "")
    margin = (result.get("margin") or result.get("summary") or "").lower()
    if not home_score or not away_score or not winner:
        return False

    winner = _norm_team(winner)
    home = _norm_team(home)
    away = _norm_team(away)
    won_by_wickets = "wkt" in margin or "wicket" in margin
    won_by_runs = "run" in margin and not won_by_wickets

    should_swap = (winner == home and won_by_wickets) or (winner == away and won_by_runs)
    if should_swap:
        result["home_score"], result["away_score"] = away_score, home_score
        result["score_order"] = "team"
    return should_swap

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 + 2 — IPL series  (2 API calls total, cached 24h)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_ipl_series() -> list[dict]:
    """
    Returns IPL 2026 match list using a 3-strategy waterfall:
      1. /v1/series  paginated search for 'IPL 2026' → get series id → /v1/series_info
      2. /v1/currentMatches → look for live IPL match → extract series_id → /v1/series_info
      3. /v1/matches fallback → filter by name containing 'IPL' and date >= 2026
    Uses meta cache (24h) so this costs 0 calls on warm runs.
    """
    meta = _load_meta_cache()

    # Use cache if present and correctly versioned
    if meta.get("ipl_matches") and meta.get("ipl_matches_version") == IPL_MATCHES_CACHE_VERSION:
        log.info("IPL series loaded from meta cache (%d matches)", len(meta["ipl_matches"]))
        return meta["ipl_matches"]

    IPL_2026_START = "2026-01-01"
    series_id      = None

    # ── Strategy 1: search /v1/series ────────────────────────────────────────
    log.info("Strategy 1: searching /v1/series for IPL 2026…")
    for offset in SERIES_SEARCH_OFFSETS:
        data = _api("series", {"offset": offset})
        if not data:
            break
        for s in data.get("data", []):
            name = s.get("name", "").lower()
            sid  = s.get("id", "")
            # Match "ipl 2026" or "indian premier league 2026"
            if sid and (("ipl" in name or "indian premier" in name) and "2026" in name):
                series_id = sid
                log.info("Strategy 1 found: %s → %s", s["name"], sid)
                break
        if series_id:
            break
        if len(data.get("data", [])) < 25:
            break

    # ── Strategy 2: scan currentMatches for a live IPL game ──────────────────
    if not series_id:
        log.info("Strategy 2: scanning /v1/currentMatches for live IPL…")
        data = _api("currentMatches")
        if data:
            for m in data.get("data", []):
                mname = m.get("name", "").lower()
                if "ipl" in mname or "indian premier" in mname:
                    sid = m.get("series_id") or m.get("seriesId", "")
                    if sid:
                        series_id = sid
                        log.info("Strategy 2 found series_id=%s from live match: %s",
                                 sid, m.get("name"))
                        break

    # ── Try series_info with whatever ID we have ──────────────────────────────
    matches_raw = []
    if series_id:
        log.info("Fetching series_info for id=%s", series_id)
        data2 = _api("series_info", {"id": series_id})
        if data2:
            matches_raw = (data2.get("data") or {}).get("matchList", [])
            log.info("series_info returned %d matches", len(matches_raw))

    # ── Strategy 3: scan /v1/matches for IPL 2026 entries ────────────────────
    if not matches_raw:
        log.info("Strategy 3: scanning /v1/matches for IPL 2026 entries…")
        for offset in MATCH_SEARCH_OFFSETS:
            data = _api("matches", {"offset": offset})
            if not data:
                break
            for m in data.get("data", []):
                mname = m.get("name", "").lower()
                mdate = _parse_date(m.get("dateTimeGMT", m.get("date", "")))
                teams = m.get("teams", [])
                if len(teams) < 2:
                    continue
                # Must be IPL by name AND 2026
                if ("ipl" in mname or "indian premier" in mname) and mdate >= IPL_2026_START:
                    matches_raw.append(m)
            if len(data.get("data", [])) < 25:
                break
        log.info("Strategy 3 found %d raw IPL matches", len(matches_raw))

    # ── Build output list filtered to 2026 ───────────────────────────────────
    out = []
    skipped = 0
    for i, m in enumerate(matches_raw, start=1):
        teams = m.get("teams", [])
        if len(teams) < 2:
            continue
        mname = m.get("name", "").lower()
        has_placeholder = _is_placeholder_team(teams[0]) or _is_placeholder_team(teams[1])
        is_playoff_row = any(word in mname for word in _PLAYOFF_WORDS)
        if (
            (has_placeholder and not is_playoff_row)
            or (not has_placeholder and (
                not _is_known_ipl_team(teams[0])
                or not _is_known_ipl_team(teams[1])
            ))
        ):
            log.info("Skipping unresolved/non-IPL fixture: %s vs %s", teams[0], teams[1])
            continue
        mdate = _parse_date(m.get("dateTimeGMT", m.get("date", "")))
        if mdate < IPL_2026_START:
            skipped += 1
            continue
        out.append({
            "match":      i,
            "date":       mdate,
            "format":     "ipl",
            "home":       _norm_team(teams[0]),
            "away":       _norm_team(teams[1]),
            "venue":      _norm_venue(m.get("venue", "")),
            "cricapi_id": m.get("id", ""),
            "status":     _infer_status(m),
            "result":     None,
        })

    if skipped:
        log.info("Skipped %d pre-2026 matches", skipped)

    if out:
        meta["ipl_matches"]         = out
        meta["ipl_matches_version"] = IPL_MATCHES_CACHE_VERSION
        _save_meta_cache(meta)
        log.info("Cached %d IPL 2026 matches", len(out))
    else:
        log.warning("No IPL 2026 matches found via any strategy — using seed schedule")

    return out

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — T20I upcoming  (1 API call, cached in meta)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_t20i_upcoming() -> list[dict]:
    meta  = _load_meta_cache()
    today = date.today().isoformat()

    # IPL franchise codes — bust cache if it contains these
    _IPL_CODES = {"RCB","CSK","MI","KKR","SRH","RR","GT","PBKS","DC","LSG"}

    cached = meta.get("t20i_upcoming", [])
    cache_clean = all(
        m.get("home","") not in _IPL_CODES and m.get("away","") not in _IPL_CODES
        for m in cached
    )
    cached_date = meta.get("t20i_fetched_on", "")

    if cached and cached_date == today and cache_clean:
        log.info("T20I upcoming loaded from meta cache (%d)", len(cached))
        return cached

    if not cache_clean:
        log.warning("Busting T20I cache — contained IPL franchise teams")

    _FRANCHISE_KW = {
        "rcb","csk","mumbai indians","kolkata knight","sunrisers","rajasthan royals",
        "gujarat titans","punjab kings","delhi capitals","lucknow super","royal challengers",
    }

    def _is_franchise(name: str) -> bool:
        nl = name.strip().lower()
        return nl in {c.lower() for c in _IPL_CODES} or any(k in nl for k in _FRANCHISE_KW)

    log.info("Fetching upcoming T20I fixtures (international only)…")
    data = _api("matches", {"offset": 0})
    out  = []
    if data:
        for m in data.get("data", []):
            # ONLY accept matchType == "t20i" — never accept plain "t20" (catches IPL)
            if m.get("matchType", "").lower() != "t20i":
                continue
            mdate = _parse_date(m.get("dateTimeGMT", m.get("date", "")))
            if not mdate or mdate < today:
                continue
            teams = m.get("teams", [])
            if len(teams) < 2:
                continue
            if _is_franchise(teams[0]) or _is_franchise(teams[1]):
                continue
            mname = m.get("name", "").lower()
            if "ipl" in mname or "indian premier" in mname:
                continue
            out.append({
                "match":      0,
                "date":       mdate,
                "format":     "t20i",
                "home":       _norm_team(teams[0]),
                "away":       _norm_team(teams[1]),
                "venue":      _norm_venue(m.get("venue", "")),
                "cricapi_id": m.get("id", ""),
                "status":     "upcoming",
                "result":     None,
                "squads":     {},
            })

    meta["t20i_upcoming"]   = out
    meta["t20i_fetched_on"] = today
    _save_meta_cache(meta)
    log.info("Found %d upcoming international T20I fixtures", len(out))
    return out

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Scorecard enrichment  (at most `limit` new API calls per run)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_scorecard(cricapi_id: str, home: str = "", away: str = "") -> dict | None:
    """
    Fetches scorecard for one match.  Checks disk cache first — if found,
    returns immediately with zero API calls.
    """
    cached = _sc_cached(cricapi_id)
    if cached:
        log.info("Scorecard for %s loaded from disk cache", cricapi_id)
        if home and away and _repair_score_order(cached, home, away):
            _sc_save(cricapi_id, cached)
        return cached

    log.info("Fetching scorecard for %s…", cricapi_id)
    data = _api("match_scorecard", {"id": cricapi_id})
    if not data or not data.get("data"):
        log.warning("No scorecard data for %s", cricapi_id)
        return None

    d      = data["data"]
    status = d.get("status", "")

    result: dict = {
        "winner":        "",
        "margin":        "",
        "home_score":    "",
        "away_score":    "",
        "summary":       status,
        "player_scores": {},
    }

    # Winner + margin
    if "won" in status.lower():
        for t in (d.get("teams") or []):
            if t.lower() in status.lower():
                result["winner"] = _norm_team(t)
                break
        m = re.search(r"won by (.+?)(?:\s*\(|$)", status, re.I)
        if m:
            result["margin"] = m.group(1).strip()

    # Innings scores
    scores = d.get("score", [])
    if isinstance(scores, list):
        teams = d.get("teams") or []
        by_team = {}
        for s in scores:
            team = _score_team(s, teams)
            if team:
                by_team[team] = _fmt_score(s)

        home_code = _norm_team(home)
        away_code = _norm_team(away)
        result["home_score"] = by_team.get(home_code, "")
        result["away_score"] = by_team.get(away_code, "")
        if result["home_score"] and result["away_score"]:
            result["score_order"] = "team"

        if not result["home_score"] or not result["away_score"]:
            result["home_score"] = _fmt_score(scores[0]) if len(scores) > 0 else ""
            result["away_score"] = _fmt_score(scores[1]) if len(scores) > 1 else ""
            if not _repair_score_order(result, home, away):
                result["score_order"] = "innings"

    # Per-player actuals
    ps: dict = {}
    for innings in (d.get("scorecard") or []):
        for bat in (innings.get("batting") or []):
            name = (bat.get("batsman") or {}).get("name", "")
            if not name:
                continue
            try:
                runs  = int(bat.get("r", 0) or 0)
                balls = int(bat.get("b", 0) or 0)
            except (ValueError, TypeError):
                runs, balls = 0, 0
            ps[name] = {"runs": runs, "balls": balls, "role": "BAT"}

        for bowl in (innings.get("bowling") or []):
            name = (bowl.get("bowler") or {}).get("name", "")
            if not name:
                continue
            try:
                wkts = int(bowl.get("w", 0) or 0)
                rc   = int(bowl.get("r", 0) or 0)
            except (ValueError, TypeError):
                wkts, rc = 0, 0
            ps[name] = {"wickets": wkts, "runs_conceded": rc, "role": "BOWL"}

    result["player_scores"] = ps
    log.info("  → %d players, winner=%s", len(ps), result["winner"])

    # Persist to disk — never fetch this match again
    _sc_save(cricapi_id, result)
    return result


def enrich_completed(schedule: list[dict], limit: int = DEFAULT_SC_LIMIT) -> list[dict]:
    """
    Enriches completed matches with scorecard data.
    - Skips any match whose scorecard is already in the disk cache (free).
    - Makes at most `limit` NEW API calls per run.
    """
    today     = date.today().isoformat()
    new_calls = 0

    for match in schedule:
        if match.get("status") != "completed":
            continue

        cid = match.get("cricapi_id", "")
        if not cid or match.get("date", "9999") > today:
            continue

        if _repair_score_order(match.get("result"), match.get("home", ""), match.get("away", "")):
            log.info("M%d %s vs %s — repaired score order",
                     match["match"], match["home"], match["away"])

        # Already fully enriched in schedule.json
        if (match.get("result") or {}).get("player_scores"):
            continue

        # Check disk cache first — zero API calls
        cached = _sc_cached(cid)
        if cached:
            if _repair_score_order(cached, match.get("home", ""), match.get("away", "")):
                _sc_save(cid, cached)
            match["result"] = cached
            log.info("M%d %s vs %s — scorecard from cache",
                     match["match"], match["home"], match["away"])
            continue

        # Need a live API call — check budget
        if new_calls >= limit:
            log.info("Scorecard limit (%d) reached — remaining matches queued for next run", limit)
            break

        sc = fetch_scorecard(cid, match.get("home", ""), match.get("away", ""))   # _api call is inside, delay is enforced there
        if sc:
            match["result"] = sc
            new_calls += 1
            log.info("M%d %s vs %s — enriched (call %d/%d)",
                     match["match"], match["home"], match["away"], new_calls, limit)

    return schedule

# ─────────────────────────────────────────────────────────────────────────────
# Merge
# ─────────────────────────────────────────────────────────────────────────────

def _key(m: dict) -> tuple:
    return (m.get("date", ""), m.get("home", ""), m.get("away", ""))


def _has_result(m: dict) -> bool:
    result = m.get("result")
    return isinstance(result, dict) and bool(result)


def _best_status(current: str, incoming: str, match_date: str) -> str:
    """
    Pick the most accurate status between what we already have and what the
    API just returned.  Never allow a past-dated match to be marked 'upcoming'
    due to cricapi lag — date is the ground truth for completed vs upcoming.
    """
    today = date.today().isoformat()
    # If the date is in the past, it must be completed regardless of API response
    if match_date and match_date < today:
        return "completed"
    # live > completed > upcoming
    rank = {"live": 2, "completed": 1, "upcoming": 0}
    return max((current, incoming), key=lambda s: rank.get(s, 0))


def merge(seed: list[dict], fetched: list[dict]) -> list[dict]:
    idx    = {_key(m): i for i, m in enumerate(seed)}
    id_idx = {m.get("cricapi_id", ""): i for i, m in enumerate(seed) if m.get("cricapi_id")}
    merged = [dict(m) for m in seed]

    for fm in fetched:
        h, a = fm.get("home", ""), fm.get("away", "")
        has_placeholder = _is_placeholder_team(h) or _is_placeholder_team(a)
        if has_placeholder and fm.get("format") != "ipl":
            continue

        k = _key(fm)
        cid = fm.get("cricapi_id", "")
        i = idx.get(k)
        if i is None and cid:
            i = id_idx.get(cid)

        if i is not None:
            for field in ("date", "home", "away", "venue"):
                incoming = fm.get(field)
                current = merged[i].get(field, "")
                if incoming and (not current or _is_placeholder_team(current)):
                    merged[i][field] = incoming
            if fm.get("cricapi_id") and not merged[i].get("cricapi_id"):
                merged[i]["cricapi_id"] = fm["cricapi_id"]
            merged[i]["status"] = _best_status(
                merged[i].get("status", "upcoming"),
                fm.get("status", "upcoming"),
                merged[i].get("date", ""),
            )
            if fm.get("result") and not _has_result(merged[i]):
                merged[i]["result"] = fm["result"]
        else:
            if not h or not a or h == h.lower() or a == a.lower():
                continue
            fm = dict(fm, match=len(merged) + 1)
            # Apply date-based status fix to new entries too
            if fm.get("date", "9999") < date.today().isoformat():
                fm["status"] = "completed"
            merged.append(fm)
            idx[k] = len(merged) - 1
            if cid:
                id_idx[cid] = len(merged) - 1

    merged.sort(key=lambda m: m.get("date", "9999"))
    for i, m in enumerate(merged, start=1):
        m["match"] = i
    return merged

# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def sync_all(enrich: bool = True, scorecard_limit: int = DEFAULT_SC_LIMIT) -> dict:
    """
    Full sync — at most 3 + scorecard_limit API calls per warm run.
    Saves output/schedule.json and returns the schedule dict.
    """
    log.info("=== cricapi sync  (limit=%d scorecards) ===", scorecard_limit)
    today = date.today().isoformat()
    existing = _load_existing_schedule()

    # Phases 1+2: IPL series  (~2 calls, cached)
    fetched_ipl = fetch_ipl_series()
    seed_ipl = existing.get("ipl") or IPL_SCHEDULE
    merged_ipl  = merge(list(seed_ipl), fetched_ipl)

    # Guarantee: any match with a past date is 'completed', regardless of API lag
    for m in merged_ipl:
        if m.get("date", "9999") < today and m.get("status") != "live":
            m["status"] = "completed"

    # Phase 3: T20I upcoming  (1 call, cached daily)
    fetched_t20i = fetch_t20i_upcoming()
    seed_t20i_src = existing.get("t20i") or T20I_SCHEDULE
    seed_t20i    = [
        dict(m, status="completed" if m["date"] < today else m.get("status", "upcoming"))
        for m in seed_t20i_src
    ]
    merged_t20i  = merge(seed_t20i, fetched_t20i)

    # Phase 4: Scorecard enrichment  (≤ scorecard_limit NEW calls)
    if enrich:
        merged_ipl = enrich_completed(merged_ipl, limit=scorecard_limit)

    final = {"ipl": merged_ipl, "t20i": merged_t20i}

    with open(SCHEDULE_FILE, "w", encoding="utf-8") as fh:
        json.dump(final, fh, indent=2, default=str)

    def _c(lst):
        return (sum(1 for m in lst if m.get("status") == "completed"),
                sum(1 for m in lst if m.get("status") == "upcoming"))

    id_, iu = _c(merged_ipl)
    td, tu  = _c(merged_t20i)
    log.info("IPL  %d completed  %d upcoming", id_, iu)
    log.info("T20I %d completed  %d upcoming", td, tu)
    log.info("Saved → %s", SCHEDULE_FILE)
    return final


def load_schedule() -> dict:
    """Called by Flask on /api/schedule. Returns cached file or seed fallback."""
    if SCHEDULE_FILE.exists():
        age_h = (time.time() - SCHEDULE_FILE.stat().st_mtime) / 3600
        if age_h < SCHED_CACHE_TTL:
            with open(SCHEDULE_FILE, encoding="utf-8") as fh:
                return json.load(fh)

    today = date.today().isoformat()
    def _stamp(lst):
        return [dict(m, status="completed" if m["date"] < today else "upcoming")
                for m in lst]
    return {"ipl": _stamp(IPL_SCHEDULE), "t20i": _stamp(T20I_SCHEDULE)}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Sync cricket schedule via cricapi.com")
    ap.add_argument("--no-enrich",        action="store_true",
                    help="Skip all scorecard API calls")
    ap.add_argument("--scorecard-limit",  type=int, default=DEFAULT_SC_LIMIT,
                    help=f"Max NEW scorecard calls this run (default {DEFAULT_SC_LIMIT})")
    ap.add_argument("--dry-run",          action="store_true",
                    help="Print merged schedule without saving")
    ap.add_argument("--clear-meta-cache", action="store_true",
                    help="Delete meta_cache.json to force re-fetch of series id")
    args = ap.parse_args()

    if args.clear_meta_cache and META_CACHE_FILE.exists():
        META_CACHE_FILE.unlink()
        log.info("Meta cache cleared")

    result = sync_all(
        enrich=not args.no_enrich,
        scorecard_limit=args.scorecard_limit,
    )

    if args.dry_run:
        print(json.dumps(result, indent=2, default=str))
    else:
        ipl_d = sum(1 for m in result["ipl"]  if m.get("status") == "completed")
        ipl_u = sum(1 for m in result["ipl"]  if m.get("status") == "upcoming")
        t20_d = sum(1 for m in result["t20i"] if m.get("status") == "completed")
        t20_u = sum(1 for m in result["t20i"] if m.get("status") == "upcoming")
        sc_cached = sum(1 for f in SC_CACHE_DIR.glob("*.json"))
        print(f"\n✓ IPL   {ipl_d} completed  {ipl_u} upcoming")
        print(f"✓ T20I  {t20_d} completed  {t20_u} upcoming")
        print(f"✓ Scorecards on disk: {sc_cached}")
        print(f"✓ Saved → {SCHEDULE_FILE}")
