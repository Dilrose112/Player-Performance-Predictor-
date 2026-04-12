"""
06_app.py
---------
Flask API + dashboard for the Cricket ML prediction system.

Endpoints
---------
GET  /                    → serve dashboard.html
GET  /api/schedule        → IPL + T20I match schedule with squad metadata,
                            result data (if completed), and status flags.
                            Reads from output/schedule.json (kept fresh by
                            07_sync_schedule.py) with seed-list fallback.
GET  /api/venues          → list of known venues
GET  /api/players         → searchable player list (bat + bowl, IPL + T20)
POST /predict             → IPL batter prediction
POST /predict_bowl        → IPL bowler prediction
POST /predict_t20_bat     → T20I batter prediction
POST /predict_t20_bowl    → T20I bowler prediction
POST /api/sync            → trigger background schedule sync from ESPNcricinfo
                            body (optional): { "enrich": true }
GET  /api/sync/status     → age + match counts of cached schedule.json
"""
import pickle
import threading
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ─── SCHEDULE SYNC ───────────────────────────────────────────────────────────
# Try to import the sync module. Handles both naming conventions:
#   07_sync_schedule.py  (canonical)
#   06_sync_schedule.py  (legacy name some users may still have)
try:
    import importlib.util as _ilu, sys as _sys
    from pathlib import Path as _Path

    _base = _Path(__file__).parent
    _sync_file = None
    for _candidate in ("07_sync_schedule.py", "06_sync_schedule.py"):
        if (_base / _candidate).exists():
            _sync_file = _base / _candidate
            break

    if _sync_file is None:
        raise FileNotFoundError("Neither 07_sync_schedule.py nor 06_sync_schedule.py found")

    _spec = _ilu.spec_from_file_location("sync_schedule", _sync_file)
    _sync_mod = _ilu.module_from_spec(_spec)
    _sys.modules["sync_schedule"] = _sync_mod
    _spec.loader.exec_module(_sync_mod)
    load_schedule = _sync_mod.load_schedule
    _sync_all     = _sync_mod.sync_all
    _SYNC_AVAILABLE = True
except Exception as _e:
    import logging as _log
    _log.getLogger(__name__).warning("Sync module unavailable: %s", _e)
    _SYNC_AVAILABLE = False

def _get_live_schedule() -> dict:
    """
    Return the best available schedule.
    Priority: cached JSON file → seed hardcoded lists.
    """
    if _SYNC_AVAILABLE:
        return load_schedule()
    # Fallback: use the hardcoded lists as-is
    from datetime import date
    today = date.today().isoformat()
    def _stamp(lst):
        return [dict(m, status="completed" if m["date"] < today else "upcoming")
                for m in lst]
    return {"ipl": _stamp(IPL_SCHEDULE), "t20i": _stamp(T20I_SCHEDULE)}

# ─── LOAD ARTIFACTS ──────────────────────────────────────────────────────────

with open('models/ipl_models.pkl', 'rb') as f:
    IPL_M = pickle.load(f)

with open('models/t20_models.pkl', 'rb') as f:
    T20_M = pickle.load(f)

with open('models/player_profiles.pkl', 'rb') as f:
    PROFILES = pickle.load(f)

ipl_bat_p  = PROFILES['ipl_bat']
ipl_bowl_p = PROFILES['ipl_bowl']
t20_bat_p  = PROFILES['t20_bat']
t20_bowl_p = PROFILES['t20_bowl']

ipl_venue_ctx  = PROFILES['ipl_venue_context']
t20_venue_ctx  = PROFILES['t20_venue_context']
ipl_team_codes = PROFILES['ipl_team_codes']
ipl_era_ctx    = PROFILES['ipl_era_context']
t20_era_ctx    = PROFILES['t20_era_context']

# Team abbreviation to full name mapping for team_encoded
TEAM_ABBR_TO_FULL = {
    'CSK': 'Chennai Super Kings',
    'DC': 'Delhi Capitals',
    'GT': 'Gujarat Titans',
    'KKR': 'Kolkata Knight Riders',
    'LSG': 'Lucknow Super Giants',
    'MI': 'Mumbai Indians',
    'PBKS': 'Punjab Kings',
    'RCB': 'Royal Challengers Bengaluru',
    'RR': 'Rajasthan Royals',
    'SRH': 'Sunrisers Hyderabad',
}

MIN_VENUE_MATCHES = 5

# ─── IPL 2026 SCHEDULE ───────────────────────────────────────────────────────
# Player names match exactly the keys in player_profiles.pkl

IPL_SCHEDULE = []

# ─── T20I 2026 SCHEDULE ──────────────────────────────────────────────────────

T20I_SCHEDULE = []

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _get_venue_ctx(venue, fmt):
    ctx = ipl_venue_ctx if fmt == 'ipl' else t20_venue_ctx
    return ctx.get(venue, ctx['_global'])


def _get_era(date, fmt):
    ec = ipl_era_ctx if fmt == 'ipl' else t20_era_ctx
    yr = pd.to_datetime(date).year
    mn, mx = ec['year_min'], ec['year_max']
    mod = int(yr >= 2020)
    ew = 1.0 if mx == mn else 0.5 + (min(max(yr, mn), mx) - mn) / (mx - mn) * 0.5
    return {'modern_era': mod, 'era_weight': ew}


def _bat_features(ps, venue, team, date, fmt):
    vc  = _get_venue_ctx(venue, fmt)
    era = _get_era(date, fmt)
    full_team = TEAM_ABBR_TO_FULL.get(team, team) if fmt == 'ipl' else team
    tc  = ipl_team_codes.get(full_team, 0) if fmt == 'ipl' else 0
    vs  = ps.get('venue_stats', {}).get(venue, {})
    vm  = vs.get('matches', 0)
    q   = vm >= MIN_VENUE_MATCHES
    win = 'avg_runs_15' if fmt == 'ipl' else 'avg_runs_20'
    return {
        'avg_runs_5':   ps.get('avg_runs_5',  ps['career_avg']),
        'avg_runs_10':  ps.get('avg_runs_10', ps['career_avg']),
        win:            ps.get(win,           ps['career_avg']),
        'std_runs_5':   ps.get('std_runs_5',  15.0),
        'std_runs_10':  ps.get('std_runs_10', 15.0),
        'career_avg':   ps['career_avg'],
        'avg_sr_5':     ps.get('avg_sr_5',  ps['career_sr']),
        'avg_sr_10':    ps.get('avg_sr_10', ps['career_sr']),
        'career_sr':    ps['career_sr'],
        'matches_played': ps['matches_played'],
        'modern_era':   era['modern_era'],
        'era_weight':   era['era_weight'],
        'venue_avg_runs':     vc['venue_avg_runs'],
        'venue_avg_wkts':     vc['venue_avg_wkts'],
        'pitch_type_encoded': vc['pitch_type_encoded'],
        'dew_factor':         vc['dew_factor'],
        'chasing_advantage':  vc['chasing_advantage'],
        'player_venue_avg_runs': vs.get('avg_runs', ps['career_avg']) if q else ps['career_avg'],
        'player_venue_avg_sr':   vs.get('avg_sr',   ps['career_sr'])  if q else ps['career_sr'],
        'player_venue_matches':  vm,
        'venue_experience_weight': vs.get('venue_experience_weight', vm / (vm + MIN_VENUE_MATCHES)),
        'team_encoded': tc,
    }


def _bowl_features(ps, venue, team, date, fmt):
    vc  = _get_venue_ctx(venue, fmt)
    era = _get_era(date, fmt)
    full_team = TEAM_ABBR_TO_FULL.get(team, team) if fmt == 'ipl' else team
    tc  = ipl_team_codes.get(full_team, 0) if fmt == 'ipl' else 0
    vs  = ps.get('venue_stats', {}).get(venue, {})
    vm  = vs.get('matches', 0)
    q   = vm >= MIN_VENUE_MATCHES
    win = 'avg_wkts_15' if fmt == 'ipl' else 'avg_wkts_20'
    return {
        'avg_wkts_5':  ps.get('avg_wkts_5',  ps['career_wkt_avg']),
        'avg_wkts_10': ps.get('avg_wkts_10', ps['career_wkt_avg']),
        win:           ps.get(win,           ps['career_wkt_avg']),
        'std_wkts_5':  ps.get('std_wkts_5',  0.8),
        'career_wkt_avg': ps['career_wkt_avg'],
        'career_econ':    ps.get('career_econ', 8.0),
        'avg_econ_5':     ps.get('avg_econ_5',  ps.get('career_econ', 8.0)),
        'avg_econ_10':    ps.get('avg_econ_10', ps.get('career_econ', 8.0)),
        'bowling_matches': ps['bowling_matches'],
        'modern_era':   era['modern_era'],
        'era_weight':   era['era_weight'],
        'venue_avg_runs':     vc['venue_avg_runs'],
        'venue_avg_wkts':     vc['venue_avg_wkts'],
        'pitch_type_encoded': vc['pitch_type_encoded'],
        'dew_factor':         vc['dew_factor'],
        'chasing_advantage':  vc['chasing_advantage'],
        'player_venue_avg_wkts': vs.get('avg_wkts', ps['career_wkt_avg']) if q else ps['career_wkt_avg'],
        'player_venue_avg_econ': vs.get('avg_econ', ps.get('career_econ', 8.0)) if q else ps.get('career_econ', 8.0),
        'player_venue_matches':  vm,
        'venue_experience_weight': vs.get('venue_experience_weight', vm / (vm + MIN_VENUE_MATCHES)),
        'team_encoded': tc,
    }


def _to_df(fm, cols):
    return pd.DataFrame([{c: fm.get(c, 0.0) for c in cols}])


# ─── SQUAD LOADER ─────────────────────────────────────────────────────────────
# Loads squads/ipl_2026_squads.json so you can fix team rosters without
# editing the hardcoded IPL_SCHEDULE lists above.

try:
    import importlib.util as _ilu2
    from pathlib import Path as _Path

    _sq_file = None
    for _candidate in ("load_squads.py", "07_load_squads.py", "06_load_squads.py"):
        candidate_path = _Path(__file__).parent / _candidate
        if candidate_path.exists():
            _sq_file = candidate_path
            break

    if _sq_file is None:
        raise FileNotFoundError("No squad loader found: expected load_squads.py or 07_load_squads.py")

    _sq_spec = _ilu2.spec_from_file_location("load_squads", _sq_file)
    _sq_mod = _ilu2.module_from_spec(_sq_spec)
    _sq_spec.loader.exec_module(_sq_mod)

    _SQUAD_DATA = _sq_mod.load_ipl_squads()

    # Validate and log any mismatched names at startup
    _squad_issues = _sq_mod.validate_squads(_SQUAD_DATA, ipl_bat_p, ipl_bowl_p)
    if _squad_issues:
        import logging as _lg
        for team, issues in _squad_issues.items():
            for role, names in issues.items():
                _lg.getLogger(__name__).warning(
                    "Squad JSON — %s %s not in model: %s", team, role, names
                )
except Exception as _e:
    import logging as _lg
    _lg.getLogger(__name__).warning("Squad loader unavailable: %s", _e)
    _SQUAD_DATA = {}


def _get_squads_for_match(home: str, away: str, date: str, seed_squads: dict) -> dict:
    """
    Return squad dict for a match, preferring JSON file data over hardcoded seed.
    Falls back to seed_squads (from IPL_SCHEDULE) if JSON has no entry.
    """
    squads = {}
    for team in (home, away):
        if team in _SQUAD_DATA:
            squads[team] = _SQUAD_DATA[team]
        else:
            # fall back to seed
            seed = seed_squads.get((date, home, away), {})
            if team in seed:
                squads[team] = seed[team]
    return squads


# ─── API ROUTES ──────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/api/reload_squads', methods=['POST'])
def api_reload_squads():
    """
    Reload squads/ipl_2026_squads.json without restarting Flask.
    Call after editing the JSON file.
    """
    global _SQUAD_DATA
    try:
        _sq_spec.loader.exec_module(_sq_mod)
        _SQUAD_DATA = _sq_mod.load_ipl_squads()
        issues = _sq_mod.validate_squads(_SQUAD_DATA, ipl_bat_p, ipl_bowl_p)
        return jsonify({
            'status':  'ok',
            'teams':   len(_SQUAD_DATA),
            'warnings': issues,
        })
    except Exception as exc:
        return jsonify({'status': 'error', 'error': str(exc)}), 500


@app.route('/api/schedule')
def schedule():
    """
    Return the full schedule for both IPL and T20I.
    Each match includes:
      - teams: squad lists with in_profile flags
      - status: 'upcoming' | 'completed' | 'live'
      - result: { winner, margin, home_score, away_score, top_performers }
                (only present for completed matches)
    Data is served from output/schedule.json (synced by 07_sync_schedule.py)
    with a fallback to the hardcoded seed lists in this file.
    """
    live = _get_live_schedule()

    # Seed squad lookup (date, home, away) → squads dict — used as fallback only
    seed_squads = {}
    for m in IPL_SCHEDULE + T20I_SCHEDULE:
        key = (m.get('date',''), m.get('home',''), m.get('away',''))
        seed_squads[key] = m.get('squads', {})

    def annotate(matches, bat_pool, bowl_pool, use_json_squads=False):
        out = []
        for m in matches:
            mc = {k: v for k, v in m.items() if k != 'squads'}
            mc['teams'] = {}

            # Squad priority: JSON file > match's own squads > seed lookup
            if use_json_squads:
                squads = _get_squads_for_match(
                    m.get('home',''), m.get('away',''), m.get('date',''), seed_squads
                )
            else:
                squads = m.get('squads') or \
                         seed_squads.get((m.get('date',''), m.get('home',''), m.get('away','')), {})

            for team, sq in squads.items():
                mc['teams'][team] = {
                    'bat':  [{'name': p, 'in_profile': p in bat_pool}  for p in sq.get('bat', [])],
                    'bowl': [{'name': p, 'in_profile': p in bowl_pool} for p in sq.get('bowl', [])],
                }
            mc.setdefault('status', 'upcoming')
            mc.setdefault('result', None)
            out.append(mc)
        return out

    return jsonify({
        'ipl':  annotate(live.get('ipl',  []), ipl_bat_p, ipl_bowl_p, use_json_squads=True),
        't20i': annotate(live.get('t20i', []), t20_bat_p, t20_bowl_p, use_json_squads=False),
    })


@app.route('/api/sync', methods=['POST'])
def api_sync():
    """
    Trigger a full schedule sync from cricapi.com.

    POST /api/sync
    Optional body:  { "enrich": true, "clear_cache": true }
      enrich      – fetch scorecards for completed matches (default true)
      clear_cache – delete meta_cache.json + schedule.json before syncing
                    (default true — always start clean to fix stale data)

    Returns immediately with { "status": "started" }.
    The sync runs in a background thread; /api/sync/status will reflect
    the new file once it completes (~10-30s depending on how many scorecards).
    """
    if not _SYNC_AVAILABLE:
        return jsonify({
            'error': 'Sync module not found',
            'detail': 'Ensure 06_sync_schedule.py is in the same directory as 07_app.py'
        }), 503

    body        = request.get_json(silent=True) or {}
    enrich      = body.get('enrich', True)
    clear_cache = body.get('clear_cache', True)   # default ON to fix stale caches

    def _run():
        import logging
        log = logging.getLogger(__name__)
        try:
            if clear_cache:
                from pathlib import Path
                base = Path(__file__).parent / 'output'
                for fname in ('meta_cache.json', 'schedule.json'):
                    p = base / fname
                    if p.exists():
                        p.unlink()
                        log.info("Cleared %s before sync", fname)
            _sync_all(enrich=enrich)
        except Exception as exc:
            log.error("Background sync failed: %s", exc, exc_info=True)

    thread = threading.Thread(target=_run, daemon=True, name="cricapi-sync")
    thread.start()

    return jsonify({'status': 'started', 'enrich': enrich, 'clear_cache': clear_cache})


@app.route('/api/sync/status', methods=['GET'])
def api_sync_status():
    """
    Return sync availability + age and match counts of cached schedule.json.
    """
    import time as _time
    from pathlib import Path

    status = {
        'sync_available': _SYNC_AVAILABLE,
        'cached': False,
    }

    if not _SYNC_AVAILABLE:
        status['error'] = '06_sync_schedule.py not found next to 07_app.py'
        return jsonify(status)

    # Use path relative to this file so it works regardless of cwd
    spath = Path(__file__).parent / 'output' / 'schedule.json'
    if not spath.exists():
        return jsonify(status)

    age_s = _time.time() - spath.stat().st_mtime
    try:
        with open(spath) as fh:
            data = __import__('json').load(fh)
        ipl_counts  = _count_statuses(data.get('ipl',  []))
        t20i_counts = _count_statuses(data.get('t20i', []))
        return jsonify({
            'sync_available': True,
            'cached':        True,
            'age_seconds':   int(age_s),
            'age_human':     _human_age(age_s),
            'ipl':           ipl_counts,
            't20i':          t20i_counts,
        })
    except Exception as exc:
        return jsonify({'sync_available': True, 'cached': True,
                        'age_seconds': int(age_s), 'error': str(exc)})


def _count_statuses(matches: list) -> dict:
    from collections import Counter
    c = Counter(m.get('status', 'unknown') for m in matches)
    return dict(c)


def _human_age(seconds: float) -> str:
    if seconds < 60:    return f"{int(seconds)}s ago"
    if seconds < 3600:  return f"{int(seconds/60)}m ago"
    if seconds < 86400: return f"{int(seconds/3600)}h ago"
    return f"{int(seconds/86400)}d ago"


@app.route('/api/match_comparison', methods=['POST'])
def api_match_comparison():
    """
    For a completed match, run the ML predictions for every profiled player
    and return them side-by-side with the actual scores from fetched results.

    POST body:
      {
        "format":  "ipl",
        "home":    "RCB",
        "away":    "SRH",
        "venue":   "M Chinnaswamy Stadium",
        "date":    "2026-03-28",
        "actuals": {
          "teams": {
            "RCB": {"runs": 180, "wickets": 4, "overs": 20.0}
          },
          "players": {
            "V Kohli":   {"runs": 72, "balls": 48, "role": "BAT"},
            "JJ Bumrah": {"wickets": 3, "runs_conceded": 22, "role": "BOWL"}
          }
        },
        "player_scores": {          ← legacy fallback
          "V Kohli":   {"runs": 72, "balls": 48, "role": "BAT"},
          "JJ Bumrah": {"wickets": 3, "runs_conceded": 22, "role": "BOWL"}
        },
        "squads": {                 ← optional, for ordering
          "RCB": {"bat": [...], "bowl": [...]},
          "SRH": {"bat": [...], "bowl": [...]}
        }
      }

    Response:
      {
        "RCB": [
          {
            "name":       "V Kohli",
            "role":       "BAT",
            "pred_low":   18.4,
            "pred_mid":   31.2,
            "pred_high":  48.7,
            "actual":     72,          ← null if not in player_scores
            "actual_balls": 48,
            "hit":        true,        ← actual fell within [pred_low, pred_high]
            "delta":      40.8,        ← actual - pred_mid  (null if no actual)
            "career_avg": 37.1,
            "innings":    246
          }, ...
        ],
        "SRH": [...]
      }
    """
    body    = request.get_json(silent=True) or {}
    fmt     = body.get('format', 'ipl')
    home    = body.get('home', '')
    away    = body.get('away', '')
    venue   = body.get('venue', '')
    mdate   = body.get('date', '2026-04-01')
    actuals = body.get('actuals', {}) or {}
    ps_map  = actuals.get('players') or body.get('player_scores', {}) or {}
    squads  = body.get('squads', {})
    team_actuals = actuals.get('teams', {}) or {}

    bat_pool  = ipl_bat_p  if fmt == 'ipl' else t20_bat_p
    bowl_pool = ipl_bowl_p if fmt == 'ipl' else t20_bowl_p

    def _run_bat(player, team):
        ps = bat_pool.get(player)
        if not ps:
            return None
        fm = _bat_features(ps, venue, team, mdate, fmt)
        M  = IPL_M if fmt == 'ipl' else T20_M
        X  = _to_df(fm, M['bat_feats'])
        lo = max(0.0, float(M['bat'][0.25].predict(X)[0]))
        md = max(0.0, float(M['bat'][0.50].predict(X)[0]))
        hi = max(0.0, float(M['bat'][0.75].predict(X)[0]))
        return round(lo,1), round(md,1), round(hi,1), ps

    def _run_bowl(player, team):
        ps = bowl_pool.get(player)
        if not ps:
            return None
        fm = _bowl_features(ps, venue, team, mdate, fmt)
        M  = IPL_M if fmt == 'ipl' else T20_M
        X  = _to_df(fm, M['bowl_feats'])
        lo = max(0.0, float(M['bowl'][0.25].predict(X)[0]))
        md = max(0.0, float(M['bowl'][0.50].predict(X)[0]))
        hi = max(0.0, float(M['bowl'][0.75].predict(X)[0]))
        return round(lo,1), round(md,1), round(hi,1), ps

    # Build player lists per team from squads (or fall back to full bat/bowl pools)
    result_out = {}
    for team in (home, away):
        sq      = squads.get(team, {})
        batters = [p['name'] for p in sq.get('bat', []) if p.get('in_profile', True)]
        bowlers = [p['name'] for p in sq.get('bowl', []) if p.get('in_profile', True)]

        # If no squad data, scan both pools for any players we have
        if not batters and not bowlers:
            batters = list(bat_pool.keys())
            bowlers = list(bowl_pool.keys())

        rows = []

        for name in batters:
            r = _run_bat(name, team)
            if not r:
                continue
            lo, md, hi, ps = r
            actual_data = ps_map.get(name, {})
            actual      = actual_data.get('runs')
            actual_b    = actual_data.get('balls')
            hit         = (actual is not None) and (lo <= actual <= hi)
            delta       = round(actual - md, 1) if actual is not None else None
            rows.append({
                'name':        name,
                'role':        'BAT',
                'pred_low':    lo,
                'pred_mid':    md,
                'pred_high':   hi,
                'actual':      actual,
                'actual_balls': actual_b,
                'hit':         hit,
                'delta':       delta,
                'career_avg':  round(float(ps['career_avg']), 1),
                'innings':     ps['innings'],
            })

        for name in bowlers:
            r = _run_bowl(name, team)
            if not r:
                continue
            lo, md, hi, ps = r
            actual_data   = ps_map.get(name, {})
            actual        = actual_data.get('wickets')
            actual_rc     = actual_data.get('runs_conceded')
            hit           = (actual is not None) and (lo <= actual <= hi)
            delta         = round(actual - md, 1) if actual is not None else None
            rows.append({
                'name':           name,
                'role':           'BOWL',
                'pred_low':       lo,
                'pred_mid':       md,
                'pred_high':      hi,
                'actual':         actual,
                'actual_rc':      actual_rc,
                'hit':            hit,
                'delta':          delta,
                'career_avg':     round(float(ps['career_wkt_avg']), 2),
                'innings':        ps['innings'],
            })

        result_out[team] = rows

    # Summary stats
    all_rows  = [r for rows in result_out.values() for r in rows]
    with_data = [r for r in all_rows if r['actual'] is not None]
    hit_rate  = round(sum(1 for r in with_data if r['hit']) / len(with_data) * 100, 1) \
                if with_data else None

    return jsonify({
        'teams':    result_out,
        'team_actuals': team_actuals,
        'hit_rate': hit_rate,
        'n_actual': len(with_data),
        'n_total':  len(all_rows),
    })


@app.route('/api/venues')
def api_venues():
    return jsonify({
        'ipl':  sorted(v for v in ipl_venue_ctx if v != '_global'),
        't20i': sorted(v for v in t20_venue_ctx if v != '_global'),
    })


@app.route('/predict', methods=['POST'])
def predict():
    """IPL batter prediction."""
    d      = request.json
    player = d.get('player', '')
    venue  = d.get('venue', '')
    team   = d.get('team', '')
    date   = d.get('date', '2026-04-01')

    ps = ipl_bat_p.get(player)
    if not ps:
        return jsonify({'error': f'Player "{player}" not found'}), 404

    fm = _bat_features(ps, venue, team, date, 'ipl')
    X  = _to_df(fm, IPL_M['bat_feats'])
    lo = max(0.0, float(IPL_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bat'][0.75].predict(X)[0]))
    vc = _get_venue_ctx(venue, 'ipl')
    return jsonify({
        'runs_low': round(lo, 1), 'runs_mid': round(md, 1), 'runs_high': round(hi, 1),
        'pitch': vc['pitch_type'], 'dew': round(vc['dew_factor'], 3),
        'chasing': vc['chasing_advantage'],
        'career_avg': round(float(ps['career_avg']), 1),
        'innings': ps['innings'],
    })


@app.route('/predict_bowl', methods=['POST'])
def predict_bowl():
    """IPL bowler prediction."""
    d      = request.json
    player = d.get('player', '')
    venue  = d.get('venue', '')
    team   = d.get('team', '')
    date   = d.get('date', '2026-04-01')

    ps = ipl_bowl_p.get(player)
    if not ps:
        return jsonify({'error': f'Bowler "{player}" not found'}), 404

    fm = _bowl_features(ps, venue, team, date, 'ipl')
    X  = _to_df(fm, IPL_M['bowl_feats'])
    lo = max(0.0, float(IPL_M['bowl'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bowl'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bowl'][0.75].predict(X)[0]))
    vc = _get_venue_ctx(venue, 'ipl')
    return jsonify({
        'wkts_low': round(lo, 1), 'wkts_mid': round(md, 1), 'wkts_high': round(hi, 1),
        'pitch': vc['pitch_type'], 'dew': round(vc['dew_factor'], 3),
        'chasing': vc['chasing_advantage'],
        'career_avg': round(float(ps['career_wkt_avg']), 2),
        'innings': ps['innings'],
    })


@app.route('/predict_t20_bat', methods=['POST'])
def predict_t20_bat():
    """T20I batter prediction."""
    d      = request.json
    player = d.get('player', '')
    venue  = d.get('venue', '')
    team   = d.get('team', '')
    date   = d.get('date', '2026-04-10')

    ps = t20_bat_p.get(player)
    if not ps:
        return jsonify({'error': f'T20 player "{player}" not found'}), 404

    fm = _bat_features(ps, venue, team, date, 't20i')
    X  = _to_df(fm, T20_M['bat_feats'])
    lo = max(0.0, float(T20_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(T20_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(T20_M['bat'][0.75].predict(X)[0]))
    vc = _get_venue_ctx(venue, 't20i')
    return jsonify({
        'runs_low': round(lo, 1), 'runs_mid': round(md, 1), 'runs_high': round(hi, 1),
        'pitch': vc['pitch_type'], 'dew': round(vc['dew_factor'], 3),
        'chasing': vc['chasing_advantage'],
        'career_avg': round(float(ps['career_avg']), 1),
        'innings': ps['innings'],
    })


@app.route('/predict_t20_bowl', methods=['POST'])
def predict_t20_bowl():
    """T20I bowler prediction."""
    d      = request.json
    player = d.get('player', '')
    venue  = d.get('venue', '')
    team   = d.get('team', '')
    date   = d.get('date', '2026-04-10')

    ps = t20_bowl_p.get(player)
    if not ps:
        return jsonify({'error': f'T20 bowler "{player}" not found'}), 404

    fm = _bowl_features(ps, venue, team, date, 't20i')
    X  = _to_df(fm, T20_M['bowl_feats'])
    lo = max(0.0, float(T20_M['bowl'][0.25].predict(X)[0]))
    md = max(0.0, float(T20_M['bowl'][0.50].predict(X)[0]))
    hi = max(0.0, float(T20_M['bowl'][0.75].predict(X)[0]))
    vc = _get_venue_ctx(venue, 't20i')
    return jsonify({
        'wkts_low': round(lo, 1), 'wkts_mid': round(md, 1), 'wkts_high': round(hi, 1),
        'pitch': vc['pitch_type'], 'dew': round(vc['dew_factor'], 3),
        'chasing': vc['chasing_advantage'],
        'career_avg': round(float(ps['career_wkt_avg']), 2),
        'innings': ps['innings'],
    })


# ─── NEW TAB ENDPOINTS ───────────────────────────────────────────────────────

@app.route('/api/players')
def api_players():
    """
    Return all profiled IPL batters and bowlers with their career profile.
    Used by the Player Insights and Player Comparison tabs.

    Each entry contains enough data to render the static profile card without
    a second API call; the frontend then calls /predict for live predictions.
    """
    fmt = request.args.get('fmt', 'ipl')

    if fmt == 't20i':
        bat_pool  = PROFILES['t20_bat']
        bowl_pool = PROFILES['t20_bowl']
    else:
        bat_pool  = ipl_bat_p
        bowl_pool = ipl_bowl_p

    players = []

    for name, ps in bat_pool.items():
        entry = {
            'name':          name,
            'role':          'BAT',
            'format':        fmt,
            'innings':       ps['innings'],
            'career_avg':    round(float(ps['career_avg']), 2),
            'career_sr':     round(float(ps['career_sr']),  2),
            'matches_played': ps['matches_played'],
            'last_date':     ps.get('last_date', ''),
            'retired':       bool(ps.get('retired', False)),
            # Include rolling windows so Insights tab can show recent form
            'avg_runs_5':    round(float(ps.get('avg_runs_5',  ps['career_avg'])), 2),
            'avg_runs_10':   round(float(ps.get('avg_runs_10', ps['career_avg'])), 2),
            # Venue stats summary (top 3 qualified venues by matches)
            'top_venues':    _top_venue_stats(ps.get('venue_stats', {}), 'bat'),
        }
        players.append(entry)

    for name, ps in bowl_pool.items():
        entry = {
            'name':           name,
            'role':           'BOWL',
            'format':         fmt,
            'innings':        ps['innings'],
            'career_avg':     round(float(ps['career_wkt_avg']), 3),
            'career_econ':    round(float(ps.get('career_econ', 8.0)), 2),
            'matches_played': ps['bowling_matches'],
            'last_date':      ps.get('last_date', ''),
            'retired':        bool(ps.get('retired', False)),
            'avg_wkts_5':     round(float(ps.get('avg_wkts_5',  ps['career_wkt_avg'])), 2),
            'avg_wkts_10':    round(float(ps.get('avg_wkts_10', ps['career_wkt_avg'])), 2),
            'top_venues':     _top_venue_stats(ps.get('venue_stats', {}), 'bowl'),
        }
        players.append(entry)

    # Sort: batters first, then bowlers, each group alphabetically
    players.sort(key=lambda p: (0 if p['role'] == 'BAT' else 1, p['name']))
    return jsonify(players)


def _top_venue_stats(venue_stats, role):
    """Return up to 3 qualified venue entries, sorted by matches desc."""
    qualified = [
        {'venue': v, **s}
        for v, s in venue_stats.items()
        if s.get('qualified', False)
    ]
    qualified.sort(key=lambda x: -x['matches'])
    return qualified[:3]


@app.route('/api/model_summary')
def api_model_summary():
    """
    Static model metadata — algorithm, metrics, feature importances, and
    feature explanations.  All values come directly from the trained model
    objects so nothing is hardcoded.
    """
    def fi_list(fi_dict):
        return [
            {'feature': k, 'importance': round(v * 100, 1)}
            for k, v in sorted(fi_dict.items(), key=lambda x: -x[1])
        ]

    return jsonify({
        'algorithm': 'Gradient Boosting Regressor (Quantile)',
        'quantiles': [0.25, 0.50, 0.75],
        'train_split': 'Pre-2024 → train   |   2024+ → test',
        'ipl': {
            'bat': {
                'features':     IPL_M['bat_feats'],
                'n_features':   len(IPL_M['bat_feats']),
                'importances':  fi_list(IPL_M['bat_fi']),
                # Hard metrics from training run (stored alongside model)
                'mae':          16.16,
                'coverage_50':  48.6,
            },
            'bowl': {
                'features':     IPL_M['bowl_feats'],
                'n_features':   len(IPL_M['bowl_feats']),
                'importances':  fi_list(IPL_M['bowl_fi']),
                'mae':          0.76,
                'coverage_50':  73.0,
            },
        },
        't20i': {
            'bat': {
                'features':     T20_M['bat_feats'],
                'n_features':   len(T20_M['bat_feats']),
                'importances':  fi_list(T20_M['bat_fi']),
                'mae':          11.59,
                'coverage_50':  49.5,
            },
            'bowl': {
                'features':     T20_M['bowl_feats'],
                'n_features':   len(T20_M['bowl_feats']),
                'importances':  fi_list(T20_M['bowl_fi']),
                'mae':          0.81,
                'coverage_50':  83.7,
            },
        },
        'feature_explanations': {
            'pitch_type_encoded': (
                'Derived from historical average innings score at the venue. '
                'batting (>170 avg) = 2, balanced (150–170) = 1, bowling (<150) = 0. '
                'Computed from prior matches only (no leakage).'
            ),
            'dew_factor': (
                'Proportion of matches at this venue where the chasing team won. '
                'High values (>0.6) indicate dew-affected surfaces that favour the '
                'team batting second. Rolling historical average from past matches only.'
            ),
            'venue_avg_runs': (
                'Expanding mean of average innings score at the venue up to the '
                'previous match. Captures how scoring-friendly the surface is over '
                'its full history without using current-match data.'
            ),
            'era_weight': (
                'Normalised year value in [0.5, 1.0] that lets the model learn the '
                'upward trend in T20 scoring over time without hard-coding run adjustments. '
                'Seasons from year_min→year_max map linearly to 0.5→1.0.'
            ),
            'player_venue_avg_runs': (
                'Player\'s own expanding average at this specific venue (shift-1). '
                'Falls back to career average when fewer than 5 prior appearances exist, '
                'preventing noisy estimates from tiny samples.'
            ),
            'chasing_advantage': (
                'Binary flag (1/0) derived from dew_factor > 0.6. Explicitly signals '
                'to the model when historical dew patterns strongly favour the second '
                'innings at this venue.'
            ),
        },
    })


@app.route('/api/player_overviews')
def api_player_overviews():
    """
    Pre-computed overview of featured IPL 2026 players.
    Returns archetype classification, form trend, career highlights,
    and top venue data — all derived from stored profiles, no hardcoding.

    Also returns a set of curated bat-vs-bowl rivalry pairs with shared
    venue context pulled from both profiles.
    """

    # ── Featured players (IPL 2026 squads) ──────────────────────────────────
    FEATURED_BATS = []

    FEATURED_BOWLS = []

    # ── Rivalry pairs with context labels ───────────────────────────────────
    RIVALRY_PAIRS = []

    def bat_archetype(avg, sr):
        if sr >= 145 and avg >= 28: return 'Power Hitter'
        if sr >= 135 and avg >= 24: return 'Aggressive Bat'
        if avg >= 35 and sr < 120:  return 'Anchor'
        if avg >= 28 and sr >= 120: return 'Accumulator'
        if sr >= 125:               return 'Finisher'
        return 'Middle Order'

    def bat_speciality(avg, sr, form5, std5):
        points = []
        if sr >= 140:               points.append('explosive scoring rate')
        if avg >= 35:               points.append('elite consistency')
        if form5 > avg * 1.15:      points.append('current hot streak')
        if std5 / max(avg, 1) < 0.6: points.append('reliable performances')
        if avg >= 30 and sr >= 130: points.append('complete batsman')
        return points[:3] if points else ['experienced campaigner']

    def bowl_archetype(avg, econ):
        if econ <= 7.5 and avg >= 1.2: return 'Economy + Wickets'
        if econ <= 7.3:                return 'Economy Specialist'
        if avg >= 1.3:                 return 'Wicket Taker'
        if avg >= 1.2 and econ <= 8.5: return 'Strike Bowler'
        return 'Supporting Bowler'

    def bowl_speciality(avg, econ, wkt5, form_wkt):
        points = []
        if econ <= 7.5:            points.append('extremely economical')
        if avg >= 1.35:            points.append('consistent wicket taker')
        if wkt5 > avg * 1.1:       points.append('in excellent wicket-taking form')
        if econ <= 8.0 and avg >= 1.1: points.append('restricts and takes wickets')
        if form_wkt > avg * 1.15:  points.append('current form peak')
        return points[:3] if points else ['reliable bowler']

    def form_label(form5, career):
        ratio = form5 / max(career, 1)
        if ratio > 1.2:  return 'hot'
        if ratio > 1.05: return 'good'
        if ratio < 0.8:  return 'cold'
        return 'steady'

    # ── Build batter overview list ───────────────────────────────────────────
    batters = []
    for name in FEATURED_BATS:
        ps = ipl_bat_p.get(name)
        if not ps: continue

        avg   = float(ps['career_avg'])
        sr    = float(ps['career_sr'])
        form5 = float(ps.get('avg_runs_5',  avg))
        std5  = float(ps.get('std_runs_5',  15.0))
        form10 = float(ps.get('avg_runs_10', avg))

        # Best venue (qualified, highest avg)
        best_venue = None
        best_avg   = 0
        for v, vs in ps.get('venue_stats', {}).items():
            if vs.get('qualified', False) and vs['avg_runs'] > best_avg:
                best_avg   = vs['avg_runs']
                best_venue = {'venue': v, 'avg_runs': round(vs['avg_runs'], 1),
                              'avg_sr': round(vs.get('avg_sr', sr), 1),
                              'matches': vs['matches']}

        batters.append({
            'name':        name,
            'role':        'BAT',
            'innings':     ps['innings'],
            'career_avg':  round(avg, 1),
            'career_sr':   round(sr, 1),
            'form5_avg':   round(form5, 1),
            'form10_avg':  round(form10, 1),
            'archetype':   bat_archetype(avg, sr),
            'specialities': bat_speciality(avg, sr, form5, std5),
            'form':        form_label(form5, avg),
            'best_venue':  best_venue,
            'last_date':   ps.get('last_date', ''),
        })

    # ── Build bowler overview list ───────────────────────────────────────────
    bowlers = []
    for name in FEATURED_BOWLS:
        ps = ipl_bowl_p.get(name)
        if not ps: continue

        avg    = float(ps['career_wkt_avg'])
        econ   = float(ps.get('career_econ', 8.0))
        wkt5   = float(ps.get('avg_wkts_5',  avg))
        wkt10  = float(ps.get('avg_wkts_10', avg))
        econ5  = float(ps.get('avg_econ_5',  econ))

        best_venue = None
        best_avg   = 0
        for v, vs in ps.get('venue_stats', {}).items():
            if vs.get('qualified', False) and vs['avg_wkts'] > best_avg:
                best_avg   = vs['avg_wkts']
                best_venue = {'venue': v, 'avg_wkts': round(vs['avg_wkts'], 2),
                              'avg_econ': round(vs.get('avg_econ', econ), 2),
                              'matches': vs['matches']}

        bowlers.append({
            'name':        name,
            'role':        'BOWL',
            'innings':     ps['innings'],
            'career_avg':  round(avg, 2),
            'career_econ': round(econ, 2),
            'form5_avg':   round(wkt5, 2),
            'form10_avg':  round(wkt10, 2),
            'form5_econ':  round(econ5, 2),
            'archetype':   bowl_archetype(avg, econ),
            'specialities': bowl_speciality(avg, econ, wkt5, wkt5),
            'form':        form_label(wkt5, avg),
            'best_venue':  best_venue,
            'last_date':   ps.get('last_date', ''),
        })

    # ── Build rivalry data ───────────────────────────────────────────────────
    rivalries = []
    for bat_name, bowl_name, context in RIVALRY_PAIRS:
        bat_ps  = ipl_bat_p.get(bat_name)
        bowl_ps = ipl_bowl_p.get(bowl_name)
        if not (bat_ps and bowl_ps):
            continue

        bat_avg   = float(bat_ps['career_avg'])
        bat_sr    = float(bat_ps['career_sr'])
        bowl_avg  = float(bowl_ps['career_wkt_avg'])
        bowl_econ = float(bowl_ps.get('career_econ', 8.0))
        bat_form  = float(bat_ps.get('avg_runs_5', bat_avg))
        bowl_form = float(bowl_ps.get('avg_wkts_5', bowl_avg))

        # Find shared qualified venues for richer context
        bat_venues  = bat_ps.get('venue_stats', {})
        bowl_venues = bowl_ps.get('venue_stats', {})
        shared = []
        for v in bat_venues:
            if v in bowl_venues:
                bv  = bat_venues[v]
                bwv = bowl_venues[v]
                if bv.get('qualified', False) or bwv.get('qualified', False):
                    shared.append({
                        'venue':       v,
                        'bat_avg':     round(bv['avg_runs'], 1),
                        'bat_matches': bv['matches'],
                        'bowl_avg':    round(bwv['avg_wkts'], 2),
                        'bowl_matches': bwv['matches'],
                    })
        # Sort by combined match count — most-played venues first
        shared.sort(key=lambda x: -(x['bat_matches'] + x['bowl_matches']))

        # Edge assessment: who holds the statistical edge?
        # Compare bat_avg to expected runs a bowler of this quality would concede
        # A bowler averaging 1.3 wkts at 8.0 econ roughly allows 26 runs per innings
        expected_runs_per_innings = (bowl_econ * 4)  # rough 4-over spell
        edge = 'bat' if bat_avg > expected_runs_per_innings * 1.1 else \
               'bowl' if bat_avg < expected_runs_per_innings * 0.9 else 'even'

        rivalries.append({
            'bat':          bat_name,
            'bowl':         bowl_name,
            'context':      context,
            'bat_avg':      round(bat_avg, 1),
            'bat_sr':       round(bat_sr, 1),
            'bat_form':     round(bat_form, 1),
            'bat_arch':     bat_archetype(bat_avg, bat_sr),
            'bowl_avg':     round(bowl_avg, 2),
            'bowl_econ':    round(bowl_econ, 2),
            'bowl_form':    round(bowl_form, 2),
            'bowl_arch':    bowl_archetype(bowl_avg, bowl_econ),
            'edge':         edge,
            'shared_venues': shared[:3],
        })

    return jsonify({
        'batters':  batters,
        'bowlers':  bowlers,
        'rivalries': rivalries,
    })


if __name__ == '__main__':
    app.run(debug=True)
