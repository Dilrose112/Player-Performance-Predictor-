"""
06_app.py
---------
Flask API + dashboard for the Cricket ML prediction system.

Endpoints
---------
GET  /                    → serve dashboard.html
GET  /api/schedule        → IPL + T20I match schedule with squad metadata
GET  /api/venues          → list of known venues
GET  /api/players         → searchable player list (bat + bowl, IPL + T20)
POST /predict             → IPL batter prediction
POST /predict_bowl        → IPL bowler prediction
POST /predict_t20_bat     → T20I batter prediction
POST /predict_t20_bowl    → T20I bowler prediction
"""
import pickle
import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

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

MIN_VENUE_MATCHES = 5

# ─── IPL 2026 SCHEDULE ───────────────────────────────────────────────────────
# Player names match exactly the keys in player_profiles.pkl

IPL_SCHEDULE = [
    {
        'match': 1, 'date': '2026-03-28', 'format': 'ipl',
        'home': 'RCB', 'away': 'SRH',
        'venue': 'M Chinnaswamy Stadium',
        'squads': {
            'RCB':  {
                'bat':  ['V Kohli', 'PD Salt', 'RM Patidar', 'VR Iyer', 'TH David'],
                'bowl': ['JR Hazlewood', 'Yash Dayal', 'Suyash Sharma'],
            },
            'SRH': {
                'bat':  ['TM Head', 'Abhishek Sharma', 'H Klaasen', 'Nithish Kumar Reddy', 'Ishan Kishan'],
                'bowl': ['PJ Cummins', 'Mohammed Shami', 'HV Patel', 'JD Unadkat'],
            },
        },
    },
    {
        'match': 2, 'date': '2026-03-29', 'format': 'ipl',
        'home': 'MI', 'away': 'KKR',
        'venue': 'Wankhede Stadium',
        'squads': {
            'MI':  {
                'bat':  ['RG Sharma', 'SA Yadav', 'Tilak Varma', 'HH Pandya'],
                'bowl': ['JJ Bumrah', 'TA Boult'],
            },
            'KKR': {
                'bat':  ['AM Rahane', 'VR Iyer', 'R Parag'],
                'bowl': ['AD Russell', 'SP Narine', 'Harshit Rana', 'M Pathirana'],
            },
        },
    },
    {
        'match': 3, 'date': '2026-03-30', 'format': 'ipl',
        'home': 'RR', 'away': 'CSK',
        'venue': 'Sawai Mansingh Stadium',
        'squads': {
            'RR':  {
                'bat':  ['YBK Jaiswal', 'R Parag', 'Shimron Hetmyer', 'Dhruv Jurel'],
                'bowl': ['M Theekshana', 'Sandeep Sharma', 'Wanindu Hasaranga'],
            },
            'CSK': {
                'bat':  ['SV Samson', 'MS Dhoni', 'S Dube', 'D Brevis'],
                'bowl': ['MJ Henry', 'Noor Ahmad', 'Khaleel Ahmed', 'R Chahar'],
            },
        },
    },
    {
        'match': 4, 'date': '2026-03-31', 'format': 'ipl',
        'home': 'PBKS', 'away': 'GT',
        'venue': 'Punjab Cricket Association Stadium',
        'squads': {
            'PBKS': {
                'bat':  ['Priyansh Arya', 'MP Stoinis', 'SS Iyer'],
                'bowl': ['Arshdeep Singh', 'YS Chahal', 'Harpreet Brar'],
            },
            'GT':   {
                'bat':  ['Shubman Gill', 'JC Buttler', 'B Sai Sudharsan'],
                'bowl': ['Rashid Khan', 'KA Rabada', 'Mohammed Siraj'],
            },
        },
    },
    {
        'match': 5, 'date': '2026-04-01', 'format': 'ipl',
        'home': 'LSG', 'away': 'DC',
        'venue': 'Bharat Ratna Shri Atal Bihari Vajpayee Ekana Cricket Stadium, Lucknow',
        'squads': {
            'LSG': {
                'bat':  ['RR Pant', 'AK Markram', 'N Pooran', 'MR Marsh'],
                'bowl': ['Mohammed Shami', 'Mayank Yadav', 'Avesh Khan', 'Ravi Bishnoi'],
            },
            'DC':  {
                'bat':  ['KL Rahul', 'J Fraser-McGurk', 'T Stubbs'],
                'bowl': ['Kuldeep Yadav', 'MA Starc', 'Mukesh Kumar'],
            },
        },
    },
    {
        'match': 6, 'date': '2026-04-03', 'format': 'ipl',
        'home': 'SRH', 'away': 'KKR',
        'venue': 'Rajiv Gandhi International Stadium',
        'squads': {
            'SRH': {
                'bat':  ['TM Head', 'Abhishek Sharma', 'H Klaasen', 'Ishan Kishan'],
                'bowl': ['PJ Cummins', 'Mohammed Shami', 'HV Patel'],
            },
            'KKR': {
                'bat':  ['AM Rahane', 'VR Iyer', 'R Parag'],
                'bowl': ['AD Russell', 'SP Narine', 'Harshit Rana'],
            },
        },
    },
    {
        'match': 7, 'date': '2026-04-05', 'format': 'ipl',
        'home': 'MI', 'away': 'RCB',
        'venue': 'Wankhede Stadium',
        'squads': {
            'MI':  {
                'bat':  ['RG Sharma', 'SA Yadav', 'Tilak Varma', 'HH Pandya'],
                'bowl': ['JJ Bumrah', 'TA Boult'],
            },
            'RCB': {
                'bat':  ['V Kohli', 'PD Salt', 'RM Patidar'],
                'bowl': ['JR Hazlewood', 'Yash Dayal', 'Suyash Sharma'],
            },
        },
    },
    {
        'match': 8, 'date': '2026-04-06', 'format': 'ipl',
        'home': 'RR', 'away': 'GT',
        'venue': 'Sawai Mansingh Stadium',
        'squads': {
            'RR': {
                'bat':  ['YBK Jaiswal', 'R Parag', 'Shimron Hetmyer'],
                'bowl': ['M Theekshana', 'Wanindu Hasaranga', 'Sandeep Sharma'],
            },
            'GT': {
                'bat':  ['Shubman Gill', 'JC Buttler', 'B Sai Sudharsan'],
                'bowl': ['Rashid Khan', 'KA Rabada', 'Mohammed Siraj'],
            },
        },
    },
]

# ─── T20I 2026 SCHEDULE ──────────────────────────────────────────────────────

T20I_SCHEDULE = [
    {
        'match': 1, 'date': '2026-04-10', 'format': 't20i',
        'home': 'India', 'away': 'Australia',
        'venue': 'Wankhede Stadium',
        'squads': {
            'India': {
                'bat':  ['V Kohli', 'RG Sharma', 'KL Rahul', 'Shubman Gill', 'YBK Jaiswal', 'RR Pant', 'Tilak Varma'],
                'bowl': ['JJ Bumrah', 'Mohammed Shami', 'YS Chahal', 'Arshdeep Singh', 'Kuldeep Yadav'],
            },
            'Australia': {
                'bat':  ['DA Warner', 'TM Head', 'MS Wade', 'GJ Maxwell', 'MP Stoinis', 'MR Marsh'],
                'bowl': ['PJ Cummins', 'A Zampa', 'JR Hazlewood', 'NT Ellis'],
            },
        },
    },
    {
        'match': 2, 'date': '2026-04-13', 'format': 't20i',
        'home': 'India', 'away': 'England',
        'venue': 'Eden Gardens',
        'squads': {
            'India': {
                'bat':  ['V Kohli', 'RG Sharma', 'Shubman Gill', 'YBK Jaiswal', 'RR Pant', 'Abhishek Sharma'],
                'bowl': ['JJ Bumrah', 'Arshdeep Singh', 'YS Chahal', 'Kuldeep Yadav'],
            },
            'England': {
                'bat':  ['JC Buttler', 'PD Salt', 'DJ Malan', 'JM Bairstow', 'MM Ali'],
                'bowl': ['MA Wood', 'CR Woakes', 'AU Rashid', 'JC Archer'],
            },
        },
    },
    {
        'match': 3, 'date': '2026-04-16', 'format': 't20i',
        'home': 'India', 'away': 'Pakistan',
        'venue': 'Narendra Modi Stadium',
        'squads': {
            'India': {
                'bat':  ['V Kohli', 'RG Sharma', 'KL Rahul', 'Shubman Gill', 'YBK Jaiswal', 'RR Pant'],
                'bowl': ['JJ Bumrah', 'Mohammed Shami', 'YS Chahal', 'Arshdeep Singh', 'Kuldeep Yadav'],
            },
            'Pakistan': {
                'bat':  ['Babar Azam', 'Mohammad Rizwan', 'Fakhar Zaman'],
                'bowl': ['Shaheen Shah Afridi', 'Haris Rauf', 'Shadab Khan', 'Imad Wasim'],
            },
        },
    },
    {
        'match': 4, 'date': '2026-04-20', 'format': 't20i',
        'home': 'Australia', 'away': 'England',
        'venue': 'Melbourne Cricket Ground',
        'squads': {
            'Australia': {
                'bat':  ['DA Warner', 'TM Head', 'GJ Maxwell', 'MP Stoinis', 'MR Marsh'],
                'bowl': ['PJ Cummins', 'A Zampa', 'JR Hazlewood', 'NT Ellis'],
            },
            'England': {
                'bat':  ['JC Buttler', 'PD Salt', 'DJ Malan', 'JM Bairstow'],
                'bowl': ['MA Wood', 'CR Woakes', 'AU Rashid', 'JC Archer'],
            },
        },
    },
]


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
    tc  = ipl_team_codes.get(team, 0) if fmt == 'ipl' else 0
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
    tc  = ipl_team_codes.get(team, 0) if fmt == 'ipl' else 0
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


# ─── API ROUTES ──────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('dashboard.html')


@app.route('/api/schedule')
def schedule():
    """
    Return the full schedule for both IPL and T20I.
    Each match includes squad lists so the frontend can drive /predict calls.
    player_in_profile flags let the UI skip players not in the ML model.
    """
    def annotate(matches, bat_pool, bowl_pool):
        out = []
        for m in matches:
            mc = {k: v for k, v in m.items() if k != 'squads'}
            mc['teams'] = {}
            for team, sq in m['squads'].items():
                mc['teams'][team] = {
                    'bat':  [{'name': p, 'in_profile': p in bat_pool}  for p in sq['bat']],
                    'bowl': [{'name': p, 'in_profile': p in bowl_pool} for p in sq['bowl']],
                }
            out.append(mc)
        return out

    return jsonify({
        'ipl':  annotate(IPL_SCHEDULE,  ipl_bat_p, ipl_bowl_p),
        't20i': annotate(T20I_SCHEDULE, t20_bat_p, t20_bowl_p),
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


if __name__ == '__main__':
    app.run(debug=True)
