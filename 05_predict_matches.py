"""
05_predict_matches.py
----------------------
Uses trained models + player profiles to generate prediction windows
for upcoming IPL 2026 and T20I matches.

IPL 2026 squads: CONFIRMED from ESPNcricinfo (March 2026)
  - RCB: Virat Kohli, Phil Salt, Rajat Patidar(c), Venkatesh Iyer, Tim David,
          Jitesh Sharma, Krunal Pandya, Romario Shepherd, Jacob Bethell,
          Bhuvneshwar Kumar, Josh Hazlewood, Yash Dayal, Suyash Sharma
  - CSK: Ruturaj Gaikwad(c), Sanju Samson, MS Dhoni, Ayush Mhatre, Shivam Dube,
          Dewald Brevis, Prashant Veer, Kartik Sharma, Matt Henry, Noor Ahmad,
          Khaleel Ahmed, Rahul Chahar, Akeal Hosein, Nathan Ellis
  - MI:  Hardik Pandya(c), Rohit Sharma, Suryakumar Yadav, Tilak Varma,
          Quinton de Kock, Trent Boult, Jasprit Bumrah, Nuwan Thushara,
          Ryan Rickelton, Naman Tiwari
  - KKR: Ajinkya Rahane(c), Venkatesh Iyer, Angkrish Raghuvanshi, Rinku Singh,
          Quinton de Kock, Andre Russell, Sunil Narine, Cameron Green,
          Matheesha Pathirana, Harshit Rana, Blessing Muzarabani
  - SRH: Pat Cummins(c), Travis Head, Abhishek Sharma, Heinrich Klaasen,
          Nitish Reddy, Ishan Kishan, Mohammed Shami, Harshal Patel,
          Jaydev Unadkat, Simarjeet Singh
  - RR:  Riyan Parag(c), Yashasvi Jaiswal, Shimron Hetmyer, Dhruv Jurel,
          Maheesh Theekshana, Wanindu Hasaranga, Sandeep Sharma, Kuldeep Sen
  - GT:  Shubman Gill(c), Jos Buttler, B Sai Sudharsan, Rashid Khan,
          Kagiso Rabada, Mohammed Siraj, Shahrukh Khan
  - PBKS: Shreyas Iyer(c), Priyansh Arya, Prabhsimran Singh, Nehal Wadhera,
           Marcus Stoinis, Arshdeep Singh, Yuzvendra Chahal, Marco Jansen,
           Harpreet Brar, Lockie Ferguson, Shashank Singh
  - DC:  Axar Patel(c), KL Rahul, Jake Fraser-McGurk, Tristan Stubbs,
          Kuldeep Yadav, Mitchell Starc, Mukesh Kumar, Rasikh Dar
  - LSG: Rishabh Pant(c), Aiden Markram, Nicholas Pooran, Mitchell Marsh,
          Wanindu Hasaranga, Mohammed Shami, Mayank Yadav, Avesh Khan,
          Shahbaz Ahmed, Ravi Bishnoi
"""
import pickle
import numpy as np
import pandas as pd

# ─── LOAD MODELS & PROFILES ─────────────────────────────────────────────────
with open('models/ipl_models.pkl', 'rb') as f:
    IPL_M = pickle.load(f)
with open('models/t20_models.pkl', 'rb') as f:
    T20_M = pickle.load(f)
with open('models/player_profiles.pkl', 'rb') as f:
    PROFILES = pickle.load(f)

MIN_VENUE_MATCHES = 5

# ─── PREDICTION FUNCTIONS ────────────────────────────────────────────────────
def get_venue_context(venue_contexts, venue):
    return venue_contexts.get(venue, venue_contexts['_global'])


def get_era_features(match_date, era_context):
    match_year = pd.to_datetime(match_date).year
    year_min = era_context['year_min']
    year_max = era_context['year_max']

    # Temporal features let the model learn how T20 scoring evolves by season
    # instead of baking in manual run adjustments.
    modern_era = int(match_year >= 2020)
    if year_max == year_min:
        era_weight = era_context.get('latest_era_weight', 1.0)
    else:
        clamped_year = min(max(match_year, year_min), year_max)
        year_norm = (clamped_year - year_min) / (year_max - year_min)
        era_weight = 0.5 + year_norm * 0.5
    return {'modern_era': modern_era, 'era_weight': era_weight}


def get_batting_feature_row(ps, venue, team_code, venue_context, era_features):
    venue_stats = ps.get('venue_stats', {}).get(venue, {})
    qualified_venue = venue_stats.get('matches', 0) >= MIN_VENUE_MATCHES
    venue_matches = venue_stats.get('matches', 0)

    feature_map = {
        'avg_runs_5': ps.get('avg_runs_5', ps['career_avg']),
        'avg_runs_10': ps.get('avg_runs_10', ps['career_avg']),
        'avg_runs_15': ps.get('avg_runs_15', ps['career_avg']),
        'avg_runs_20': ps.get('avg_runs_20', ps['career_avg']),
        'std_runs_5': ps.get('std_runs_5', 15.0),
        'std_runs_10': ps.get('std_runs_10', 15.0),
        'career_avg': ps['career_avg'],
        'avg_sr_5': ps.get('avg_sr_5', ps['career_sr']),
        'avg_sr_10': ps.get('avg_sr_10', ps['career_sr']),
        'career_sr': ps['career_sr'],
        'matches_played': ps['matches_played'],
        'modern_era': era_features['modern_era'],
        'era_weight': era_features['era_weight'],
        'venue_avg_runs': venue_context['venue_avg_runs'],
        'venue_avg_wkts': venue_context['venue_avg_wkts'],
        'pitch_type_encoded': venue_context.get('pitch_type_encoded', 1),
        'dew_factor': venue_context.get('dew_factor', 0.5),
        'chasing_advantage': venue_context.get('chasing_advantage', 0),
        'player_venue_avg_runs': venue_stats.get('avg_runs', ps['career_avg']) if qualified_venue else ps['career_avg'],
        'player_venue_avg_sr': venue_stats.get('avg_sr', ps['career_sr']) if qualified_venue else ps['career_sr'],
        'player_venue_matches': venue_matches,
        'venue_experience_weight': venue_stats.get('venue_experience_weight', venue_matches / (venue_matches + MIN_VENUE_MATCHES) if venue_matches or MIN_VENUE_MATCHES else 0.0),
        'team_encoded': team_code,
    }
    return feature_map


def get_bowling_feature_row(ps, venue, team_code, venue_context, era_features):
    venue_stats = ps.get('venue_stats', {}).get(venue, {})
    qualified_venue = venue_stats.get('matches', 0) >= MIN_VENUE_MATCHES
    venue_matches = venue_stats.get('matches', 0)

    feature_map = {
        'avg_wkts_5': ps.get('avg_wkts_5', ps['career_wkt_avg']),
        'avg_wkts_10': ps.get('avg_wkts_10', ps['career_wkt_avg']),
        'avg_wkts_15': ps.get('avg_wkts_15', ps['career_wkt_avg']),
        'avg_wkts_20': ps.get('avg_wkts_20', ps['career_wkt_avg']),
        'std_wkts_5': ps.get('std_wkts_5', 0.8),
        'career_wkt_avg': ps['career_wkt_avg'],
        'career_econ': ps.get('career_econ', 8.0),
        'avg_econ_5': ps.get('avg_econ_5', ps.get('career_econ', 8.0)),
        'avg_econ_10': ps.get('avg_econ_10', ps.get('career_econ', 8.0)),
        'bowling_matches': ps['bowling_matches'],
        'modern_era': era_features['modern_era'],
        'era_weight': era_features['era_weight'],
        'venue_avg_runs': venue_context['venue_avg_runs'],
        'venue_avg_wkts': venue_context['venue_avg_wkts'],
        'pitch_type_encoded': venue_context.get('pitch_type_encoded', 1),
        'dew_factor': venue_context.get('dew_factor', 0.5),
        'chasing_advantage': venue_context.get('chasing_advantage', 0),
        'player_venue_avg_wkts': venue_stats.get('avg_wkts', ps['career_wkt_avg']) if qualified_venue else ps['career_wkt_avg'],
        'player_venue_avg_econ': venue_stats.get('avg_econ', ps.get('career_econ', 8.0)) if qualified_venue else ps.get('career_econ', 8.0),
        'player_venue_matches': venue_matches,
        'venue_experience_weight': venue_stats.get('venue_experience_weight', venue_matches / (venue_matches + MIN_VENUE_MATCHES) if venue_matches or MIN_VENUE_MATCHES else 0.0),
        'team_encoded': team_code,
    }
    return feature_map


def build_feature_array(feature_map, feature_names):
    return pd.DataFrame([{name: feature_map.get(name, 0.0) for name in feature_names}])


def predict_ipl_bat(ps, venue, team, match_date):
    venue_context = get_venue_context(PROFILES['ipl_venue_context'], venue)
    team_code = PROFILES['ipl_team_codes'].get(team, 0)
    era_features = get_era_features(match_date, PROFILES['ipl_era_context'])
    feature_map = get_batting_feature_row(ps, venue, team_code, venue_context, era_features)
    X = build_feature_array(feature_map, IPL_M['bat_feats'])
    lo = max(0.0, float(IPL_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bat'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_ipl_bowl(ps, venue, team, match_date):
    venue_context = get_venue_context(PROFILES['ipl_venue_context'], venue)
    team_code = PROFILES['ipl_team_codes'].get(team, 0)
    era_features = get_era_features(match_date, PROFILES['ipl_era_context'])
    feature_map = get_bowling_feature_row(ps, venue, team_code, venue_context, era_features)
    X = build_feature_array(feature_map, IPL_M['bowl_feats'])
    lo = max(0.0, float(IPL_M['bowl'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bowl'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bowl'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_t20_bat(ps, venue, match_date):
    venue_context = get_venue_context(PROFILES['t20_venue_context'], venue)
    era_features = get_era_features(match_date, PROFILES['t20_era_context'])
    feature_map = get_batting_feature_row(ps, venue, 0, venue_context, era_features)
    X = build_feature_array(feature_map, T20_M['bat_feats'])
    lo = max(0.0, float(T20_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(T20_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(T20_M['bat'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_t20_bowl(ps, venue, match_date):
    venue_context = get_venue_context(PROFILES['t20_venue_context'], venue)
    era_features = get_era_features(match_date, PROFILES['t20_era_context'])
    feature_map = get_bowling_feature_row(ps, venue, 0, venue_context, era_features)
    X = build_feature_array(feature_map, T20_M['bowl_feats'])
    lo = max(0.0, float(T20_M['bowl'][0.25].predict(X)[0]))
    md = max(0.0, float(T20_M['bowl'][0.50].predict(X)[0]))
    hi = max(0.0, float(T20_M['bowl'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

# ─── UPDATED IPL 2026 SQUADS (ESPNcricinfo verified) ─────────────────────────
IPL_2026_SQUADS = {
    'RCB':  {
        'batters': ['V Kohli','PD Salt','Rajat Patidar','Venkatesh Iyer',
                    'Tim David','Jitesh Sharma'],
        'bowlers': ['Bhuvneshwar Kumar','JR Hazlewood','Yash Dayal',
                    'Suyash Sharma','Krunal Pandya','Romario Shepherd']
    },
    'CSK':  {
        'batters': ['Ruturaj Gaikwad','SV Samson','MS Dhoni','A Mhatre',
                    'S Dube','D Brevis'],
        'bowlers': ['MJ Henry','Noor Ahmad','Khaleel Ahmed','R Chahar',
                    'AJ Hosein','Spencer Johnson']
    },
    'MI':   {
        'batters': ['RG Sharma','SA Yadav','Tilak Varma','HH Pandya',
                    'Ryan Rickelton'],
        'bowlers': ['JJ Bumrah','TA Boult','Nuwan Thushara','Naman Tiwari']
    },
    'KKR':  {
        'batters': ['AM Rahane','Venkatesh Iyer','Rinku Singh',
                    'Angkrish Raghuvanshi'],
        'bowlers': ['AD Russell','SP Narine','Harshit Rana',
                    'M Pathirana','Blessing Muzarabani']
    },
    'SRH':  {
        'batters': ['TM Head','Abhishek Sharma','H Klaasen',
                    'N Reddy','Ishan Kishan'],
        'bowlers': ['Pat Cummins','Mohammed Shami','HV Patel',
                    'Simarjeet Singh','JD Unadkat']
    },
    'RR':   {
        'batters': ['YBK Jaiswal','Riyan Parag','Shimron Hetmyer',
                    'Dhruv Jurel'],
        'bowlers': ['M Theekshana','Sandeep Sharma','Kuldeep Sen',
                    'Wanindu Hasaranga']
    },
    'GT':   {
        'batters': ['Shubman Gill','JC Buttler','B Sai Sudharsan',
                    'Shahrukh Khan'],
        'bowlers': ['Rashid Khan','KA Rabada','Mohammed Siraj']
    },
    'PBKS': {
        'batters': ['Shreyas Iyer','Priyansh Arya','Prabhsimran Singh',
                    'Nehal Wadhera','Marcus Stoinis'],
        'bowlers': ['Arshdeep Singh','YS Chahal','M Jansen',
                    'Harpreet Brar','L Ferguson']
    },
    'DC':   {
        'batters': ['KL Rahul','JJ Fraser-McGurk','T Stubbs','Axar Patel'],
        'bowlers': ['Kuldeep Yadav','Mitchell Starc','Mukesh Kumar','Rasikh Dar']
    },
    'LSG':  {
        'batters': ['RP Pant','AK Markram','NL Pooran','MR Marsh',
                    'Abdul Samad'],
        'bowlers': ['Mohammed Shami','Mayank Yadav','Avesh Khan',
                    'Ravi Bishnoi','WS Hasaranga']
    },
}

# ─── PREDICTION FOR ONE MATCH ─────────────────────────────────────────────────
def predict_match(home, away, venue, match_no, date, ipl_bat_p, ipl_bowl_p, ipl_retired):
    venue_context = get_venue_context(PROFILES['ipl_venue_context'], venue)
    era_features = get_era_features(date, PROFILES['ipl_era_context'])
    result = {'match': match_no, 'date': date, 'home': home,
              'away': away, 'venue': venue,
              'pitch_type': venue_context.get('pitch_type', 'balanced'),
              'dew_factor': venue_context.get('dew_factor', 0.5),
              'era_weight': era_features['era_weight'],
              'home_players': [], 'away_players': []}

    for team, side in [(home,'home_players'), (away,'away_players')]:
        sq = IPL_2026_SQUADS.get(team, {'batters':[],'bowlers':[]})
        for p in sq['batters']:
            if p in ipl_bat_p:
                lo,md,hi = predict_ipl_bat(ipl_bat_p[p], venue, team, date)
                result[side].append({
                    'name':p,'role':'BAT','team':team,
                    'retired': p in ipl_retired,
                    'pred_low':lo,'pred_mid':md,'pred_high':hi,
                    'career_avg':ipl_bat_p[p]['career_avg'],
                    'innings':ipl_bat_p[p]['innings']
                })
        for p in sq['bowlers']:
            if p in ipl_bowl_p:
                lo,md,hi = predict_ipl_bowl(ipl_bowl_p[p], venue, team, date)
                result[side].append({
                    'name':p,'role':'BOWL','team':team,
                    'retired': p in ipl_retired,
                    'pred_low':lo,'pred_mid':md,'pred_high':hi,
                    'career_avg':ipl_bowl_p[p]['career_wkt_avg'],
                    'innings':ipl_bowl_p[p]['innings']
                })
    return result

if __name__ == '__main__':
    ipl_bat_p  = PROFILES['ipl_bat']
    ipl_bowl_p = PROFILES['ipl_bowl']
    ipl_retired= PROFILES['ipl_retired']

    # Sample prediction: RCB vs SRH Match 1
    m = predict_match('RCB','SRH','M Chinnaswamy Stadium',1,'2026-03-28',
                      ipl_bat_p, ipl_bowl_p, ipl_retired)
    print(f"\n=== {m['home']} vs {m['away']} | {m['venue']} | {m['date']} ===")
    print(f"Pitch: {m['pitch_type']} | Dew factor: {m['dew_factor']:.2f} | Era weight: {m['era_weight']:.2f}")
    for p in m['home_players']:
        tag = '[RETIRED]' if p['retired'] else '[active]'
        unit = 'runs' if p['role']=='BAT' else 'wkts'
        print(f"  {tag} {p['name']:25s} [{p['role']}] "
              f"Predicted: {p['pred_low']}–{p['pred_high']} {unit} (mid:{p['pred_mid']})")
