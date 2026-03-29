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

# ─── VENUE & TEAM LOOKUPS (from training data) ───────────────────────────────
ipl_df = pd.read_csv('output/ipl_records.csv')
VENUE_BAT_AVG  = ipl_df.groupby('venue')['runs'].mean().to_dict()
VENUE_BOWL_AVG = ipl_df.groupby('venue')['runs_conceded'].mean().to_dict()
TEAM_CODES     = {t: i for i, t in enumerate(sorted(ipl_df['team'].unique()))}
GLOBAL_BAT_AVG = ipl_df['runs'].mean()
GLOBAL_BOWL_AVG= ipl_df['runs_conceded'].mean()

# ─── PREDICTION FUNCTIONS ────────────────────────────────────────────────────
def predict_ipl_bat(ps, venue, team):
    va = VENUE_BAT_AVG.get(venue, GLOBAL_BAT_AVG)
    tc = TEAM_CODES.get(team, 0)
    X = np.array([[
        ps.get('avg_runs_5',  ps['career_avg']),
        ps.get('avg_runs_10', ps['career_avg']),
        ps.get('avg_runs_15', ps['career_avg']),
        ps.get('std_runs_5',  15.0),
        ps.get('std_runs_10', 15.0),
        ps['career_avg'], ps.get('avg_sr_5', ps['career_sr']),
        ps.get('avg_sr_10', ps['career_sr']), ps['career_sr'],
        ps['matches_played'], va, tc
    ]])
    lo = max(0.0, float(IPL_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bat'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_ipl_bowl(ps, venue, team):
    va = VENUE_BOWL_AVG.get(venue, GLOBAL_BOWL_AVG)
    tc = TEAM_CODES.get(team, 0)
    X = np.array([[
        ps.get('avg_wkts_5',  ps['career_wkt_avg']),
        ps.get('avg_wkts_10', ps['career_wkt_avg']),
        ps.get('avg_wkts_15', ps['career_wkt_avg']),
        ps.get('std_wkts_5',  0.8),
        ps['career_wkt_avg'],
        ps.get('avg_econ_5',  8.0), ps.get('avg_econ_10', 8.0),
        ps['bowling_matches'], va, tc
    ]])
    lo = max(0.0, float(IPL_M['bowl'][0.25].predict(X)[0]))
    md = max(0.0, float(IPL_M['bowl'][0.50].predict(X)[0]))
    hi = max(0.0, float(IPL_M['bowl'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_t20_bat(ps):
    X = np.array([[
        ps.get('avg_runs_5',  ps['career_avg']),
        ps.get('avg_runs_10', ps['career_avg']),
        ps.get('avg_runs_20', ps['career_avg']),
        ps.get('std_runs_5',  15.0), ps.get('std_runs_10', 15.0),
        ps['career_avg'], ps.get('avg_sr_5', ps['career_sr']),
        ps.get('avg_sr_10', ps['career_sr']), ps['career_sr'],
        ps['matches_played']
    ]])
    lo = max(0.0, float(T20_M['bat'][0.25].predict(X)[0]))
    md = max(0.0, float(T20_M['bat'][0.50].predict(X)[0]))
    hi = max(0.0, float(T20_M['bat'][0.75].predict(X)[0]))
    return round(lo,1), round(md,1), round(hi,1)

def predict_t20_bowl(ps):
    X = np.array([[
        ps.get('avg_wkts_5',  ps['career_wkt_avg']),
        ps.get('avg_wkts_10', ps['career_wkt_avg']),
        ps.get('avg_wkts_20', ps['career_wkt_avg']),
        ps.get('std_wkts_5',  0.8),
        ps['career_wkt_avg'],
        ps.get('avg_econ_5',  8.0), ps.get('avg_econ_10', 8.0),
        ps['bowling_matches']
    ]])
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
    result = {'match': match_no, 'date': date, 'home': home,
              'away': away, 'venue': venue,
              'home_players': [], 'away_players': []}

    for team, side in [(home,'home_players'), (away,'away_players')]:
        sq = IPL_2026_SQUADS.get(team, {'batters':[],'bowlers':[]})
        for p in sq['batters']:
            if p in ipl_bat_p:
                lo,md,hi = predict_ipl_bat(ipl_bat_p[p], venue, team)
                result[side].append({
                    'name':p,'role':'BAT','team':team,
                    'retired': p in ipl_retired,
                    'pred_low':lo,'pred_mid':md,'pred_high':hi,
                    'career_avg':ipl_bat_p[p]['career_avg'],
                    'innings':ipl_bat_p[p]['innings']
                })
        for p in sq['bowlers']:
            if p in ipl_bowl_p:
                lo,md,hi = predict_ipl_bowl(ipl_bowl_p[p], venue, team)
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
    for p in m['home_players']:
        tag = '[RETIRED]' if p['retired'] else '[active]'
        unit = 'runs' if p['role']=='BAT' else 'wkts'
        print(f"  {tag} {p['name']:25s} [{p['role']}] "
              f"Predicted: {p['pred_low']}–{p['pred_high']} {unit} (mid:{p['pred_mid']})")
