"""
04_player_profiles.py
----------------------
Builds latest player stat profiles from the raw records CSVs.
These are used at inference time to predict upcoming match performance.

For each player computes:
  - Rolling averages (last 5/10/15 or 5/10/20 innings)
  - Career averages
  - Last match date (for retired detection)
  - Retired flag (no match since 2024-01-01)
"""
import pandas as pd
import numpy as np

RETIRED_CUTOFF = '2024-01-01'
EXTRA_RETIRED_T20 = {'V Kohli', 'RG Sharma'}   # retired from T20I post-WC 2024

def build_bat_profile(df, windows, min_innings=5):
    df = df.sort_values(['player', 'date'])
    bat = df[df['balls_faced'] > 0].copy()
    bat['sr'] = bat['runs'] / bat['balls_faced'] * 100

    profiles = {}
    for player, grp in bat.groupby('player'):
        if len(grp) < min_innings:
            continue
        p = {
            'player':      player,
            'innings':     len(grp),
            'career_avg':  round(grp['runs'].mean(), 3),
            'career_sr':   round(grp['sr'].mean(), 3),
            'last_date':   grp['date'].max().strftime('%Y-%m-%d'),
            'matches_played': len(grp),
        }
        for w in windows:
            p[f'avg_runs_{w}'] = round(grp['runs'].tail(w).mean(), 3)
            p[f'std_runs_{w}'] = round(grp['runs'].tail(w).std(),  3)
            p[f'avg_sr_{w}']   = round(grp['sr'].tail(w).mean(),   3)
        profiles[player] = p
    return profiles

def build_bowl_profile(df, windows, min_innings=5):
    df = df.sort_values(['player', 'date'])
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    profiles = {}
    for player, grp in bowl.groupby('player'):
        if len(grp) < min_innings:
            continue
        p = {
            'player':          player,
            'innings':         len(grp),
            'career_wkt_avg':  round(grp['wickets'].mean(), 3),
            'last_date':       grp['date'].max().strftime('%Y-%m-%d'),
            'bowling_matches': len(grp),
        }
        for w in windows:
            p[f'avg_wkts_{w}'] = round(grp['wickets'].tail(w).mean(), 3)
            p[f'std_wkts_{w}'] = round(grp['wickets'].tail(w).std(),  3)
            p[f'avg_econ_{w}'] = round(grp['economy'].tail(w).mean(), 3)
        profiles[player] = p
    return profiles

def flag_retired(profiles, retired_set):
    for p in profiles.values():
        p['retired'] = p['player'] in retired_set
    return profiles

if __name__ == '__main__':
    ipl = pd.read_csv('ipl_records.csv', parse_dates=['date'])
    t20 = pd.read_csv('t20_records.csv', parse_dates=['date'])

    ipl_last = ipl.groupby('player')['date'].max()
    ipl_retired = set(ipl_last[ipl_last < RETIRED_CUTOFF].index)
    t20_last = t20.groupby('player')['date'].max()
    t20_retired = set(t20_last[t20_last < RETIRED_CUTOFF].index) | EXTRA_RETIRED_T20

    ipl_bat_p  = flag_retired(build_bat_profile(ipl,  [5,10,15]), ipl_retired)
    ipl_bowl_p = flag_retired(build_bowl_profile(ipl, [5,10,15]), ipl_retired)
    t20_bat_p  = flag_retired(build_bat_profile(t20,  [5,10,20]), t20_retired)
    t20_bowl_p = flag_retired(build_bowl_profile(t20, [5,10,20]), t20_retired)

    import pickle
    with open('player_profiles.pkl', 'wb') as f:
        pickle.dump({
            'ipl_bat': ipl_bat_p,  'ipl_bowl': ipl_bowl_p,
            't20_bat': t20_bat_p,  't20_bowl': t20_bowl_p,
            'ipl_retired': ipl_retired, 't20_retired': t20_retired,
        }, f)

    print(f"IPL batters: {len(ipl_bat_p)}, bowlers: {len(ipl_bowl_p)}")
    print(f"T20I batters: {len(t20_bat_p)}, bowlers: {len(t20_bowl_p)}")
    print(f"IPL retired: {len(ipl_retired)}, T20I retired: {len(t20_retired)}")
