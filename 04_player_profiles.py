"""
04_player_profiles.py
---------------------
Builds inference-time player profiles and venue context from historical records.

Profiles store career form, rolling windows, and per-venue splits so prediction
can mirror the same 5-match venue threshold used during training.
"""
import pickle

import pandas as pd

RETIRED_CUTOFF = '2024-01-01'
EXTRA_RETIRED_T20 = {'V Kohli', 'RG Sharma'}
MIN_VENUE_MATCHES = 5
PITCH_TYPE_MAP = {'bowling': 0, 'balanced': 1, 'batting': 2}


def classify_pitch_type(avg_runs):
    if avg_runs > 170:
        return 'batting'
    if avg_runs < 150:
        return 'bowling'
    return 'balanced'


def build_venue_context(df):
    innings_level = (
        df.groupby(['venue', 'match_id', 'team'], as_index=False)
        .agg(
            innings_runs=('runs', 'sum'),
            innings_wkts=('wickets', 'sum'),
            chasing_win=('chasing_win', 'first'),
        )
    )
    match_level = (
        innings_level.groupby(['venue', 'match_id'], as_index=False)
        .agg(
            venue_match_avg_runs=('innings_runs', 'mean'),
            venue_match_avg_wkts=('innings_wkts', 'mean'),
            chasing_win=('chasing_win', 'first'),
        )
    )
    by_venue = (
        match_level.groupby('venue')
        .agg(
            venue_avg_runs=('venue_match_avg_runs', 'mean'),
            venue_avg_wkts=('venue_match_avg_wkts', 'mean'),
            dew_factor=('chasing_win', 'mean'),
            venue_matches=('match_id', 'nunique'),
        )
        .reset_index()
    )
    venues = {
        row['venue']: {
            'venue_avg_runs': round(row['venue_avg_runs'], 3),
            'venue_avg_wkts': round(row['venue_avg_wkts'], 3),
            'pitch_type': classify_pitch_type(row['venue_avg_runs']),
            'pitch_type_encoded': PITCH_TYPE_MAP[classify_pitch_type(row['venue_avg_runs'])],
            'dew_factor': round(0.5 if pd.isna(row['dew_factor']) else row['dew_factor'], 3),
            'chasing_advantage': int((0.5 if pd.isna(row['dew_factor']) else row['dew_factor']) > 0.6),
            'venue_matches': int(row['venue_matches']),
        }
        for _, row in by_venue.iterrows()
    }
    global_dew_factor = match_level['chasing_win'].dropna().mean()
    if pd.isna(global_dew_factor):
        global_dew_factor = 0.5
    global_avg_runs = match_level['venue_match_avg_runs'].mean()
    venues['_global'] = {
        'venue_avg_runs': round(global_avg_runs, 3),
        'venue_avg_wkts': round(match_level['venue_match_avg_wkts'].mean(), 3),
        'pitch_type': classify_pitch_type(global_avg_runs),
        'pitch_type_encoded': PITCH_TYPE_MAP[classify_pitch_type(global_avg_runs)],
        'dew_factor': round(global_dew_factor, 3),
        'chasing_advantage': int(global_dew_factor > 0.6),
        'venue_matches': int(match_level['match_id'].nunique()),
    }
    return venues


def build_bat_profile(df, windows, min_innings=5):
    df = df.sort_values(['player', 'date', 'match_id'])
    bat = df[df['balls_faced'] > 0].copy()
    bat['sr'] = bat['runs'] / bat['balls_faced'] * 100

    profiles = {}
    for player, grp in bat.groupby('player'):
        if len(grp) < min_innings:
            continue
        profile = {
            'player': player,
            'innings': len(grp),
            'career_avg': round(grp['runs'].mean(), 3),
            'career_sr': round(grp['sr'].mean(), 3),
            'last_date': grp['date'].max().strftime('%Y-%m-%d'),
            'matches_played': len(grp),
            'venue_stats': {},
        }
        for window in windows:
            profile[f'avg_runs_{window}'] = round(grp['runs'].tail(window).mean(), 3)
            profile[f'std_runs_{window}'] = round(grp['runs'].tail(window).std(), 3)
            profile[f'avg_sr_{window}'] = round(grp['sr'].tail(window).mean(), 3)

        for venue, venue_grp in grp.groupby('venue'):
            profile['venue_stats'][venue] = {
                'matches': len(venue_grp),
                'avg_runs': round(venue_grp['runs'].mean(), 3),
                'avg_sr': round(venue_grp['sr'].mean(), 3),
                'qualified': len(venue_grp) >= MIN_VENUE_MATCHES,
                'venue_experience_weight': round(len(venue_grp) / (len(venue_grp) + MIN_VENUE_MATCHES), 3),
            }
        profiles[player] = profile
    return profiles


def build_bowl_profile(df, windows, min_innings=5):
    df = df.sort_values(['player', 'date', 'match_id'])
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    profiles = {}
    for player, grp in bowl.groupby('player'):
        if len(grp) < min_innings:
            continue
        profile = {
            'player': player,
            'innings': len(grp),
            'career_wkt_avg': round(grp['wickets'].mean(), 3),
            'career_econ': round(grp['economy'].mean(), 3),
            'last_date': grp['date'].max().strftime('%Y-%m-%d'),
            'bowling_matches': len(grp),
            'venue_stats': {},
        }
        for window in windows:
            profile[f'avg_wkts_{window}'] = round(grp['wickets'].tail(window).mean(), 3)
            profile[f'std_wkts_{window}'] = round(grp['wickets'].tail(window).std(), 3)
            profile[f'avg_econ_{window}'] = round(grp['economy'].tail(window).mean(), 3)

        for venue, venue_grp in grp.groupby('venue'):
            profile['venue_stats'][venue] = {
                'matches': len(venue_grp),
                'avg_wkts': round(venue_grp['wickets'].mean(), 3),
                'avg_econ': round(venue_grp['economy'].mean(), 3),
                'qualified': len(venue_grp) >= MIN_VENUE_MATCHES,
                'venue_experience_weight': round(len(venue_grp) / (len(venue_grp) + MIN_VENUE_MATCHES), 3),
            }
        profiles[player] = profile
    return profiles


def flag_retired(profiles, retired_set):
    for profile in profiles.values():
        profile['retired'] = profile['player'] in retired_set
    return profiles


def build_era_context(df):
    years = df['date'].dt.year
    year_min = int(years.min())
    year_max = int(years.max())
    latest_modern_era = int(year_max >= 2020)
    latest_era_weight = 1.0 if year_max == year_min else 1.0
    return {
        'year_min': year_min,
        'year_max': year_max,
        'latest_year': year_max,
        'latest_modern_era': latest_modern_era,
        'latest_era_weight': latest_era_weight,
    }


if __name__ == '__main__':
    ipl = pd.read_csv('output/ipl_records.csv', parse_dates=['date'])
    t20 = pd.read_csv('output/t20_records.csv', parse_dates=['date'])

    ipl_last = ipl.groupby('player')['date'].max()
    ipl_retired = set(ipl_last[ipl_last < RETIRED_CUTOFF].index)
    t20_last = t20.groupby('player')['date'].max()
    t20_retired = set(t20_last[t20_last < RETIRED_CUTOFF].index) | EXTRA_RETIRED_T20

    ipl_bat_p = flag_retired(build_bat_profile(ipl, [5, 10, 15]), ipl_retired)
    ipl_bowl_p = flag_retired(build_bowl_profile(ipl, [5, 10, 15]), ipl_retired)
    t20_bat_p = flag_retired(build_bat_profile(t20, [5, 10, 20]), t20_retired)
    t20_bowl_p = flag_retired(build_bowl_profile(t20, [5, 10, 20]), t20_retired)

    with open('models/player_profiles.pkl', 'wb') as f:
        pickle.dump({
            'ipl_bat': ipl_bat_p,
            'ipl_bowl': ipl_bowl_p,
            't20_bat': t20_bat_p,
            't20_bowl': t20_bowl_p,
            'ipl_retired': ipl_retired,
            't20_retired': t20_retired,
            'ipl_venue_context': build_venue_context(ipl),
            't20_venue_context': build_venue_context(t20),
            'ipl_team_codes': {team: idx for idx, team in enumerate(sorted(ipl['team'].dropna().unique()))},
            'ipl_era_context': build_era_context(ipl),
            't20_era_context': build_era_context(t20),
        }, f)

    print(f"IPL batters: {len(ipl_bat_p)}, bowlers: {len(ipl_bowl_p)}")
    print(f"T20I batters: {len(t20_bat_p)}, bowlers: {len(t20_bowl_p)}")
    print(f"IPL retired: {len(ipl_retired)}, T20I retired: {len(t20_retired)}")
