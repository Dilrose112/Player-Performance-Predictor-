"""
02_feature_engineering.py
--------------------------
Builds rolling-window features for batting and bowling models.

IPL Model  (12 features): rolling 5/10/15 innings + venue_avg_runs + team_encoded
T20I Model (10 features): rolling 5/10/20 innings  (no venue/team context)

Key design: shift(1) before every rolling window to prevent data leakage.
"""
import pandas as pd
import numpy as np

# ─── RETIRED PLAYER DETECTION ───────────────────────────────────────────────
def detect_retired(df, cutoff='2024-01-01', extra_retired=None):
    last_active = df.groupby('player')['date'].max()
    retired = set(last_active[last_active < pd.Timestamp(cutoff)].index)
    if extra_retired:
        retired.update(extra_retired)
    return retired

# ─── IPL BATTING FEATURES ────────────────────────────────────────────────────
def make_ipl_bat_features(df):
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    bat = df[df['balls_faced'] > 0].copy()
    bat['strike_rate'] = bat['runs'] / bat['balls_faced'] * 100

    for w in [5, 10, 15]:
        bat[f'avg_runs_{w}'] = bat.groupby('player')['runs'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        bat[f'std_runs_{w}'] = bat.groupby('player')['runs'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
        bat[f'avg_sr_{w}']   = bat.groupby('player')['strike_rate'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())

    bat['career_avg']      = bat.groupby('player')['runs'].transform(
        lambda x: x.shift(1).expanding().mean())
    bat['career_sr']       = bat.groupby('player')['strike_rate'].transform(
        lambda x: x.shift(1).expanding().mean())
    bat['matches_played']  = bat.groupby('player').cumcount()

    # IPL-specific: venue run-rate context
    venue_avg = df.groupby('venue')['runs'].mean().rename('venue_avg_runs')
    bat = bat.join(venue_avg, on='venue')
    bat['team_encoded'] = bat['team'].astype('category').cat.codes
    return bat

# ─── IPL BOWLING FEATURES ────────────────────────────────────────────────────
def make_ipl_bowl_features(df):
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    for w in [5, 10, 15]:
        bowl[f'avg_wkts_{w}'] = bowl.groupby('player')['wickets'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        bowl[f'std_wkts_{w}'] = bowl.groupby('player')['wickets'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
        bowl[f'avg_econ_{w}'] = bowl.groupby('player')['economy'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())

    bowl['career_wkt_avg']  = bowl.groupby('player')['wickets'].transform(
        lambda x: x.shift(1).expanding().mean())
    bowl['bowling_matches'] = bowl.groupby('player').cumcount()

    venue_avg = df.groupby('venue')['runs_conceded'].mean().rename('venue_avg_conceded')
    bowl = bowl.join(venue_avg, on='venue')
    bowl['team_encoded'] = bowl['team'].astype('category').cat.codes
    return bowl

# ─── T20I BATTING FEATURES ───────────────────────────────────────────────────
def make_t20_bat_features(df):
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    bat = df[df['balls_faced'] > 0].copy()
    bat['strike_rate'] = bat['runs'] / bat['balls_faced'] * 100

    for w in [5, 10, 20]:   # wider window — inter-year form cycles
        bat[f'avg_runs_{w}'] = bat.groupby('player')['runs'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        bat[f'std_runs_{w}'] = bat.groupby('player')['runs'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
        bat[f'avg_sr_{w}']   = bat.groupby('player')['strike_rate'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())

    bat['career_avg']     = bat.groupby('player')['runs'].transform(
        lambda x: x.shift(1).expanding().mean())
    bat['career_sr']      = bat.groupby('player')['strike_rate'].transform(
        lambda x: x.shift(1).expanding().mean())
    bat['matches_played'] = bat.groupby('player').cumcount()
    return bat

# ─── T20I BOWLING FEATURES ───────────────────────────────────────────────────
def make_t20_bowl_features(df):
    df = df.sort_values(['player', 'date']).reset_index(drop=True)
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    for w in [5, 10, 20]:
        bowl[f'avg_wkts_{w}'] = bowl.groupby('player')['wickets'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())
        bowl[f'std_wkts_{w}'] = bowl.groupby('player')['wickets'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std())
        bowl[f'avg_econ_{w}'] = bowl.groupby('player')['economy'].transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean())

    bowl['career_wkt_avg']  = bowl.groupby('player')['wickets'].transform(
        lambda x: x.shift(1).expanding().mean())
    bowl['bowling_matches'] = bowl.groupby('player').cumcount()
    return bowl

if __name__ == '__main__':
    ipl = pd.read_csv('ipl_records.csv', parse_dates=['date'])
    t20 = pd.read_csv('t20_records.csv', parse_dates=['date'])

    ipl_retired = detect_retired(ipl, extra_retired=None)
    t20_retired = detect_retired(t20, extra_retired=['V Kohli', 'RG Sharma'])

    print(f"IPL retired: {len(ipl_retired)}, T20I retired: {len(t20_retired)}")

    ipl_bat  = make_ipl_bat_features(ipl)
    ipl_bowl = make_ipl_bowl_features(ipl)
    t20_bat  = make_t20_bat_features(t20)
    t20_bowl = make_t20_bowl_features(t20)

    ipl_bat.to_csv('ipl_bat_features.csv',   index=False)
    ipl_bowl.to_csv('ipl_bowl_features.csv', index=False)
    t20_bat.to_csv('t20_bat_features.csv',   index=False)
    t20_bowl.to_csv('t20_bowl_features.csv', index=False)
    print("Feature CSVs saved.")
