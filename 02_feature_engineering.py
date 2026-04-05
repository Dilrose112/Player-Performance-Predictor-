"""
02_feature_engineering.py
-------------------------
Builds leakage-safe batting and bowling features for IPL and T20I models.

Contextual features added:
  - Pitch behavior  : venue_avg_runs, venue_avg_wkts, pitch_type_encoded
  - Dew / chasing   : dew_factor, chasing_advantage
  - Venue form      : player_venue_avg_runs/wkts/sr/econ, player_venue_matches,
                      venue_experience_weight
  - Era evolution   : modern_era (binary), era_weight (continuous 0.5→1.0)

ALL rolling and expanding features use shift(1) so no future data leaks into
training rows.  Venue stats below MIN_VENUE_MATCHES fall back to career averages.
"""
import numpy as np
import pandas as pd

# ─── CONSTANTS ───────────────────────────────────────────────────────────────
MIN_VENUE_MATCHES = 5                           # minimum past matches at a venue
                                                # before venue-specific stats are trusted
PITCH_TYPE_MAP = {'bowling': 0, 'balanced': 1, 'batting': 2}


# ─── UTILITIES ───────────────────────────────────────────────────────────────

def detect_retired(df, cutoff='2024-01-01', extra_retired=None):
    """Return the set of players whose last recorded match predates cutoff."""
    last_active = df.groupby('player')['date'].max()
    retired = set(last_active[last_active < pd.Timestamp(cutoff)].index)
    if extra_retired:
        retired.update(extra_retired)
    return retired


def prepare_base_frame(df):
    """Sort chronologically so all cumulative transforms are forward-only."""
    return df.sort_values(['date', 'match_id', 'player']).reset_index(drop=True)


# ─── ERA FEATURES ────────────────────────────────────────────────────────────

def add_era_features(df):
    """
    Expose T20's upward scoring trend to the model as learned signals rather
    than hard-coded run bumps.

    modern_era  : binary flag, 1 if season >= 2020.
    era_weight  : continuous value in [0.5, 1.0] based on normalised year,
                  so the model can smoothly learn trend effects.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['modern_era'] = (df['year'] >= 2020).astype(int)

    year_min = df['year'].min()
    year_max = df['year'].max()
    if year_max == year_min:
        df['year_norm'] = 1.0
    else:
        df['year_norm'] = (df['year'] - year_min) / (year_max - year_min)
    # Scale to [0.5, 1.0] so even the earliest season gets a non-zero weight
    df['era_weight'] = 0.5 + df['year_norm'] * 0.5
    return df


# ─── PITCH TYPE ──────────────────────────────────────────────────────────────

def classify_pitch_type(avg_runs):
    """
    Categorise a venue surface from its historical average innings score.
      > 170  → batting-friendly
      < 150  → bowling-friendly
      else   → balanced
    """
    if avg_runs > 170:
        return 'batting'
    if avg_runs < 150:
        return 'bowling'
    return 'balanced'


# ─── MATCH-LEVEL CONTEXT (PITCH + DEW) ───────────────────────────────────────

def add_match_context(df):
    """
    Attach per-match venue context to every player row using only prior matches.

    Computes at match level:
      venue_avg_runs      : expanding mean of innings scores at the venue
                            (shift(1) prevents the current match leaking in)
      venue_avg_wkts      : same for wickets
      dew_factor          : expanding mean of chasing_win at the venue —
                            proxy for how often dew helps the chasing team
      chasing_advantage   : 1 if dew_factor > 0.6 else 0
      pitch_type          : categorical label derived from venue_avg_runs
      pitch_type_encoded  : ordinal encoding (bowling=0, balanced=1, batting=2)

    Venues with no prior history fall back to global averages.
    """
    # Step 1 – innings-level aggregates (one row per team per match)
    innings_level = (
        df.groupby(['venue', 'match_id', 'team'], as_index=False)
        .agg(
            date=('date', 'first'),
            innings_runs=('runs', 'sum'),
            innings_wkts=('wickets', 'sum'),
            chasing_win=('chasing_win', 'first'),
        )
    )

    # Step 2 – match-level aggregates (one row per match)
    match_level = (
        innings_level.groupby(['venue', 'match_id'], as_index=False)
        .agg(
            date=('date', 'first'),
            venue_match_avg_runs=('innings_runs', 'mean'),
            venue_match_avg_wkts=('innings_wkts', 'mean'),
            chasing_win=('chasing_win', 'first'),
        )
        .sort_values(['venue', 'date', 'match_id'])
    )

    # Step 3 – leakage-safe rolling stats: shift(1) then expanding mean.
    # Feature for match N uses only matches 1…N-1 at this venue.
    match_level['venue_avg_runs'] = match_level.groupby('venue')['venue_match_avg_runs'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    match_level['venue_avg_wkts'] = match_level.groupby('venue')['venue_match_avg_wkts'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Dew proxy: historical chasing win rate at this venue (prior matches only)
    match_level['dew_factor'] = match_level.groupby('venue')['chasing_win'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Global fallbacks for venues that appear for the first time
    global_avg_runs   = match_level['venue_match_avg_runs'].mean()
    global_avg_wkts   = match_level['venue_match_avg_wkts'].mean()
    global_dew_factor = match_level['chasing_win'].dropna().mean()
    if pd.isna(global_dew_factor):
        global_dew_factor = 0.5

    match_level['venue_avg_runs']  = match_level['venue_avg_runs'].fillna(global_avg_runs)
    match_level['venue_avg_wkts']  = match_level['venue_avg_wkts'].fillna(global_avg_wkts)
    match_level['dew_factor']      = match_level['dew_factor'].fillna(global_dew_factor)

    # Derived pitch / dew features
    match_level['chasing_advantage']  = (match_level['dew_factor'] > 0.6).astype(int)
    match_level['pitch_type']         = match_level['venue_avg_runs'].map(classify_pitch_type)
    match_level['pitch_type_encoded'] = match_level['pitch_type'].map(PITCH_TYPE_MAP).astype(int)

    return df.merge(
        match_level[[
            'venue', 'match_id',
            'venue_avg_runs', 'venue_avg_wkts',
            'pitch_type', 'pitch_type_encoded',
            'dew_factor', 'chasing_advantage',
        ]],
        on=['venue', 'match_id'],
        how='left',
    )


# ─── PLAYER-AT-VENUE FEATURES ────────────────────────────────────────────────

def add_batting_player_venue_features(bat):
    """
    Per-(player, venue) rolling batting stats — leakage-free.

    player_venue_matches       : cumulative count of prior appearances at venue
    player_venue_avg_runs      : expanding mean of runs at venue (shift(1))
    player_venue_avg_sr        : expanding mean of strike-rate at venue

    If fewer than MIN_VENUE_MATCHES prior appearances, venue stats are set to NaN
    and filled with player's global career average (same fallback as inference).

    venue_experience_weight    : smooth 0→1 weight; reaches 0.5 at 5 matches
                                 and approaches 1 asymptotically.
    """
    bat['player_venue_matches'] = bat.groupby(['player', 'venue']).cumcount()

    bat['player_venue_avg_runs'] = bat.groupby(['player', 'venue'])['runs'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    bat['player_venue_avg_sr'] = bat.groupby(['player', 'venue'])['strike_rate'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Null out venue stats when sample is too small to be reliable
    below_threshold = bat['player_venue_matches'] < MIN_VENUE_MATCHES
    bat.loc[below_threshold, ['player_venue_avg_runs', 'player_venue_avg_sr']] = np.nan

    # Fall back to career averages so no NaNs reach the model
    bat['player_venue_avg_runs'] = bat['player_venue_avg_runs'].fillna(bat['career_avg'])
    bat['player_venue_avg_sr']   = bat['player_venue_avg_sr'].fillna(bat['career_sr'])

    # Blending weight: high when venue sample is large, low when sparse
    bat['venue_experience_weight'] = (
        bat['player_venue_matches'] / (bat['player_venue_matches'] + MIN_VENUE_MATCHES)
    )
    return bat


def add_bowling_player_venue_features(bowl):
    """
    Per-(player, venue) rolling bowling stats — leakage-free.
    Mirrors the batting version; uses wickets and economy.
    """
    bowl['player_venue_matches'] = bowl.groupby(['player', 'venue']).cumcount()

    bowl['player_venue_avg_wkts'] = bowl.groupby(['player', 'venue'])['wickets'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    bowl['player_venue_avg_econ'] = bowl.groupby(['player', 'venue'])['economy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    below_threshold = bowl['player_venue_matches'] < MIN_VENUE_MATCHES
    bowl.loc[below_threshold, ['player_venue_avg_wkts', 'player_venue_avg_econ']] = np.nan

    bowl['player_venue_avg_wkts'] = bowl['player_venue_avg_wkts'].fillna(bowl['career_wkt_avg'])
    bowl['player_venue_avg_econ'] = bowl['player_venue_avg_econ'].fillna(bowl['career_econ'])

    bowl['venue_experience_weight'] = (
        bowl['player_venue_matches'] / (bowl['player_venue_matches'] + MIN_VENUE_MATCHES)
    )
    return bowl


# ─── MAIN FEATURE BUILDERS ───────────────────────────────────────────────────

def build_batting_features(df, windows, include_team):
    """
    Full batting feature pipeline:
      1. Sort chronologically (leakage guard)
      2. Attach match-level pitch / dew context
      3. Attach era features
      4. Compute rolling windows (shift-based, no leakage)
      5. Compute career aggregates (shift-based)
      6. Attach player-at-venue features
      7. Optionally encode team
    """
    df = add_era_features(add_match_context(prepare_base_frame(df)))
    bat = df[df['balls_faced'] > 0].copy()
    bat['strike_rate'] = bat['runs'] / bat['balls_faced'] * 100

    player_runs = bat.groupby('player')['runs']
    player_sr   = bat.groupby('player')['strike_rate']

    # Rolling windows — shift(1) excludes current match
    for window in windows:
        bat[f'avg_runs_{window}'] = player_runs.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        bat[f'std_runs_{window}'] = player_runs.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        )
        bat[f'avg_sr_{window}'] = player_sr.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )

    # Career-level aggregates (expanding, shift(1))
    bat['career_avg']     = player_runs.transform(lambda x: x.shift(1).expanding().mean())
    bat['career_sr']      = player_sr.transform(lambda x: x.shift(1).expanding().mean())
    bat['matches_played'] = bat.groupby('player').cumcount()

    bat = add_batting_player_venue_features(bat)

    if include_team:
        team_codes = {team: idx for idx, team in enumerate(sorted(df['team'].dropna().unique()))}
        bat['team_encoded'] = bat['team'].map(team_codes).fillna(-1).astype(int)

    return bat


def build_bowling_features(df, windows, include_team):
    """
    Full bowling feature pipeline — mirrors build_batting_features.
    Economy = runs_conceded / (balls_bowled / 6), clipped at 15 to cap outliers.
    """
    df = add_era_features(add_match_context(prepare_base_frame(df)))
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    player_wickets = bowl.groupby('player')['wickets']
    player_econ    = bowl.groupby('player')['economy']

    for window in windows:
        bowl[f'avg_wkts_{window}'] = player_wickets.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )
        bowl[f'std_wkts_{window}'] = player_wickets.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).std()
        )
        bowl[f'avg_econ_{window}'] = player_econ.transform(
            lambda x: x.shift(1).rolling(window, min_periods=2).mean()
        )

    bowl['career_wkt_avg']  = player_wickets.transform(lambda x: x.shift(1).expanding().mean())
    bowl['career_econ']     = player_econ.transform(lambda x: x.shift(1).expanding().mean())
    bowl['bowling_matches'] = bowl.groupby('player').cumcount()

    bowl = add_bowling_player_venue_features(bowl)

    if include_team:
        team_codes = {team: idx for idx, team in enumerate(sorted(df['team'].dropna().unique()))}
        bowl['team_encoded'] = bowl['team'].map(team_codes).fillna(-1).astype(int)

    return bowl


# ─── FORMAT-SPECIFIC ENTRY POINTS ────────────────────────────────────────────

def make_ipl_bat_features(df):
    return build_batting_features(df, windows=[5, 10, 15], include_team=True)

def make_ipl_bowl_features(df):
    return build_bowling_features(df, windows=[5, 10, 15], include_team=True)

def make_t20_bat_features(df):
    return build_batting_features(df, windows=[5, 10, 20], include_team=False)

def make_t20_bowl_features(df):
    return build_bowling_features(df, windows=[5, 10, 20], include_team=False)


# ─── MAIN ────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ipl = pd.read_csv('output/ipl_records.csv', parse_dates=['date'])
    t20 = pd.read_csv('output/t20_records.csv', parse_dates=['date'])

    ipl_retired = detect_retired(ipl, extra_retired=None)
    t20_retired = detect_retired(t20, extra_retired=['V Kohli', 'RG Sharma'])
    print(f"IPL retired: {len(ipl_retired)}, T20I retired: {len(t20_retired)}")

    ipl_bat  = make_ipl_bat_features(ipl)
    ipl_bowl = make_ipl_bowl_features(ipl)
    t20_bat  = make_t20_bat_features(t20)
    t20_bowl = make_t20_bowl_features(t20)

    ipl_bat.to_csv('output/ipl_bat_features.csv',   index=False)
    ipl_bowl.to_csv('output/ipl_bowl_features.csv', index=False)
    t20_bat.to_csv('output/t20_bat_features.csv',   index=False)
    t20_bowl.to_csv('output/t20_bowl_features.csv', index=False)
    print('Feature CSVs saved.')
