"""
02_feature_engineering.py
-------------------------
Builds leakage-safe batting and bowling features for IPL and T20I models.

Improvements over v1
--------------------
  - Recency-weighted venue averages: last 20 matches at venue weighted more
    than the full expanding mean — captures how venues are playing NOW, not
    how they played in 2010.
  - Exponential-decay career averages: recent innings count more than old ones,
    so a player's 2025 form dominates over their 2015 form.
  - form_vs_career ratio: dimensionless signal showing whether a player is
    above or below their long-run average right now.
  - recent_venue_avg_runs: venue average computed on last 3 seasons only —
    captures pitch renovations, boundary changes, rule changes.
  - innings_position: proxy for batting order (cumulative position in match),
    top-order vs tail matters for expected runs.
  - Updated pitch thresholds: >175 batting, <155 bowling — reflects modern T20
    scoring rather than 2010-era norms.
  - std_econ added for bowlers (consistency signal).

ALL rolling/expanding features use shift(1) — zero future leakage.
"""
import numpy as np
import pandas as pd

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
MIN_VENUE_MATCHES    = 5     # minimum prior venue appearances before trusting venue stats
VENUE_ROLLING_WINDOW = 20   # matches used for rolling venue average (recency-weighted)
RECENT_SEASONS       = 3    # seasons used for recent_venue_avg_runs
DECAY_HALFLIFE       = 30   # innings half-life for exponential-decay career average
PITCH_TYPE_MAP       = {'bowling': 0, 'balanced': 1, 'batting': 2}


# ─── UTILITIES ────────────────────────────────────────────────────────────────

def detect_retired(df, cutoff='2026-01-01', extra_retired=None):
    """Return the set of players whose last recorded match predates cutoff."""
    last_active = df.groupby('player')['date'].max()
    retired = set(last_active[last_active < pd.Timestamp(cutoff)].index)
    if extra_retired:
        retired.update(extra_retired)
    return retired


def prepare_base_frame(df):
    """Sort chronologically so all cumulative transforms are forward-only."""
    return df.sort_values(['date', 'match_id', 'player']).reset_index(drop=True)


# ─── ERA FEATURES ─────────────────────────────────────────────────────────────

def add_era_features(df):
    """
    modern_era  : 1 if year >= 2020.
    era_weight  : continuous [0.5, 1.0] — smoothly encodes T20 scoring trend.
    """
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['modern_era'] = (df['year'] >= 2020).astype(int)

    year_min = df['year'].min()
    year_max = df['year'].max()
    if year_max == year_min:
        df['era_weight'] = 1.0
    else:
        df['era_weight'] = 0.5 + (df['year'] - year_min) / (year_max - year_min) * 0.5
    return df


# ─── PITCH TYPE ───────────────────────────────────────────────────────────────

def classify_pitch_type(avg_runs):
    """
    Updated thresholds for modern T20 scoring (IPL 2023-25 avg ~173).
      > 175  → batting
      < 155  → bowling
      else   → balanced
    """
    if avg_runs > 175:
        return 'batting'
    if avg_runs < 155:
        return 'bowling'
    return 'balanced'


# ─── MATCH-LEVEL CONTEXT ──────────────────────────────────────────────────────

def add_match_context(df):
    """
    Attaches per-match venue context using ONLY prior matches (leakage-safe).

    New vs v1:
      recent_venue_avg_runs : venue average over last RECENT_SEASONS seasons only
                              (captures boundary changes, pitch refurbs, etc.)
      venue_rolling_avg     : rolling mean of last VENUE_ROLLING_WINDOW matches —
                              more responsive than the full expanding mean.
    """
    # Innings-level (one row per team per match)
    innings_level = (
        df.groupby(['venue', 'match_id', 'team'], as_index=False)
        .agg(
            date=('date', 'first'),
            innings_runs=('runs', 'sum'),
            innings_wkts=('wickets', 'sum'),
            chasing_win=('chasing_win', 'first'),
        )
    )

    # Match-level (one row per match)
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

    # Full expanding mean (leakage-free: shift(1) then expand)
    match_level['venue_avg_runs'] = match_level.groupby('venue')['venue_match_avg_runs'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    match_level['venue_avg_wkts'] = match_level.groupby('venue')['venue_match_avg_wkts'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # NEW: rolling window venue average (last VENUE_ROLLING_WINDOW matches)
    match_level['venue_rolling_avg'] = match_level.groupby('venue')['venue_match_avg_runs'].transform(
        lambda x: x.shift(1).rolling(VENUE_ROLLING_WINDOW, min_periods=3).mean()
    )

    # NEW: recent-season venue average (last RECENT_SEASONS * ~60 IPL matches ≈ 180 rows)
    # We use a 3-year rolling window on date rather than row count
    def recent_venue_mean(grp):
        out = pd.Series(index=grp.index, dtype=float)
        for i, (idx, row) in enumerate(grp.iterrows()):
            cutoff = row['date'] - pd.DateOffset(years=RECENT_SEASONS)
            past   = grp.iloc[:i]  # prior rows only (shift already applied via iloc)
            recent = past[past['date'] >= cutoff]['venue_match_avg_runs']
            out[idx] = recent.mean() if len(recent) >= 3 else np.nan
        return out

    match_level['recent_venue_avg_runs'] = (
        match_level.groupby('venue', group_keys=False)
        .apply(recent_venue_mean)
    )

    # Dew proxy
    match_level['dew_factor'] = match_level.groupby('venue')['chasing_win'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    # Global fallbacks
    global_avg_runs         = match_level['venue_match_avg_runs'].mean()
    global_avg_wkts         = match_level['venue_match_avg_wkts'].mean()
    global_dew_factor       = match_level['chasing_win'].dropna().mean()
    if pd.isna(global_dew_factor):
        global_dew_factor = 0.5

    match_level['venue_avg_runs']        = match_level['venue_avg_runs'].fillna(global_avg_runs)
    match_level['venue_avg_wkts']        = match_level['venue_avg_wkts'].fillna(global_avg_wkts)
    match_level['venue_rolling_avg']     = match_level['venue_rolling_avg'].fillna(
                                            match_level['venue_avg_runs'])
    match_level['recent_venue_avg_runs'] = match_level['recent_venue_avg_runs'].fillna(
                                            match_level['venue_avg_runs'])
    match_level['dew_factor']            = match_level['dew_factor'].fillna(global_dew_factor)

    match_level['chasing_advantage']  = (match_level['dew_factor'] > 0.6).astype(int)
    match_level['pitch_type']         = match_level['venue_avg_runs'].map(classify_pitch_type)
    match_level['pitch_type_encoded'] = match_level['pitch_type'].map(PITCH_TYPE_MAP).astype(int)

    return df.merge(
        match_level[[
            'venue', 'match_id',
            'venue_avg_runs', 'venue_avg_wkts',
            'venue_rolling_avg', 'recent_venue_avg_runs',
            'pitch_type', 'pitch_type_encoded',
            'dew_factor', 'chasing_advantage',
        ]],
        on=['venue', 'match_id'],
        how='left',
    )


# ─── PLAYER-LEVEL FEATURES ────────────────────────────────────────────────────

def _exp_decay_avg(series, halflife=DECAY_HALFLIFE):
    """
    Exponential-decay weighted average: recent innings count more.
    halflife = number of innings for a score to lose half its weight.
    Returns a shifted (leakage-safe) expanding series.
    """
    weights = np.exp(-np.log(2) / halflife * np.arange(len(series)))[::-1]

    def ewm_shift(x):
        out = pd.Series(index=x.index, dtype=float)
        vals = x.values
        for i in range(len(vals)):
            if i == 0:
                out.iloc[i] = np.nan
                continue
            w = weights[-i:]          # most recent first
            w = w / w.sum()
            out.iloc[i] = np.dot(w, vals[:i][::-1][:len(w)])
        return out

    return series.groupby(level=0).transform(ewm_shift) if series.index.nlevels > 1 \
        else ewm_shift(series)


def add_batting_player_features(bat, windows):
    """
    All batting player-level features.

    New vs v1:
      career_avg_decay   : exponential-decay weighted career average
                           (recent form matters more than ancient history)
      form_vs_career     : avg_runs_5 / career_avg — dimensionless form ratio
                           >1 means in form, <1 means below par
      innings_position   : cumulative count of innings played by this player in
                           this match (proxy for batting order depth)
    """
    bat = bat.copy()

    player_runs = bat.groupby('player')['runs']
    player_sr   = bat.groupby('player')['strike_rate']

    # Rolling windows (leakage-safe)
    for w in windows:
        bat[f'avg_runs_{w}'] = player_runs.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )
        bat[f'std_runs_{w}'] = player_runs.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std()
        )
        bat[f'avg_sr_{w}'] = player_sr.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )

    # Career averages — standard expanding (leakage-safe)
    bat['career_avg'] = player_runs.transform(lambda x: x.shift(1).expanding().mean())
    bat['career_sr']  = player_sr.transform(lambda x: x.shift(1).expanding().mean())
    bat['matches_played'] = bat.groupby('player').cumcount()

    # NEW: exponential-decay career average
    bat = bat.sort_values(['player', 'date', 'match_id'])
    decay_vals = []
    for player, grp in bat.groupby('player', sort=False):
        vals   = grp['runs'].values.astype(float)
        n      = len(vals)
        result = np.full(n, np.nan)
        hl     = DECAY_HALFLIFE
        for i in range(1, n):
            ages   = np.arange(i)[::-1]        # 0 = most recent past
            w      = np.exp(-np.log(2) / hl * ages)
            w      = w / w.sum()
            result[i] = np.dot(w, vals[:i][::-1][:len(w)])
        decay_vals.append(pd.Series(result, index=grp.index))
    bat['career_avg_decay'] = pd.concat(decay_vals).reindex(bat.index)
    bat['career_avg_decay'] = bat['career_avg_decay'].fillna(bat['career_avg'])

    # NEW: form vs career ratio (clipped to [0.1, 5] to avoid extreme values)
    short_window = windows[0]   # e.g. 5
    bat['form_vs_career'] = (
        bat[f'avg_runs_{short_window}'] / bat['career_avg'].replace(0, np.nan)
    ).clip(0.1, 5.0).fillna(1.0)

    # NEW: innings position proxy (how many innings in this match before this one)
    bat['innings_position'] = bat.groupby(['match_id', 'team']).cumcount()

    return bat


def add_batting_player_venue_features(bat):
    bat['player_venue_matches'] = bat.groupby(['player', 'venue']).cumcount()

    bat['player_venue_avg_runs'] = bat.groupby(['player', 'venue'])['runs'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    bat['player_venue_avg_sr'] = bat.groupby(['player', 'venue'])['strike_rate'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    below = bat['player_venue_matches'] < MIN_VENUE_MATCHES
    bat.loc[below, ['player_venue_avg_runs', 'player_venue_avg_sr']] = np.nan
    bat['player_venue_avg_runs'] = bat['player_venue_avg_runs'].fillna(bat['career_avg'])
    bat['player_venue_avg_sr']   = bat['player_venue_avg_sr'].fillna(bat['career_sr'])

    bat['venue_experience_weight'] = (
        bat['player_venue_matches'] / (bat['player_venue_matches'] + MIN_VENUE_MATCHES)
    )
    return bat


def add_bowling_player_features(bowl, windows):
    """
    All bowling player-level features.

    New vs v1:
      career_wkt_decay : exponential-decay weighted wicket average
      form_vs_career   : avg_wkts_5 / career_wkt_avg ratio
      std_econ         : rolling std of economy rate — consistency signal
    """
    bowl = bowl.copy()

    player_wkts = bowl.groupby('player')['wickets']
    player_econ = bowl.groupby('player')['economy']

    for w in windows:
        bowl[f'avg_wkts_{w}'] = player_wkts.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )
        bowl[f'std_wkts_{w}'] = player_wkts.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).std()
        )
        bowl[f'avg_econ_{w}'] = player_econ.transform(
            lambda x: x.shift(1).rolling(w, min_periods=2).mean()
        )

    # NEW: rolling economy std (consistency)
    bowl['std_econ_5'] = player_econ.transform(
        lambda x: x.shift(1).rolling(5, min_periods=2).std()
    )

    bowl['career_wkt_avg']  = player_wkts.transform(lambda x: x.shift(1).expanding().mean())
    bowl['career_econ']     = player_econ.transform(lambda x: x.shift(1).expanding().mean())
    bowl['bowling_matches'] = bowl.groupby('player').cumcount()

    # NEW: exponential-decay wicket average
    bowl = bowl.sort_values(['player', 'date', 'match_id'])
    decay_vals = []
    for player, grp in bowl.groupby('player', sort=False):
        vals   = grp['wickets'].values.astype(float)
        n      = len(vals)
        result = np.full(n, np.nan)
        hl     = DECAY_HALFLIFE
        for i in range(1, n):
            ages = np.arange(i)[::-1]
            w    = np.exp(-np.log(2) / hl * ages)
            w    = w / w.sum()
            result[i] = np.dot(w, vals[:i][::-1][:len(w)])
        decay_vals.append(pd.Series(result, index=grp.index))
    bowl['career_wkt_decay'] = pd.concat(decay_vals).reindex(bowl.index)
    bowl['career_wkt_decay'] = bowl['career_wkt_decay'].fillna(bowl['career_wkt_avg'])

    # NEW: form vs career ratio
    short_window = windows[0]
    bowl['form_vs_career'] = (
        bowl[f'avg_wkts_{short_window}'] / bowl['career_wkt_avg'].replace(0, np.nan)
    ).clip(0.1, 5.0).fillna(1.0)

    return bowl


def add_bowling_player_venue_features(bowl):
    bowl['player_venue_matches'] = bowl.groupby(['player', 'venue']).cumcount()

    bowl['player_venue_avg_wkts'] = bowl.groupby(['player', 'venue'])['wickets'].transform(
        lambda x: x.shift(1).expanding().mean()
    )
    bowl['player_venue_avg_econ'] = bowl.groupby(['player', 'venue'])['economy'].transform(
        lambda x: x.shift(1).expanding().mean()
    )

    below = bowl['player_venue_matches'] < MIN_VENUE_MATCHES
    bowl.loc[below, ['player_venue_avg_wkts', 'player_venue_avg_econ']] = np.nan
    bowl['player_venue_avg_wkts'] = bowl['player_venue_avg_wkts'].fillna(bowl['career_wkt_avg'])
    bowl['player_venue_avg_econ'] = bowl['player_venue_avg_econ'].fillna(bowl['career_econ'])

    bowl['venue_experience_weight'] = (
        bowl['player_venue_matches'] / (bowl['player_venue_matches'] + MIN_VENUE_MATCHES)
    )
    return bowl


# ─── MAIN FEATURE BUILDERS ────────────────────────────────────────────────────

def build_batting_features(df, windows, include_team):
    df  = add_era_features(add_match_context(prepare_base_frame(df)))
    bat = df[df['balls_faced'] > 0].copy()
    bat['strike_rate'] = bat['runs'] / bat['balls_faced'] * 100

    bat = add_batting_player_features(bat, windows)
    bat = add_batting_player_venue_features(bat)

    if include_team:
        team_codes = {team: idx for idx, team in enumerate(sorted(df['team'].dropna().unique()))}
        bat['team_encoded'] = bat['team'].map(team_codes).fillna(-1).astype(int)

    return bat


def build_bowling_features(df, windows, include_team):
    df   = add_era_features(add_match_context(prepare_base_frame(df)))
    bowl = df[df['balls_bowled'] > 0].copy()
    bowl['economy'] = (bowl['runs_conceded'] / (bowl['balls_bowled'] / 6)).clip(0, 15)

    bowl = add_bowling_player_features(bowl, windows)
    bowl = add_bowling_player_venue_features(bowl)

    if include_team:
        team_codes = {team: idx for idx, team in enumerate(sorted(df['team'].dropna().unique()))}
        bowl['team_encoded'] = bowl['team'].map(team_codes).fillna(-1).astype(int)

    return bowl


# ─── FORMAT-SPECIFIC ENTRY POINTS ─────────────────────────────────────────────

def make_ipl_bat_features(df):
    return build_batting_features(df, windows=[5, 10, 15], include_team=True)

def make_ipl_bowl_features(df):
    return build_bowling_features(df, windows=[5, 10, 15], include_team=True)

def make_t20_bat_features(df):
    return build_batting_features(df, windows=[5, 10, 20], include_team=False)

def make_t20_bowl_features(df):
    return build_bowling_features(df, windows=[5, 10, 20], include_team=False)


# ─── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import time
    ipl = pd.read_csv('output/ipl_records.csv', parse_dates=['date'])
    t20 = pd.read_csv('output/t20_records.csv', parse_dates=['date'])

    ipl_retired = detect_retired(ipl)
    t20_retired = detect_retired(t20, extra_retired=['V Kohli', 'RG Sharma'])
    print(f"IPL retired: {len(ipl_retired)}, T20I retired: {len(t20_retired)}")

    for name, fn, data in [
        ('IPL bat',   make_ipl_bat_features,   ipl),
        ('IPL bowl',  make_ipl_bowl_features,  ipl),
        ('T20I bat',  make_t20_bat_features,   t20),
        ('T20I bowl', make_t20_bowl_features,  t20),
    ]:
        t0 = time.time()
        print(f"Building {name} features…", end=' ', flush=True)
        feat = fn(data)
        elapsed = time.time() - t0
        key = name.lower().replace(' ', '_').replace('t20i', 't20')
        out = f'output/{key}_features.csv' 
        feat.to_csv(out, index=False)
        print(f"{len(feat):,} rows  →  {out}  ({elapsed:.1f}s)")

    print('\nFeature CSVs saved.')
