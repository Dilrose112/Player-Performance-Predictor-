"""
03_train_models.py
------------------
Trains quantile Gradient Boosting models for IPL and T20I batting/bowling.

Improvements over v1
--------------------
  - Sample weights: rows from recent seasons (2022+) are weighted higher
    during training so the model learns modern T20 patterns, not 2010-era ones.
    Weight = exp(year_offset / WEIGHT_HALFLIFE) normalised to mean=1.
  - Updated train/test split: test on 2025 only (single recent season) —
    gives a cleaner picture of real-world accuracy on current-era data.
  - New features from 02_feature_engineering.py added to all feature lists:
      career_avg_decay, form_vs_career, recent_venue_avg_runs,
      venue_rolling_avg, innings_position (bat), std_econ_5 (bowl),
      career_wkt_decay (bowl).
  - Tuned hyperparameters: more trees (400 vs 200), lower learning rate (0.03),
    min_samples_leaf=10 to reduce overfitting on sparse player-venue combos.
  - Extended evaluation: per-year MAE breakdown so you can see if the model
    is accurate on 2025 data specifically.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ─── SAMPLE WEIGHT CONFIG ─────────────────────────────────────────────────────
# Recent seasons are weighted higher so the model learns modern scoring norms.
# Weight for a row in year Y = exp((Y - WEIGHT_BASE_YEAR) / WEIGHT_HALFLIFE)
# normalised so the mean weight = 1.0 (no change to effective sample size).
WEIGHT_BASE_YEAR  = 2018   # rows from this year get weight ~1.0
WEIGHT_HALFLIFE   = 4      # years for weight to double  (2022 ≈ 2× of 2018)
RECENT_BOOST_YEAR = 2023   # rows from 2023+ get an extra multiplier
RECENT_BOOST_MULT = 1.5    # extra multiplier for 2023+ rows


def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """
    Exponential recency weighting normalised to mean = 1.
    2024 rows weight ~2× more than 2018 rows.
    2023+ rows get an additional RECENT_BOOST_MULT.
    """
    years   = dates.dt.year.values.astype(float)
    weights = np.exp((years - WEIGHT_BASE_YEAR) / WEIGHT_HALFLIFE)
    # Extra boost for the most recent era
    weights[years >= RECENT_BOOST_YEAR] *= RECENT_BOOST_MULT
    # Normalise so mean = 1 (keeps total loss scale the same)
    weights = weights / weights.mean()
    return weights


# ─── FEATURE COLUMN DEFINITIONS ───────────────────────────────────────────────

IPL_BAT_FEATS = [
    # ── Recent form ──────────────────────────────────────────────────────────
    'avg_runs_5', 'avg_runs_10', 'avg_runs_15',
    'std_runs_5', 'std_runs_10',
    # ── Strike rate form ─────────────────────────────────────────────────────
    'avg_sr_5', 'avg_sr_10',
    # ── Career baselines ─────────────────────────────────────────────────────
    'career_avg',
    'career_avg_decay',      # NEW: recency-weighted career average
    'career_sr',
    'matches_played',
    # ── Dimensionless form signal ─────────────────────────────────────────────
    'form_vs_career',        # NEW: avg_runs_5 / career_avg ratio
    # ── Batting position proxy ───────────────────────────────────────────────
    'innings_position',      # NEW: proxy for batting order depth
    # ── Era / trend ──────────────────────────────────────────────────────────
    'modern_era', 'era_weight',
    # ── Venue context ────────────────────────────────────────────────────────
    'venue_avg_runs', 'venue_avg_wkts',
    'venue_rolling_avg',         # NEW: recency-weighted venue average
    'recent_venue_avg_runs',     # NEW: venue avg over last 3 seasons
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    # ── Player-at-venue ──────────────────────────────────────────────────────
    'player_venue_avg_runs', 'player_venue_avg_sr',
    'player_venue_matches', 'venue_experience_weight',
    # ── Team identity ────────────────────────────────────────────────────────
    'team_encoded',
]

IPL_BOWL_FEATS = [
    # ── Recent form ──────────────────────────────────────────────────────────
    'avg_wkts_5', 'avg_wkts_10', 'avg_wkts_15', 'std_wkts_5',
    # ── Economy form ─────────────────────────────────────────────────────────
    'avg_econ_5', 'avg_econ_10',
    'std_econ_5',            # NEW: economy consistency signal
    # ── Career baselines ─────────────────────────────────────────────────────
    'career_wkt_avg',
    'career_wkt_decay',      # NEW: recency-weighted wicket average
    'career_econ',
    'bowling_matches',
    # ── Dimensionless form signal ─────────────────────────────────────────────
    'form_vs_career',        # NEW
    # ── Era / trend ──────────────────────────────────────────────────────────
    'modern_era', 'era_weight',
    # ── Venue context ────────────────────────────────────────────────────────
    'venue_avg_runs', 'venue_avg_wkts',
    'venue_rolling_avg',         # NEW
    'recent_venue_avg_runs',     # NEW
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    # ── Player-at-venue ──────────────────────────────────────────────────────
    'player_venue_avg_wkts', 'player_venue_avg_econ',
    'player_venue_matches', 'venue_experience_weight',
    # ── Team identity ────────────────────────────────────────────────────────
    'team_encoded',
]

T20_BAT_FEATS = [
    'avg_runs_5', 'avg_runs_10', 'avg_runs_20',
    'std_runs_5', 'std_runs_10',
    'avg_sr_5', 'avg_sr_10',
    'career_avg', 'career_avg_decay',
    'career_sr', 'matches_played',
    'form_vs_career',
    'innings_position',
    'modern_era', 'era_weight',
    'venue_avg_runs', 'venue_avg_wkts',
    'venue_rolling_avg', 'recent_venue_avg_runs',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    'player_venue_avg_runs', 'player_venue_avg_sr',
    'player_venue_matches', 'venue_experience_weight',
]

T20_BOWL_FEATS = [
    'avg_wkts_5', 'avg_wkts_10', 'avg_wkts_20', 'std_wkts_5',
    'avg_econ_5', 'avg_econ_10',
    'std_econ_5',
    'career_wkt_avg', 'career_wkt_decay',
    'career_econ', 'bowling_matches',
    'form_vs_career',
    'modern_era', 'era_weight',
    'venue_avg_runs', 'venue_avg_wkts',
    'venue_rolling_avg', 'recent_venue_avg_runs',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    'player_venue_avg_wkts', 'player_venue_avg_econ',
    'player_venue_matches', 'venue_experience_weight',
]

# ─── MODEL HYPERPARAMETERS ────────────────────────────────────────────────────
# Tuned vs v1:
#   n_estimators  200 → 400  : more trees = finer fit on complex interactions
#   learning_rate 0.05 → 0.03: lower rate needs more trees but generalises better
#   max_depth     4 → 4      : unchanged — deep enough for interactions
#   min_samples_leaf (new) 10: prevents overfitting on sparse player-venue cells
#   subsample     0.8 → 0.75 : slightly more regularisation
GBR_PARAMS = dict(
    n_estimators=400,
    max_depth=4,
    min_samples_leaf=10,
    learning_rate=0.03,
    subsample=0.75,
    random_state=42,
)


# ─── TRAINING HELPER ──────────────────────────────────────────────────────────

def train_quantile_trio(df, feat_cols, target, min_innings_col,
                        split_date='2025-01-01', min_innings=5,
                        use_sample_weights=True):
    """
    Fit three quantile GBR models (Q25, Q50, Q75).

    Changes vs v1:
      - Default split_date moved to 2025-01-01 (test on most recent season)
      - Sample weights applied during fit (recent rows weighted more)
      - Only features present in the CSV are used (graceful missing-feature handling)
      - Per-year MAE breakdown printed so you can diagnose recency accuracy
    """
    # Only use features that actually exist in this CSV
    # (handles the case where 02_ was run with an older version)
    available = [c for c in feat_cols if c in df.columns]
    missing   = [c for c in feat_cols if c not in df.columns]
    if missing:
        print(f"  ⚠ Features not in CSV (skipped): {missing}")
    feat_cols = available

    clean = df.copy()
    clean[feat_cols] = clean[feat_cols].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=feat_cols + [target]).copy()
    clean = clean[clean[min_innings_col] >= min_innings]

    train = clean[clean['date'] <  split_date]
    test  = clean[clean['date'] >= split_date]

    if train.empty or test.empty:
        raise ValueError(
            f'Not enough data for target={target}. '
            f'train={len(train)}, test={len(test)}'
        )

    # Sample weights for training rows
    if use_sample_weights:
        sw = compute_sample_weights(train['date'])
    else:
        sw = None

    models = {}
    for alpha in [0.25, 0.50, 0.75]:
        m = GradientBoostingRegressor(loss='quantile', alpha=alpha, **GBR_PARAMS)
        m.fit(train[feat_cols], train[target], sample_weight=sw)
        models[alpha] = m

    preds  = {a: np.clip(models[a].predict(test[feat_cols]), 0, None)
              for a in [0.25, 0.50, 0.75]}
    actual = test[target].values

    mae      = mean_absolute_error(actual, preds[0.50])
    rmse     = np.sqrt(np.mean((actual - preds[0.50]) ** 2))
    coverage = ((actual >= preds[0.25]) & (actual <= preds[0.75])).mean()
    width    = (preds[0.75] - preds[0.25]).mean()

    pl_lo = np.mean(np.where(actual >= preds[0.25],
                             0.25 * (actual - preds[0.25]),
                             0.75 * (preds[0.25] - actual)))
    pl_hi = np.mean(np.where(actual <= preds[0.75],
                             0.75 * (preds[0.75] - actual),
                             0.25 * (actual - preds[0.75])))

    print(f"  Train:{len(train):>6} | Test:{len(test):>6} | "
          f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | "
          f"Cov:{coverage * 100:.1f}% | Width:{width:.2f} | "
          f"Pinball(lo/hi):{pl_lo:.3f}/{pl_hi:.3f}")

    # Per-year MAE breakdown on test set
    test_copy = test.copy()
    test_copy['pred_mid'] = preds[0.50]
    test_copy['abs_err']  = (test_copy['pred_mid'] - test_copy[target]).abs()
    by_year = test_copy.groupby(test_copy['date'].dt.year)['abs_err'].mean()
    year_str = '  '.join(f"{yr}:{v:.2f}" for yr, v in by_year.items())
    print(f"  MAE by year → {year_str}")

    fi = dict(zip(feat_cols, models[0.50].feature_importances_))
    return models, fi, test, preds


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    print("=== IPL BATTING ===")
    ipl_bat = pd.read_csv('output/ipl_bat_features.csv', parse_dates=['date'])
    ipl_bat_m, ipl_bat_fi, _, _ = train_quantile_trio(
        ipl_bat, IPL_BAT_FEATS, 'runs', 'matches_played')

    print("\n=== IPL BOWLING ===")
    ipl_bowl = pd.read_csv('output/ipl_bowl_features.csv', parse_dates=['date'])
    ipl_bowl['matches_played'] = ipl_bowl['bowling_matches']
    ipl_bowl_m, ipl_bowl_fi, _, _ = train_quantile_trio(
        ipl_bowl, IPL_BOWL_FEATS, 'wickets', 'matches_played')

    print("\n=== T20I BATTING ===")
    t20_bat = pd.read_csv('output/t20_bat_features.csv', parse_dates=['date'])
    t20_bat_m, t20_bat_fi, _, _ = train_quantile_trio(
        t20_bat, T20_BAT_FEATS, 'runs', 'matches_played')

    print("\n=== T20I BOWLING ===")
    t20_bowl = pd.read_csv('output/t20_bowl_features.csv', parse_dates=['date'])
    t20_bowl['matches_played'] = t20_bowl['bowling_matches']
    t20_bowl_m, t20_bowl_fi, _, _ = train_quantile_trio(
        t20_bowl, T20_BOWL_FEATS, 'wickets', 'matches_played')

    # Save models (feature lists stored inside pkl so inference always matches)
    with open('models/ipl_models.pkl', 'wb') as f:
        pickle.dump({
            'bat':        ipl_bat_m,
            'bowl':       ipl_bowl_m,
            'bat_feats':  [c for c in IPL_BAT_FEATS  if c in ipl_bat.columns],
            'bowl_feats': [c for c in IPL_BOWL_FEATS if c in ipl_bowl.columns],
            'bat_fi':     ipl_bat_fi,
            'bowl_fi':    ipl_bowl_fi,
        }, f)

    with open('models/t20_models.pkl', 'wb') as f:
        pickle.dump({
            'bat':        t20_bat_m,
            'bowl':       t20_bowl_m,
            'bat_feats':  [c for c in T20_BAT_FEATS  if c in t20_bat.columns],
            'bowl_feats': [c for c in T20_BOWL_FEATS if c in t20_bowl.columns],
            'bat_fi':     t20_bat_fi,
            'bowl_fi':    t20_bowl_fi,
        }, f)

    print("\nModels saved → models/ipl_models.pkl  models/t20_models.pkl")

    print("\nTop 10 batting features (IPL):")
    for feat, imp in sorted(ipl_bat_fi.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:<35s}: {imp * 100:.1f}%")

    print("\nTop 10 bowling features (IPL):")
    for feat, imp in sorted(ipl_bowl_fi.items(), key=lambda x: -x[1])[:10]:
        print(f"  {feat:<35s}: {imp * 100:.1f}%")


if __name__ == '__main__':
    main()
