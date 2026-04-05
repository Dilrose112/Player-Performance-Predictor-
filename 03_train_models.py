"""
03_train_models.py
------------------
Trains quantile Gradient Boosting models for IPL and T20I batting/bowling.

Train/test split : temporal (pre-2024 → train, 2024+ → test)
Evaluation       : MAE on median, interval coverage, and pinball loss

All contextual features (pitch, dew, era, player-at-venue) are defined here
in the feature column lists, which are also stored inside the .pkl so the
Flask app and predict script always use exactly the same schema.
"""
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')

# ─── FEATURE COLUMN DEFINITIONS ──────────────────────────────────────────────
# These lists are the single source of truth for what enters each model.
# Any change here must be reflected in the feature builders (02_) and the
# inference helpers (05_, 06_app.py).

IPL_BAT_FEATS = [
    # Recent form
    'avg_runs_5', 'avg_runs_10', 'avg_runs_15',
    'std_runs_5', 'std_runs_10',
    # Career baselines
    'career_avg', 'avg_sr_5', 'avg_sr_10', 'career_sr',
    'matches_played',
    # Era / trend signals — let the model learn scoring evolution
    'modern_era', 'era_weight',
    # Venue / pitch context
    'venue_avg_runs', 'venue_avg_wkts',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    # Player-at-venue form (falls back to career avg when sample < 5)
    'player_venue_avg_runs', 'player_venue_avg_sr',
    'player_venue_matches', 'venue_experience_weight',
    # Team identity (IPL-only; team composition affects strategy)
    'team_encoded',
]

IPL_BOWL_FEATS = [
    # Recent form
    'avg_wkts_5', 'avg_wkts_10', 'avg_wkts_15', 'std_wkts_5',
    # Career baselines
    'career_wkt_avg', 'career_econ', 'avg_econ_5', 'avg_econ_10',
    'bowling_matches',
    # Era / trend signals
    'modern_era', 'era_weight',
    # Venue / pitch context
    'venue_avg_runs', 'venue_avg_wkts',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    # Player-at-venue form
    'player_venue_avg_wkts', 'player_venue_avg_econ',
    'player_venue_matches', 'venue_experience_weight',
    # Team identity
    'team_encoded',
]

T20_BAT_FEATS = [
    'avg_runs_5', 'avg_runs_10', 'avg_runs_20',
    'std_runs_5', 'std_runs_10',
    'career_avg', 'avg_sr_5', 'avg_sr_10', 'career_sr', 'matches_played',
    'modern_era', 'era_weight',
    'venue_avg_runs', 'venue_avg_wkts',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    'player_venue_avg_runs', 'player_venue_avg_sr',
    'player_venue_matches', 'venue_experience_weight',
    # No team_encoded for international — squads vary too much
]

T20_BOWL_FEATS = [
    'avg_wkts_5', 'avg_wkts_10', 'avg_wkts_20', 'std_wkts_5',
    'career_wkt_avg', 'career_econ', 'avg_econ_5', 'avg_econ_10',
    'bowling_matches',
    'modern_era', 'era_weight',
    'venue_avg_runs', 'venue_avg_wkts',
    'pitch_type_encoded', 'dew_factor', 'chasing_advantage',
    'player_venue_avg_wkts', 'player_venue_avg_econ',
    'player_venue_matches', 'venue_experience_weight',
]

# ─── MODEL HYPER-PARAMETERS ──────────────────────────────────────────────────
GBR_PARAMS = dict(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    random_state=42,
)


# ─── TRAINING HELPER ─────────────────────────────────────────────────────────

def train_quantile_trio(df, feat_cols, target, min_innings_col,
                        split_date='2024-01-01', min_innings=5):
    """
    Fit three quantile GBR models (Q25, Q50, Q75) on a temporal train/test split.

    Parameters
    ----------
    df              : DataFrame with all feature columns and the target
    feat_cols       : list of feature column names (must match training schema)
    target          : column name of the prediction target ('runs' or 'wickets')
    min_innings_col : column used to filter players with too few appearances
    split_date      : cutoff; rows before → train, rows on/after → test
    min_innings     : minimum appearances required for a row to be included

    Returns
    -------
    models  : dict {0.25: model, 0.50: model, 0.75: model}
    fi      : feature importance dict from the median model
    test    : held-out test DataFrame
    preds   : dict {alpha: np.ndarray} of clipped predictions on test
    """
    clean = df.copy()
    # Replace infinities with NaN, then drop any row with a missing feature or target
    clean[feat_cols] = clean[feat_cols].replace([np.inf, -np.inf], np.nan)
    clean = clean.dropna(subset=feat_cols + [target]).copy()
    clean = clean[clean[min_innings_col] >= min_innings]

    train = clean[clean['date'] <  split_date]
    test  = clean[clean['date'] >= split_date]

    if train.empty or test.empty:
        raise ValueError(
            f'Not enough data after filtering for target={target}. '
            f'train={len(train)}, test={len(test)}'
        )

    models = {}
    for alpha in [0.25, 0.50, 0.75]:
        m = GradientBoostingRegressor(loss='quantile', alpha=alpha, **GBR_PARAMS)
        m.fit(train[feat_cols], train[target])
        models[alpha] = m

    preds  = {a: np.clip(models[a].predict(test[feat_cols]), 0, None)
              for a in [0.25, 0.50, 0.75]}
    actual = test[target].values

    mae      = mean_absolute_error(actual, preds[0.50])
    rmse     = np.sqrt(np.mean((actual - preds[0.50]) ** 2))
    coverage = ((actual >= preds[0.25]) & (actual <= preds[0.75])).mean()
    width    = (preds[0.75] - preds[0.25]).mean()

    # Pinball (quantile) loss for the two interval edges
    pl_lo = np.mean(np.where(actual >= preds[0.25],
                             0.25 * (actual - preds[0.25]),
                             0.75 * (preds[0.25] - actual)))
    pl_hi = np.mean(np.where(actual <= preds[0.75],
                             0.75 * (preds[0.75] - actual),
                             0.25 * (actual - preds[0.75])))

    fi = dict(zip(feat_cols, models[0.50].feature_importances_))

    print(f"  Train:{len(train):>6} | Test:{len(test):>6} | "
          f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | "
          f"Cov:{coverage * 100:.1f}% | Width:{width:.2f} | "
          f"Pinball(lo/hi):{pl_lo:.3f}/{pl_hi:.3f}")

    return models, fi, test, preds


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    print("=== IPL BATTING ===")
    ipl_bat = pd.read_csv('output/ipl_bat_features.csv', parse_dates=['date'])
    ipl_bat_m, ipl_bat_fi, _, _ = train_quantile_trio(
        ipl_bat, IPL_BAT_FEATS, 'runs', 'matches_played')

    print("=== IPL BOWLING ===")
    ipl_bowl = pd.read_csv('output/ipl_bowl_features.csv', parse_dates=['date'])
    ipl_bowl['matches_played'] = ipl_bowl['bowling_matches']
    ipl_bowl_m, ipl_bowl_fi, _, _ = train_quantile_trio(
        ipl_bowl, IPL_BOWL_FEATS, 'wickets', 'matches_played')

    print("=== T20I BATTING ===")
    t20_bat = pd.read_csv('output/t20_bat_features.csv', parse_dates=['date'])
    t20_bat_m, t20_bat_fi, _, _ = train_quantile_trio(
        t20_bat, T20_BAT_FEATS, 'runs', 'matches_played')

    print("=== T20I BOWLING ===")
    t20_bowl = pd.read_csv('output/t20_bowl_features.csv', parse_dates=['date'])
    t20_bowl['matches_played'] = t20_bowl['bowling_matches']
    t20_bowl_m, t20_bowl_fi, _, _ = train_quantile_trio(
        t20_bowl, T20_BOWL_FEATS, 'wickets', 'matches_played')

    # ─── SAVE ALL MODELS ─────────────────────────────────────────────────────
    # Feature lists are stored alongside models so inference code can
    # always reconstruct the correct feature vector without guessing.
    with open('models/ipl_models.pkl', 'wb') as f:
        pickle.dump({
            'bat':        ipl_bat_m,
            'bowl':       ipl_bowl_m,
            'bat_feats':  IPL_BAT_FEATS,
            'bowl_feats': IPL_BOWL_FEATS,
            'bat_fi':     ipl_bat_fi,
            'bowl_fi':    ipl_bowl_fi,
        }, f)

    with open('models/t20_models.pkl', 'wb') as f:
        pickle.dump({
            'bat':        t20_bat_m,
            'bowl':       t20_bowl_m,
            'bat_feats':  T20_BAT_FEATS,
            'bowl_feats': T20_BOWL_FEATS,
            'bat_fi':     t20_bat_fi,
            'bowl_fi':    t20_bowl_fi,
        }, f)

    print("\nAll models saved to models/ipl_models.pkl and models/t20_models.pkl")

    print("\nTop batting features (IPL):")
    for feat, imp in sorted(ipl_bat_fi.items(), key=lambda x: -x[1])[:8]:
        print(f"  {feat:<35s}: {imp * 100:.1f}%")

    print("\nTop bowling features (IPL):")
    for feat, imp in sorted(ipl_bowl_fi.items(), key=lambda x: -x[1])[:8]:
        print(f"  {feat:<35s}: {imp * 100:.1f}%")


if __name__ == '__main__':
    main()
