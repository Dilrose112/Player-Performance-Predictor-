"""
03_train_models.py
-------------------
Trains 4 separate Gradient Boosting Quantile Regression models:
  1. IPL  Batting  — α = 0.25, 0.50, 0.75
  2. IPL  Bowling  — α = 0.25, 0.50, 0.75
  3. T20I Batting  — α = 0.25, 0.50, 0.75
  4. T20I Bowling  — α = 0.25, 0.50, 0.75

Train/test split: temporal (pre-2024 → train, 2024+ → test)
Evaluation: MAE on median, interval coverage, pinball loss
"""
import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# ─── FEATURE COLUMN DEFINITIONS ─────────────────────────────────────────────
IPL_BAT_FEATS  = [
    'avg_runs_5','avg_runs_10','avg_runs_15',
    'std_runs_5','std_runs_10',
    'career_avg','avg_sr_5','avg_sr_10','career_sr',
    'matches_played','venue_avg_runs','team_encoded'
]
IPL_BOWL_FEATS = [
    'avg_wkts_5','avg_wkts_10','avg_wkts_15','std_wkts_5',
    'career_wkt_avg','avg_econ_5','avg_econ_10',
    'bowling_matches','venue_avg_conceded','team_encoded'
]
T20_BAT_FEATS  = [
    'avg_runs_5','avg_runs_10','avg_runs_20',
    'std_runs_5','std_runs_10',
    'career_avg','avg_sr_5','avg_sr_10','career_sr','matches_played'
]
T20_BOWL_FEATS = [
    'avg_wkts_5','avg_wkts_10','avg_wkts_20','std_wkts_5',
    'career_wkt_avg','avg_econ_5','avg_econ_10','bowling_matches'
]

GBR_PARAMS = dict(n_estimators=200, max_depth=4, learning_rate=0.05,
                  subsample=0.8, random_state=42)

def train_quantile_trio(df, feat_cols, target, min_innings_col,
                        split_date='2024-01-01', min_innings=5):
    clean = df.dropna(subset=feat_cols + [target]).copy()
    clean = clean[clean[min_innings_col] >= min_innings]
    train = clean[clean['date'] < split_date]
    test  = clean[clean['date'] >= split_date]

    models = {}
    for alpha in [0.25, 0.50, 0.75]:
        m = GradientBoostingRegressor(loss='quantile', alpha=alpha, **GBR_PARAMS)
        m.fit(train[feat_cols], train[target])
        models[alpha] = m

    preds = {a: np.clip(models[a].predict(test[feat_cols]), 0, None)
             for a in [0.25, 0.50, 0.75]}
    actual = test[target].values

    mae      = mean_absolute_error(actual, preds[0.50])
    rmse     = np.sqrt(np.mean((actual - preds[0.50])**2))
    coverage = ((actual >= preds[0.25]) & (actual <= preds[0.75])).mean()
    width    = (preds[0.75] - preds[0.25]).mean()

    # Pinball losses
    pl_lo = np.mean(np.where(actual >= preds[0.25],
                             0.25*(actual-preds[0.25]), 0.75*(preds[0.25]-actual)))
    pl_hi = np.mean(np.where(actual <= preds[0.75],
                             0.75*(preds[0.75]-actual), 0.25*(actual-preds[0.75])))

    fi = dict(zip(feat_cols, models[0.50].feature_importances_))

    print(f"  Train:{len(train):>6} | Test:{len(test):>6} | "
          f"MAE:{mae:.3f} | RMSE:{rmse:.3f} | "
          f"Cov:{coverage*100:.1f}% | Width:{width:.2f} | "
          f"Pinball(lo/hi):{pl_lo:.3f}/{pl_hi:.3f}")

    return models, fi, test, preds

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

    # ─── SAVE ALL MODELS ────────────────────────────────────────────────────
    with open('models/ipl_models.pkl', 'wb') as f:
        pickle.dump({'bat': ipl_bat_m,  'bowl': ipl_bowl_m,
                     'bat_feats': IPL_BAT_FEATS, 'bowl_feats': IPL_BOWL_FEATS,
                     'bat_fi': ipl_bat_fi, 'bowl_fi': ipl_bowl_fi}, f)

    with open('models/t20_models.pkl', 'wb') as f:
        pickle.dump({'bat': t20_bat_m,  'bowl': t20_bowl_m,
                     'bat_feats': T20_BAT_FEATS, 'bowl_feats': T20_BOWL_FEATS,
                     'bat_fi': t20_bat_fi, 'bowl_fi': t20_bowl_fi}, f)

    print("\nAll models saved to ipl_models.pkl and t20_models.pkl")
    print("\nTop batting features (IPL):")
    for f, v in sorted(ipl_bat_fi.items(), key=lambda x: -x[1])[:5]:
        print(f"  {f}: {v*100:.1f}%")

if __name__ == '__main__':
    main()
