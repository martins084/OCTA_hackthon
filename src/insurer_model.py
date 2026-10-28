"""Insurer side: MTPL claim-frequency model (freMTPL2, the French equivalent of OCTA).

Baseline = what an insurer prices with today (driver age, bonus-malus, vehicle, region...).
The telematics factor from our driving score multiplies this baseline.

Holdout: 20% of policies, split once with a fixed seed, never used for fitting or tuning.
Metrics: Poisson deviance (lower is better), exposure-weighted normalised Gini (higher is better).
Usage: python src/insurer_model.py
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.metrics import mean_poisson_deviance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/raw/fremtpl2/freMTPL2freq.csv"
MODEL_DIR = ROOT / "models/insurer_gbm"
CAT = ["Area", "VehBrand", "VehGas", "Region"]
NUM = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "LogDensity"]


def load() -> pd.DataFrame:
    d = pd.read_csv(DATA)
    d["ClaimNb"] = d.ClaimNb.clip(upper=4)          # standard cleaning for this dataset
    d["Exposure"] = d.Exposure.clip(upper=1)
    d["LogDensity"] = np.log(d.Density)
    for c in CAT:
        d[c] = d[c].astype("category")
    d["freq"] = d.ClaimNb / d.Exposure
    return d


def gini(y, pred, w) -> float:
    """Exposure-weighted normalised Gini of claim frequency ranked by predicted risk."""
    if np.ptp(pred) == 0:          # constant prediction ranks nothing; tie order would be arbitrary
        return 0.0

    def g(score):
        o = np.argsort(score, kind="stable")
        cw = np.cumsum(w[o]) / w.sum()
        cl = np.cumsum((y * w)[o]) / (y * w).sum()
        return 1 - 2 * np.trapezoid(cl, cw)
    return g(pred) / g(y)


def main():
    d = load()
    rng = np.random.default_rng(42)
    test = rng.random(len(d)) < 0.2
    tr, te = d[~test], d[test]
    print(f"{len(tr):,} train / {len(te):,} holdout policies, claim freq {tr.ClaimNb.sum() / tr.Exposure.sum():.4f}")

    res = {}
    # 1. Poisson GLM with exposure offset (actuarial standard)
    glm_f = ("ClaimNb ~ C(Area) + C(VehGas) + C(Region) + C(VehBrand) + C(pd.cut(DrivAge, [17,21,26,31,41,51,71,101]))"
             " + C(pd.cut(VehAge, [-1,0,1,4,10,100])) + C(VehPower.clip(upper=9)) + np.log(BonusMalus) + LogDensity")
    glm = smf.glm(glm_f, tr, family=sm.families.Poisson(), offset=np.log(tr.Exposure)).fit()
    p_glm = glm.predict(te, offset=np.log(te.Exposure)) / te.Exposure
    # 2. Poisson GBM with exposure as weight on frequency target (equivalent to offset)
    params = dict(objective="poisson", n_estimators=600, learning_rate=0.03, num_leaves=31,
                  min_child_samples=200, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    # early-stopping set carved out of train, not holdout
    inner = rng.random(len(tr)) < 0.15
    gbm = lgb.LGBMRegressor(**params).fit(
        tr.loc[~inner, CAT + NUM], tr.freq[~inner], sample_weight=tr.Exposure[~inner],
        eval_set=[(tr.loc[inner, CAT + NUM], tr.freq[inner])], eval_sample_weight=[tr.Exposure[inner]],
        callbacks=[lgb.early_stopping(50, verbose=False)])
    p_gbm = gbm.predict(te[CAT + NUM])
    p_null = np.full(len(te), tr.ClaimNb.sum() / tr.Exposure.sum())
    for name, p in [("constant", p_null), ("glm", p_glm.to_numpy()), ("gbm", p_gbm)]:
        res[name] = dict(poisson_deviance=float(mean_poisson_deviance(te.freq, p, sample_weight=te.Exposure)),
                         gini=float(gini(te.freq.to_numpy(), p, te.Exposure.to_numpy())))
    print(json.dumps(res, indent=2))

    imp = pd.Series(gbm.booster_.feature_importance("gain"), CAT + NUM).sort_values(ascending=False)
    print("GBM importance:", (imp / imp.sum()).round(3).to_dict())

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gbm.booster_.save_model(MODEL_DIR / "model.txt", num_iteration=gbm.best_iteration_)
    (MODEL_DIR / "meta.json").write_text(json.dumps(dict(features=CAT + NUM, categorical=CAT,
                                                         holdout=res, best_iteration=gbm.best_iteration_), indent=2))
    (ROOT / "data/processed").mkdir(parents=True, exist_ok=True)
    (ROOT / "data/processed/insurer_metrics.json").write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
