#!/usr/bin/env python3
"""Why do the mixture and the flow not recover the true conditional law?

Two hypotheses, each with a measurement rather than an opinion:

1. **The flow never learns x-dependence.** Early stopping selected iteration 1,
   so the fit should be the marginal: its predictive distribution would then be
   *identical* at every test point. Measured as the spread, across test points,
   of the predictive mean and of the 10th/90th percentiles. A value near zero
   confirms it, and would mean the flow's flat PIT histogram reflects a good
   *marginal* fit rather than a good conditional one.

2. **The mixture finds the modes but makes them too narrow.** Its U-shaped PIT
   says the predictive intervals are too tight. Measured by comparing predicted
   component locations, scales and weight against the known truth.

Also prints the whole learning-rate grid per family -- chosen iteration and
validation loss -- to show whether any rate trained the flow at all.
"""
from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "rfc", pathlib.Path(__file__).resolve().parent / "run_flow_calibration.py")
rfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfc)

SEED = 123


def main() -> int:
    import lightgbm as lgb
    from lightgbmlss.model import LightGBMLSS

    X, y = rfc.bimodal(seed=SEED)
    n = len(X)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_te = int(0.1 * n)
    n_va = int(0.2 * (n - n_te))
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xva, yva = X.iloc[va], y.iloc[va]
    Xte, yte = X.iloc[te], y.iloc[te]

    m, s_y = float(ytr.mean()), float(ytr.std())
    ytr_z = ((ytr - m) / s_y).to_numpy()
    yva_z = ((yva - m) / s_y).to_numpy()

    for name, make in rfc.families().items():
        print(f"\n=== {name} ===")
        best = None
        for eta in rfc.ETA_GRID:
            model = LightGBMLSS(make())
            params = {"eta": eta, "max_depth": 4, "num_leaves": 31,
                      "min_data_in_leaf": 20, "verbose": -1, "seed": SEED,
                      "feature_pre_filter": False}
            train_set = lgb.Dataset(Xtr, ytr_z)
            model.train(params, train_set, num_boost_round=2000,
                        valid_sets=[lgb.Dataset(Xva, yva_z, reference=train_set)],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
            score = min(v for d in model.booster.best_score.values() for v in d.values())
            it = int(getattr(model.booster, "best_iteration", 0) or 2000)
            print(f"  eta={eta:<6} best_iter={it:<6} valid_loss={score:.4f}")
            if best is None or score < best[0]:
                best = (score, eta, model)

        _, eta, model = best
        p = model.predict(Xte, pred_type="parameters")
        s = np.asarray(model.predict(Xte, pred_type="samples", n_samples=1000,
                                     seed=SEED), dtype=float) * s_y + m

        # Hypothesis 1: does the predictive distribution move with x at all?
        mean_x = s.mean(axis=1)
        q10, q90 = np.quantile(s, 0.10, axis=1), np.quantile(s, 0.90, axis=1)
        print(f"  chosen eta={eta}; spread ACROSS test points of the predictive")
        print(f"    mean  sd={mean_x.std():.4f}   q10 sd={q10.std():.4f}   "
              f"q90 sd={q90.std():.4f}")
        print(f"    (true conditional mean varies with sd="
              f"{true_cond_mean(Xte.to_numpy()).std():.4f})")
        print(f"  predicted parameter columns: {list(p.columns)}")

        # Hypothesis 2: for the mixture, compare components against the truth.
        if name.startswith("Mixture"):
            cols = list(p.columns)
            locs = [c for c in cols if "loc" in c]
            scales = [c for c in cols if "scale" in c]
            mixes = [c for c in cols if "mix" in c or "prob" in c]
            if len(locs) >= 2 and len(scales) >= 2:
                Xa = Xte.to_numpy()
                sep = 1.0 + 1.5 * (Xa[:, 0] + 2) / 4
                true_lo = (-sep + 0.3 * Xa[:, 2])
                true_hi = (sep + 0.3 * Xa[:, 2])
                l1 = p[locs[0]].to_numpy(dtype=float) * s_y + m
                l2 = p[locs[1]].to_numpy(dtype=float) * s_y + m
                lo_hat, hi_hat = np.minimum(l1, l2), np.maximum(l1, l2)
                sd1 = p[scales[0]].to_numpy(dtype=float) * s_y
                sd2 = p[scales[1]].to_numpy(dtype=float) * s_y
                print(f"  component location error (mean abs): "
                      f"lower {np.abs(lo_hat - true_lo).mean():.3f}   "
                      f"upper {np.abs(hi_hat - true_hi).mean():.3f}")
                print(f"  component scale: predicted median "
                      f"{np.median(np.concatenate([sd1, sd2])):.3f}  vs true 0.350")
                if mixes:
                    w_hat = p[mixes[0]].to_numpy(dtype=float)
                    w_true = 1 / (1 + np.exp(-Xa[:, 1]))
                    print(f"  mixing weight error (mean abs): "
                          f"{np.abs(np.minimum(w_hat, 1 - w_hat) - np.minimum(w_true, 1 - w_true)).mean():.3f}")
    return 0


def true_cond_mean(Xa):
    sep = 1.0 + 1.5 * (Xa[:, 0] + 2) / 4
    w = 1 / (1 + np.exp(-Xa[:, 1]))
    return w * (sep + 0.3 * Xa[:, 2]) + (1 - w) * (-sep + 0.3 * Xa[:, 2])


if __name__ == "__main__":
    sys.exit(main())
