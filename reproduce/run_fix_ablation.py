#!/usr/bin/env python3
"""Do the candidate fixes actually fix anything?

    python reproduce/run_fix_ablation.py --part flow
    python reproduce/run_fix_ablation.py --part mixture

Two failures were diagnosed in the family study, each with several plausible
remedies. Rather than adopt the plausible-sounding one, this measures them.

**Flow: never leaves the marginal fit.** Early stopping returns iteration 1 at
every learning rate, so the model has no covariate dependence. Candidates: a
much smaller learning rate; more patience (the loss may dip after an initial
rise); fewer spline parameters; the library's gradient stabilization; coarser
trees. The diagnostic is ``best_iter`` -- anything that trains will show it
climb well above 1, and the spread of the predictive mean across test points
should approach the true conditional spread.

**Mixture: a low-weight component escapes.** It costs almost nothing in
likelihood -- which is why selection on NLL does not catch it -- but wrecks
every sample-based metric. Candidates: selecting the learning rate on
validation CRPS instead of NLL (CRPS sees the escape), a softplus response for
the scale instead of exp (which cannot blow up as fast), and stronger tree
regularization. The diagnostic is how many seeds go degenerate.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
_spec = importlib.util.spec_from_file_location(
    "rfc", pathlib.Path(__file__).resolve().parent / "run_flow_calibration.py")
rfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfc)

SEEDS = (0, 1, 2)


def make_flow(count_bins=8, stab="None"):
    from lightgbmlss.distributions.SplineFlow import SplineFlow
    return lambda: SplineFlow(target_support="real", count_bins=count_bins,
                              bound=3.0, order="linear", stabilization=stab,
                              loss_fn="nll")


def make_mixture(response_fn="exp", stab="None"):
    from lightgbmlss.distributions.Gaussian import Gaussian
    from lightgbmlss.distributions.Mixture import Mixture
    return lambda: Mixture(Gaussian(stabilization=stab, response_fn=response_fn,
                                    loss_fn="nll"), M=2)


FLOW_VARIANTS = {
    "baseline":        dict(make=make_flow(), etas=(0.005, 0.01, 0.05), patience=100),
    "tiny_eta":        dict(make=make_flow(), etas=(0.0005, 0.001), patience=100),
    "patience_500":    dict(make=make_flow(), etas=(0.005, 0.01), patience=500),
    "fewer_bins":      dict(make=make_flow(count_bins=4), etas=(0.005, 0.01), patience=100),
    "stabilization_L2": dict(make=make_flow(stab="L2"), etas=(0.005, 0.01), patience=100),
    "coarse_trees":    dict(make=make_flow(), etas=(0.005, 0.01), patience=100,
                            params=dict(max_depth=2, min_data_in_leaf=200)),
}

MIX_VARIANTS = {
    "baseline":     dict(make=make_mixture(), etas=(0.005, 0.01, 0.05), select="nll"),
    "select_crps":  dict(make=make_mixture(), etas=(0.005, 0.01, 0.05), select="crps"),
    "softplus":     dict(make=make_mixture(response_fn="softplus"),
                         etas=(0.005, 0.01, 0.05), select="nll"),
    "regularized":  dict(make=make_mixture(), etas=(0.005, 0.01, 0.05), select="nll",
                         params=dict(max_depth=3, min_data_in_leaf=100,
                                     lambda_l2=10.0)),
    "crps_softplus_reg": dict(make=make_mixture(response_fn="softplus"),
                              etas=(0.005, 0.01, 0.05), select="crps",
                              params=dict(max_depth=3, min_data_in_leaf=100,
                                          lambda_l2=10.0)),
}


def fit_one(X, y, seed, make, etas, patience=100, select="nll", params=None,
            rounds=2000):
    import lightgbm as lgb
    from lightgbmlss.model import LightGBMLSS

    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = int(0.1 * n)
    n_va = int(0.2 * (n - n_te))
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    Xtr, ytr, Xva, yva, Xte, yte = (X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va],
                                    X.iloc[te], y.iloc[te])
    m, s_y = float(ytr.mean()), float(ytr.std())
    ytr_z, yva_z = ((ytr - m) / s_y).to_numpy(), ((yva - m) / s_y).to_numpy()

    best = None
    for eta in etas:
        model = LightGBMLSS(make())
        p = {"eta": eta, "max_depth": 4, "num_leaves": 31, "min_data_in_leaf": 20,
             "verbose": -1, "seed": seed, "feature_pre_filter": False}
        p.update(params or {})
        train_set = lgb.Dataset(Xtr, ytr_z)
        try:
            model.train(p, train_set, num_boost_round=rounds,
                        valid_sets=[lgb.Dataset(Xva, yva_z, reference=train_set)],
                        valid_names=["valid"],
                        callbacks=[lgb.early_stopping(patience, verbose=False)])
        except Exception as exc:                                # noqa: BLE001
            print(f"      eta={eta}: failed {type(exc).__name__}")
            continue
        if select == "nll":
            score = min(v for d in model.booster.best_score.values() for v in d.values())
        else:
            # Sample-based selection: this is what sees a component escaping,
            # because the likelihood barely notices a low-weight component.
            sv = np.asarray(model.predict(Xva, pred_type="samples", n_samples=200,
                                          seed=seed), dtype=float) * s_y + m
            score = rfc.metrics(sv, yva.to_numpy())["CRPS"]
        if not np.isfinite(score):
            continue
        if best is None or score < best[0]:
            best = (score, eta, model)
    if best is None:
        return None
    _, eta, model = best
    s = np.asarray(model.predict(Xte, pred_type="samples", n_samples=1000,
                                 seed=seed), dtype=float) * s_y + m
    out = rfc.metrics(s, yte.to_numpy())
    out["NLL"] = rfc.test_nll(model, Xte, ((yte - m) / s_y).to_numpy(), np.log(s_y))
    out["best_iteration"] = int(getattr(model.booster, "best_iteration", 0) or rounds)
    out["eta"] = eta
    out["pred_mean_sd"] = float(s.mean(axis=1).std())
    out["true_mean_sd"] = float(y.iloc[te].std())
    out["max_abs_sample"] = float(np.abs(s).max())
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=("flow", "mixture"), required=True)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO_ROOT / "reproduce" / "flow_calibration")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if args.part == "flow":
        variants, datasets = FLOW_VARIANTS, {"Synthetic bimodal": None}
    else:
        variants, datasets = MIX_VARIANTS, {"Synthetic bimodal": None, "Concrete": None}

    results = {}
    for ds in datasets:
        print(f"\n########## {ds} ##########")
        for vname, cfg in variants.items():
            rows = []
            for seed in SEEDS:
                X, y = (rfc.bimodal(seed=seed) if ds == "Synthetic bimodal"
                        else rfc.uci(ds))
                r = fit_one(X, y, seed, **{k: v for k, v in cfg.items()})
                if r is not None:
                    rows.append(r)
            if not rows:
                print(f"  {vname:<20} all seeds failed")
                continue
            g = lambda k: np.array([r[k] for r in rows], dtype=float)
            crps, nll, it = g("CRPS"), g("NLL"), g("best_iteration")
            degen = int((~np.isfinite(crps)).sum()
                        + (np.nan_to_num(crps, nan=0) > 10 * np.nanmedian(crps)).sum())
            print(f"  {vname:<20} iters {np.mean(it):>7.0f}   NLL {np.nanmean(nll):>7.3f}"
                  f"   CRPS med {np.nanmedian(crps):>9.3f}   PITdev "
                  f"{np.nanmean(g('PIT_deviation')):>5.2f}   degen {degen}/{len(rows)}"
                  f"   predSD {np.mean(g('pred_mean_sd')):.3f}"
                  f"/{np.mean(g('true_mean_sd')):.3f}")
            results[f"{ds}|{vname}"] = rows
    (args.out / f"fix_ablation_{args.part}.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote fix_ablation_{args.part}.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
