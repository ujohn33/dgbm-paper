#!/usr/bin/env python3
"""Flexible-family and calibration study for DGBM.

    python reproduce/run_flow_calibration.py --out reproduce/flow_calibration

Addresses two gaps the benchmark tables cannot speak to:

1. **Richer families.** The main experiments use a Gaussian likelihood, so they
   cannot show what the mixture and normalizing-flow families buy. Here the same
   DGBM-LGB backend is fitted with a Gaussian, a two-component Gaussian mixture
   and a spline flow, on a target whose conditional law is genuinely bimodal.

2. **Calibration beyond NLL and CRPS.** Proper scoring rules conflate
   calibration with sharpness. We add the probability integral transform (PIT)
   and central-interval coverage, both computed from predictive samples so they
   are defined identically for every family, including the flow, which has no
   closed-form location and scale.

A well-calibrated model has uniform PIT and empirical coverage matching nominal.
A Gaussian fitted to a bimodal target is typically over-dispersed in the middle
and under-dispersed in the tails, which shows up as a U-shaped or peaked PIT
histogram even when its CRPS looks respectable.

Deliberately small: a synthetic target with known ground truth plus a few real
UCI datasets, fixed hyperparameters, no tuning. It runs in minutes.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

N_SAMPLES = 1000
NOMINAL = (0.5, 0.9)
# The flow needs rates an order of magnitude below the Gaussian's before it
# trains at all; at the Gaussian's rates early stopping returns iteration 1.
ETA_GRID = {"SplineFlow": (0.0005, 0.001, 0.005, 0.01),
            "_default": (0.005, 0.01, 0.05)}

# Tree capacity is searched, not fixed. Unregularized trees let the flexible
# families produce degenerate components; the regularized setting removes that
# without a hand-tuned exception for one family.
REG_GRID = ({},
            {"max_depth": 3, "min_data_in_leaf": 100, "lambda_l2": 10.0})


def bimodal(n=4000, seed=0):
    """y | x is a two-component Gaussian mixture; separation and weight vary with x."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 3))
    sep = 1.0 + 1.5 * (x[:, 0] + 2) / 4          # components pull apart with x0
    w = 1 / (1 + np.exp(-x[:, 1]))               # mixing weight varies with x1
    pick = rng.random(n) < w
    mu = np.where(pick, sep, -sep) + 0.3 * x[:, 2]
    y = rng.normal(mu, 0.35)
    return pd.DataFrame(x, columns=[f"x{i}" for i in range(3)]), pd.Series(y, name="y")


def uci(name):
    import openpyxl  # noqa: F401  (xlsx loaders)
    loaders = {
        "Boston Housing": lambda: pd.read_csv(
            REPO_ROOT / "reproduce/uci_cache/housing.data", header=None, sep=r"\s+"),
        "Concrete": lambda: pd.read_excel(REPO_ROOT / "reproduce/uci_cache/Concrete_Data.xls"),
        "Energy Efficiency": lambda: pd.read_excel(
            REPO_ROOT / "reproduce/uci_cache/ENB2012_data.xlsx").iloc[:, :-1],
        "Yacht Hydrodynamics": lambda: pd.read_csv(
            REPO_ROOT / "reproduce/uci_cache/yacht_hydrodynamics.data", header=None, sep=r"\s+"),
    }
    d = loaders[name]().dropna()
    return d.iloc[:, :-1].reset_index(drop=True), d.iloc[:, -1].reset_index(drop=True)


def metrics(samples: np.ndarray, y: np.ndarray) -> dict:
    """samples: (n_obs, n_samples). Everything here is sample-based."""
    from utils.metrics import crps
    y = np.asarray(y, dtype=float).ravel()
    crps_val = crps(y, samples)[0]

    # PIT: fraction of predictive samples at or below the observation, with a
    # random tie-break so a discrete sample set does not bias the transform.
    rng = np.random.default_rng(0)
    below = (samples < y[:, None]).mean(axis=1)
    equal = (samples == y[:, None]).mean(axis=1)
    pit = below + rng.random(len(y)) * equal

    # deviation of the PIT histogram from uniform (0 = perfectly calibrated)
    hist, _ = np.histogram(pit, bins=10, range=(0, 1))
    pit_dev = float(np.abs(hist / len(pit) - 0.1).sum())

    out = {"CRPS": float(crps_val), "PIT_deviation": pit_dev,
           "PIT_hist": (hist / len(pit)).round(4).tolist()}
    for lvl in NOMINAL:
        lo = np.quantile(samples, (1 - lvl) / 2, axis=1)
        hi = np.quantile(samples, 1 - (1 - lvl) / 2, axis=1)
        out[f"coverage_{int(lvl*100)}"] = float(((y >= lo) & (y <= hi)).mean())
        out[f"width_{int(lvl*100)}"] = float(np.mean(hi - lo))
    return out


def families():
    """Each family maps to the constructors the validation search may choose from.

    The mixture is offered with both an exponential and a softplus response for
    its scales. With ``exp`` a low-weight component can run away to extreme
    values -- costing almost nothing in likelihood, since a low-weight component
    contributes little density, while corrupting every sample-based metric.
    Softplus grows linearly and cannot escape as fast. Which one is used is
    decided on validation data, not by hand.
    """
    from lightgbmlss.distributions.Gaussian import Gaussian
    from lightgbmlss.distributions.Mixture import Mixture
    from lightgbmlss.distributions.SplineFlow import SplineFlow
    return {
        "Gaussian": [
            lambda: Gaussian(stabilization="None", response_fn="exp", loss_fn="nll"),
        ],
        "Mixture(2)": [
            lambda: Mixture(Gaussian(stabilization="None", response_fn="exp",
                                     loss_fn="nll"), M=2),
            lambda: Mixture(Gaussian(stabilization="None", response_fn="softplus",
                                     loss_fn="nll"), M=2),
        ],
        "SplineFlow": [
            lambda: SplineFlow(target_support="real", count_bins=8, bound=3.0,
                               order="linear", stabilization="None", loss_fn="nll"),
        ],
    }


def _normalizes(dist) -> bool:
    """Does this family's ``metric_fn`` already divide the loss by n_obs?

    LightGBMLSS is not consistent about this. ``mixture_distribution_utils`` and
    ``flow_utils`` return ``loss / n_obs``; the plain ``distribution_utils`` used
    by the Gaussian returns the undivided sum. Comparing the two straight out of
    ``metric_fn`` therefore scales the Gaussian by the number of observations --
    which is exactly the factor by which its NLL first came out wrong here.
    """
    from lightgbmlss.distributions.flow_utils import NormalizingFlowClass
    from lightgbmlss.distributions.mixture_distribution_utils import (
        MixtureDistributionClass)
    return isinstance(dist, (NormalizingFlowClass, MixtureDistributionClass))


def raw_metric(model, X, y_z) -> float:
    """The loss exactly as ``metric_fn`` reports it, on the family's own scale.

    Two conventions have to be matched for ``metric_fn`` to see what it sees
    during training, and getting either wrong yields plausible-looking nonsense:

    * ``booster.predict(raw_score=True)`` returns a C-ordered ``(n, n_param)``
      array, but ``get_params_loss`` re-reads its input with ``order="F"``.
      Passing the array through unflattened interleaves the parameters across
      observations.
    * ``predict`` omits the init score, whereas the raw predictions handed to
      ``feval`` during training already include it, so the start values have to
      be added back.
    """
    import lightgbm as lgb
    ds = lgb.Dataset(X, y_z, free_raw_data=False)
    ds.construct()
    model.set_init_score(ds)
    raw = np.asarray(model.booster.predict(X, raw_score=True), dtype=float)
    raw = raw.reshape(-1, model.dist.n_dist_param)
    raw = raw + np.asarray(model.start_values, dtype=float).reshape(1, -1)
    _, loss, _ = model.dist.metric_fn(raw.flatten(order="F"), ds)
    return float(loss)


def test_nll(model, Xte, yte_z, log_scale) -> float:
    """Held-out NLL per observation, in the target's own units.

    Once put on a common per-observation scale, the likelihood is directly
    comparable across families. It is the metric that sees distributional
    *shape*: CRPS is comparatively insensitive to multimodality, because a
    unimodal fit with roughly the right spread can score well on it while
    placing density where there is none.

    The models are trained on a standardized target, so the density carries a
    constant Jacobian term; adding log(sd) puts the value back on the original
    scale. It is identical for every family, so rankings are unaffected.
    """
    value = raw_metric(model, Xte, yte_z)
    if not _normalizes(model.dist):
        value /= len(yte_z)
    return value + log_scale


def run(X, y, rounds, seed=123, dump=None):
    import lightgbm as lgb
    from lightgbmlss.model import LightGBMLSS

    n = len(X)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_te = int(0.1 * n)
    n_va = int(0.2 * (n - n_te))
    te, va, tr = perm[:n_te], perm[n_te:n_te + n_va], perm[n_te + n_va:]
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xva, yva = X.iloc[va], y.iloc[va]
    Xte, yte = X.iloc[te], y.iloc[te]

    # Standardize the target on training statistics. The spline flow's bounding
    # box [-K,K] has to match the range of the data, so a flow fitted to raw
    # targets on a scale like Boston's (5-50) with the default K=3 is
    # misspecified before it sees a single tree. Standardizing makes one K
    # correct for every dataset. It is a linear map, so PIT and coverage are
    # unaffected; predictive samples are mapped back before scoring, so CRPS and
    # interval widths stay in the target's own units.
    # ``.to_numpy()`` matters: the folds leave a permuted index on the Series,
    # and calculate_start_values indexes the label positionally.
    m, s_y = float(ytr.mean()), float(ytr.std())
    ytr_z = ((ytr - m) / s_y).to_numpy()
    yva_z = ((yva - m) / s_y).to_numpy()

    rows = {}
    dumped = {}
    for name, ctors in families().items():
        # Each family searches its own grid of (parameterisation, learning rate,
        # tree capacity), selected on validation loss. The grids differ because
        # the failure modes differ -- the flow needs far smaller learning rates
        # before it trains at all, and the flexible families need the option of
        # regularized trees to avoid degenerate components -- but nothing here
        # is chosen by looking at test results.
        best = None
        for ci, ctor in enumerate(ctors):
            for eta in ETA_GRID.get(name, ETA_GRID["_default"]):
                for ri, reg in enumerate(REG_GRID):
                    model = LightGBMLSS(ctor())
                    # No start_values override: LightGBMLSS fits the marginal
                    # MLE itself. Forcing the constant 0.5 used by the benchmark
                    # protocol leaves the many-parameter families unable to
                    # reach the target's location within the boosting budget.
                    params = {"eta": eta, "max_depth": 4, "num_leaves": 31,
                              "min_data_in_leaf": 20, "verbose": -1, "seed": seed,
                              "feature_pre_filter": False}
                    params.update(reg)
                    train_set = lgb.Dataset(Xtr, ytr_z)
                    # Early stopping on held-out likelihood. Without it the
                    # flexible families run away on the small datasets: the extra
                    # parameters keep buying training likelihood long after they
                    # stop generalising, which shows up as predictive intervals
                    # orders of magnitude too wide.
                    try:
                        model.train(params, train_set, num_boost_round=rounds,
                                    valid_sets=[lgb.Dataset(Xva, yva_z,
                                                            reference=train_set)],
                                    valid_names=["valid"],
                                    callbacks=[lgb.early_stopping(100, verbose=False)])
                    except Exception as exc:                # noqa: BLE001
                        print(f"    {name} c{ci} eta={eta} r{ri}: training failed "
                              f"({type(exc).__name__})")
                        continue
                    score = min((v for d in model.booster.best_score.values()
                                 for v in d.values()), default=float("inf"))
                    if not np.isfinite(score):
                        continue
                    if best is None or score < best[0]:
                        best = (score, eta, model, ci, ri)
        if best is None:
            print(f"    {name}: no usable fit anywhere on the grid")
            continue
        score, eta, model, ci, ri = best
        s = np.asarray(model.predict(Xte, pred_type="samples",
                                     n_samples=N_SAMPLES, seed=seed), dtype=float)
        row = metrics(s * s_y + m, yte.to_numpy())
        row["best_iteration"] = int(getattr(model.booster, "best_iteration", 0) or rounds)
        row["eta"] = eta
        row["variant"] = ci
        row["regularized"] = bool(ri)
        row["valid_loss"] = float(score)
        # Self-check: recomputing the loss on the validation set must reproduce
        # the score LightGBM recorded, otherwise the test NLL is not measuring
        # what we think it is.
        check = raw_metric(model, Xva, yva_z)
        if not np.isclose(check, score, rtol=1e-3, atol=1e-3):
            print(f"    WARNING {name}: recomputed valid loss {check:.4f} != "
                  f"recorded {score:.4f}")
        yte_z = ((yte - m) / s_y).to_numpy()
        row["NLL"] = test_nll(model, Xte, yte_z, np.log(s_y))
        # Second, independent check for the one family whose density scipy can
        # evaluate directly. The normalisation above is the kind of thing that
        # fails silently and produces numbers that merely look odd, so it is
        # worth verifying against a route that shares no code with LightGBMLSS.
        if name == "Gaussian":
            from scipy.stats import norm
            p = model.predict(Xte, pred_type="parameters")
            ref = float(-norm.logpdf(yte_z, loc=p["loc"].to_numpy(dtype=float),
                                     scale=p["scale"].to_numpy(dtype=float)).mean())
            ref += np.log(s_y)
            if not np.isclose(row["NLL"], ref, rtol=1e-2, atol=1e-2):
                print(f"    WARNING Gaussian NLL {row['NLL']:.4f} != scipy {ref:.4f}")
        rows[name] = row
        if dump is not None:
            dumped[name] = s * s_y + m

    if dump is not None:
        np.savez_compressed(dump, X_test=Xte.to_numpy(dtype=float),
                            y_test=yte.to_numpy(dtype=float),
                            **{f"samples_{k}": v for k, v in dumped.items()})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO_ROOT / "reproduce" / "flow_calibration")
    ap.add_argument("--rounds", type=int, default=300)
    ap.add_argument("--quick", action="store_true", help="synthetic only, few rounds")
    ap.add_argument("--seed", type=int, default=123,
                    help="drives the synthetic draw and the train/valid/test split; "
                         "results are written per seed so repeats can be pooled")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    datasets = {"Synthetic bimodal": bimodal(seed=args.seed)}
    if not args.quick:
        for nm in ("Boston Housing", "Concrete", "Energy Efficiency", "Yacht Hydrodynamics"):
            try:
                datasets[nm] = uci(nm)
            except Exception as exc:                        # noqa: BLE001
                print(f"  skipping {nm}: {type(exc).__name__}: {exc}")

    rounds = 40 if args.quick else args.rounds
    all_rows = {}
    for ds, (X, y) in datasets.items():
        print(f"\n=== {ds}  (n={len(X)}, p={X.shape[1]}) ===")
        # The synthetic case is the one the figure is drawn from: its true
        # conditional density is known in closed form, so fitted predictive
        # densities can be shown against the truth rather than against a KDE.
        res = run(X, y, rounds, seed=args.seed,
                  dump=(args.out / f"synthetic_samples_seed{args.seed}.npz"
                        if ds == "Synthetic bimodal" else None))
        all_rows[ds] = res
        print(f"  {'family':<12}{'NLL':>9}{'CRPS':>9}{'PIT dev':>10}{'cov50':>8}{'cov90':>8}"
              f"{'width90':>10}{'eta':>8}{'iters':>8}")
        for fam, m in res.items():
            print(f"  {fam:<12}{m['NLL']:>9.4f}{m['CRPS']:>9.4f}{m['PIT_deviation']:>10.3f}"
                  f"{m['coverage_50']:>8.3f}{m['coverage_90']:>8.3f}{m['width_90']:>10.3f}"
                  f"{m['eta']:>8.3f}{m['best_iteration']:>8d}")

    path = args.out / f"flow_calibration_seed{args.seed}.json"
    path.write_text(json.dumps(all_rows, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
