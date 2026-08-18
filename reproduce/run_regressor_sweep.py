#!/usr/bin/env python3
"""When does a flexible family stop being worth it?

    python reproduce/run_regressor_sweep.py --out reproduce/flow_calibration

Multimodality in a *conditional* law is a statement about what the model can
see. If the variable that selects the mode is observed, the conditional
distribution given the covariates is unimodal and a Gaussian head suffices; the
apparent bimodality was only ever marginal. A flexible family earns its cost
exactly when the mode-selecting mechanism is latent.

This makes that precise by sweeping how informative the observed regressor is.
The target is a two-component mixture whose component is chosen by a latent
Bernoulli ``z``. The feature matrix carries a noisy copy ``s`` of ``z``, correct
with probability ``p``:

* ``p = 0.5`` -- ``s`` is pure noise, ``z`` is effectively latent, and the
  conditional law is genuinely bimodal;
* ``p = 1.0`` -- ``s`` reveals the component, and the conditional law is a
  single Gaussian.

Everything else -- component separation, noise, sample size -- is held fixed, so
the only thing varying is how much the regressors reveal. The prediction is that
the mixture's NLL advantage over the Gaussian decays to zero as ``p`` rises.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import sys

import numpy as np
import pandas as pd

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "rfc", pathlib.Path(__file__).resolve().parent / "run_flow_calibration.py")
rfc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rfc)

# Dense near 1.0: the advantage collapses only as the regressor becomes
# *fully* informative, so the interesting structure is in the last decile.
P_GRID = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 1.0)


def dgp(p_informative, n=4000, seed=0):
    """Two-component target; the observed indicator is correct w.p. p."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, 3))
    sep = 1.0 + 1.5 * (x[:, 0] + 2) / 4
    z = rng.random(n) < 0.5                      # latent component indicator
    # Observed indicator: equals z with probability p, flipped otherwise. At
    # p=0.5 it is independent of z and carries nothing.
    s = np.where(rng.random(n) < p_informative, z, ~z).astype(float)
    mu = np.where(z, sep, -sep) + 0.3 * x[:, 2]
    y = rng.normal(mu, 0.35)
    X = pd.DataFrame(np.column_stack([x, s]),
                     columns=["x0", "x1", "x2", "s"])
    return X, pd.Series(y, name="y")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path,
                    default=REPO_ROOT / "reproduce" / "flow_calibration")
    ap.add_argument("--rounds", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Gaussian and mixture only: the question here is whether the extra
    # component pays for itself, and the flow is a separate matter.
    keep = ("Gaussian", "Mixture(2)")
    all_families = rfc.families()
    rfc.families = lambda: {k: v for k, v in all_families.items() if k in keep}

    out = {}
    print(f"{'p':>6}{'Gauss NLL':>12}{'Mix NLL':>12}{'advantage':>12}"
          f"{'Gauss CRPS':>12}{'Mix CRPS':>12}")
    for p in P_GRID:
        X, y = dgp(p, seed=args.seed)
        res = rfc.run(X, y, args.rounds, seed=args.seed)
        adv = res["Gaussian"]["NLL"] - res["Mixture(2)"]["NLL"]
        out[f"{p:.2f}"] = {"advantage_NLL": adv, **res}
        print(f"{p:>6.1f}{res['Gaussian']['NLL']:>12.4f}"
              f"{res['Mixture(2)']['NLL']:>12.4f}{adv:>12.4f}"
              f"{res['Gaussian']['CRPS']:>12.4f}{res['Mixture(2)']['CRPS']:>12.4f}")

    path = args.out / f"regressor_sweep_seed{args.seed}.json"
    path.write_text(json.dumps(out, indent=1))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
