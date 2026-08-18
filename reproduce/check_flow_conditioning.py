#!/usr/bin/env python3
"""Does each family actually condition on the covariates?

    python reproduce/check_flow_conditioning.py

Source of the conditioning numbers quoted in the supplementary material
(Section on flexible families): the standard deviation across test points of
each family's predictive mean, against the true conditional mean's, and their
correlation. A family that fits only the marginal shows near-zero spread and
low correlation even when its NLL beats the Gaussian's -- which is precisely
the spline flow's behavior on the bimodal target.

Reads the per-seed sample dump written by run_flow_calibration.py.
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path,
                    default=REPO_ROOT / "reproduce" / "flow_calibration"
                    / "synthetic_samples_seed0.npz")
    args = ap.parse_args()

    d = np.load(args.dump)
    X = d["X_test"]
    sep = 1.0 + 1.5 * (X[:, 0] + 2) / 4
    w = 1 / (1 + np.exp(-X[:, 1]))
    true_mean = w * (sep + 0.3 * X[:, 2]) + (1 - w) * (-sep + 0.3 * X[:, 2])
    print(f"true conditional mean: sd across test points = {true_mean.std():.3f}")
    for key in d:
        if not key.startswith("samples_"):
            continue
        pm = d[key].mean(axis=1)
        print(f"  {key[8:]:<12} predictive-mean sd {pm.std():.3f}   "
              f"corr with truth {np.corrcoef(pm, true_mean)[0, 1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
