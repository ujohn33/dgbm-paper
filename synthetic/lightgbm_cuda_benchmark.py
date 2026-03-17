import argparse
import csv
import json
import os
import random
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[1]


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark regular LightGBM CPU vs CUDA on synthetic "
            "Gaussian data."
        )
    )
    parser.add_argument(
        "--data-sizes",
        type=int,
        nargs="+",
        required=True,
        help="Synthetic sample sizes to benchmark.",
    )
    parser.add_argument("--n-features", type=int, default=50)
    parser.add_argument("--n-informative", type=int, default=20)
    parser.add_argument("--train-fraction", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=200)
    parser.add_argument("--early-stopping-rounds", type=int, default=30)
    parser.add_argument("--num-leaves", type=int, default=63)
    parser.add_argument("--max-depth", type=int, default=-1)
    parser.add_argument("--min-data-in-leaf", type=int, default=50)
    parser.add_argument("--feature-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-fraction", type=float, default=0.9)
    parser.add_argument("--bagging-freq", type=int, default=1)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--lambda-l1", type=float, default=0.0)
    parser.add_argument("--lambda-l2", type=float, default=0.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--base-seed", type=int, default=123)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument(
        "--devices",
        nargs="+",
        default=["cpu", "cuda"],
        choices=["cpu", "cuda", "gpu"],
    )
    parser.add_argument(
        "--time-prediction",
        action="store_true",
        help="Also time prediction on the validation set.",
    )
    parser.add_argument(
        "--warmup",
        action="store_true",
        help="Run a short warmup fit per device before timed benchmarks.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help=(
            "CSV path for benchmark output. If omitted, a timestamped "
            "file is created."
        ),
    )
    return parser.parse_args()


def make_synthetic_gaussian_data(
    n_samples: int,
    n_features: int,
    n_informative: int,
    seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n_samples, n_features)).astype(np.float32)

    beta = np.zeros(n_features, dtype=np.float32)
    beta[:n_informative] = rng.normal(loc=0.0, scale=1.0, size=n_informative)

    mu = X @ beta
    mu = mu / max(1.0, np.sqrt(float(n_informative)))

    raw_scale = 0.25 * X[:, 0] - 0.15 * X[:, 1] + 0.10 * X[:, 2]
    sigma = np.exp(raw_scale).astype(np.float32) + 0.1
    y = mu + sigma * rng.normal(size=n_samples).astype(np.float32)

    columns = [f"x{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=columns), y.astype(np.float32)


def build_lgb_params(args: argparse.Namespace, device: str, seed: int) -> dict:
    return {
        "objective": "regression",
        "metric": "l2",
        "learning_rate": args.eta,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "feature_fraction": args.feature_fraction,
        "bagging_fraction": args.bagging_fraction,
        "bagging_freq": args.bagging_freq,
        "feature_pre_filter": False,
        "lambda_l1": args.lambda_l1,
        "lambda_l2": args.lambda_l2,
        "device": device,
        "seed": seed,
        "bagging_seed": seed,
        "feature_fraction_seed": seed,
        "data_random_seed": seed,
        "num_threads": args.num_threads,
        "verbosity": -1,
    }


def probe_lightgbm_device(device: str) -> tuple[bool, str]:
    X = np.random.default_rng(0).normal(size=(512, 8)).astype(np.float32)
    y = np.random.default_rng(1).normal(size=512).astype(np.float32)
    train_set = lgb.Dataset(X, label=y)
    params = {
        "objective": "regression",
        "metric": "l2",
        "verbosity": -1,
        "num_leaves": 15,
        "learning_rate": 0.1,
        "device": device,
    }
    try:
        lgb.train(params, train_set, num_boost_round=5)
    except Exception as exc:
        return False, str(exc)
    return True, "ok"


def maybe_warmup(device: str, args: argparse.Namespace) -> None:
    X, y = make_synthetic_gaussian_data(
        n_samples=2048,
        n_features=min(args.n_features, 16),
        n_informative=min(args.n_informative, 8),
        seed=args.base_seed,
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        train_size=args.train_fraction,
        random_state=args.base_seed,
    )

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_set,
        free_raw_data=False,
    )
    params = build_lgb_params(args, device=device, seed=args.base_seed)

    lgb.train(
        params,
        train_set,
        num_boost_round=10,
        valid_sets=[valid_set],
        valid_names=["valid"],
        callbacks=[lgb.early_stopping(5, verbose=False)],
    )


def benchmark_device(
    device: str,
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> dict:
    result = {
        "device": device,
        "status": "ok",
        "probe_status": "unknown",
        "probe_message": "",
        "start_value_seconds": None,
        "train_seconds": None,
        "predict_seconds": None,
        "best_iteration": None,
        "error": "",
    }

    is_supported, probe_message = probe_lightgbm_device(device)
    result["probe_status"] = "ok" if is_supported else "failed"
    result["probe_message"] = probe_message
    if not is_supported:
        result["status"] = "skipped"
        result["error"] = probe_message
        return result

    train_set = lgb.Dataset(X_train, label=y_train, free_raw_data=False)
    valid_set = lgb.Dataset(
        X_valid,
        label=y_valid,
        reference=train_set,
        free_raw_data=False,
    )
    params = build_lgb_params(args, device=device, seed=seed)

    try:
        train_begin = time.perf_counter()
        booster = lgb.train(
            params,
            train_set,
            num_boost_round=args.num_boost_round,
            valid_sets=[valid_set],
            valid_names=["valid"],
            callbacks=[
                lgb.early_stopping(
                    args.early_stopping_rounds,
                    verbose=False,
                )
            ],
        )
        result["train_seconds"] = time.perf_counter() - train_begin
        result["best_iteration"] = booster.best_iteration

        if args.time_prediction:
            predict_begin = time.perf_counter()
            pred = booster.predict(
                X_valid,
                num_iteration=booster.best_iteration,
            )
            result["predict_seconds"] = time.perf_counter() - predict_begin
            result["prediction_shape"] = str(pred.shape)
    except Exception as exc:
        result["status"] = "failed"
        result["error"] = str(exc)

    return result


def build_output_path(args: argparse.Namespace) -> Path:
    if args.output_path is not None:
        return args.output_path

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    default_dir = (
        REPO_ROOT / "results" / "synthetic" / "lightgbm_cuda_benchmark"
    )
    return default_dir / f"benchmark_{timestamp}.csv"


def write_results(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seed_everything(args.base_seed)

    output_path = build_output_path(args)
    env_info = {
        "lightgbm_version": lgb.__version__,
        "devices_requested": args.devices,
        "output_path": str(output_path),
    }
    print(json.dumps(env_info, indent=2))

    if args.warmup:
        for device in args.devices:
            try:
                maybe_warmup(device, args)
                print(f"[warmup] device={device} status=ok")
            except Exception as exc:
                print(f"[warmup] device={device} status=failed error={exc}")

    rows: list[dict] = []
    for repeat_idx in range(args.repeats):
        repeat_seed = args.base_seed + repeat_idx
        for n_samples in args.data_sizes:
            X, y = make_synthetic_gaussian_data(
                n_samples=n_samples,
                n_features=args.n_features,
                n_informative=min(args.n_informative, args.n_features),
                seed=repeat_seed,
            )
            X_train, X_valid, y_train, y_valid = train_test_split(
                X,
                y,
                train_size=args.train_fraction,
                random_state=repeat_seed,
            )

            device_results = {}
            for device in args.devices:
                device_results[device] = benchmark_device(
                    device=device,
                    X_train=X_train,
                    X_valid=X_valid,
                    y_train=y_train,
                    y_valid=y_valid,
                    args=args,
                    seed=repeat_seed,
                )

            cpu_train = device_results.get("cpu", {}).get("train_seconds")
            cuda_train = device_results.get("cuda", {}).get("train_seconds")
            speedup = None
            if cpu_train and cuda_train:
                speedup = cpu_train / cuda_train if cuda_train > 0 else None

            for device, result in device_results.items():
                row = {
                    "n_samples": n_samples,
                    "n_features": args.n_features,
                    "n_informative": min(args.n_informative, args.n_features),
                    "train_fraction": args.train_fraction,
                    "num_boost_round": args.num_boost_round,
                    "early_stopping_rounds": args.early_stopping_rounds,
                    "repeat_idx": repeat_idx,
                    "seed": repeat_seed,
                    "device": device,
                    "status": result["status"],
                    "probe_status": result["probe_status"],
                    "probe_message": result["probe_message"],
                    "start_value_seconds": result["start_value_seconds"],
                    "train_seconds": result["train_seconds"],
                    "predict_seconds": result["predict_seconds"],
                    "best_iteration": result["best_iteration"],
                    "speedup_vs_cpu": (
                        speedup
                        if device == "cuda"
                        else 1.0 if cpu_train else None
                    ),
                    "natural_grad": None,
                    "response_fn": None,
                    "stabilization": None,
                    "loss_fn": "l2",
                    "num_threads": args.num_threads,
                    "lightgbm_version": lgb.__version__,
                    "torch_version": None,
                    "torch_cuda_available": None,
                    "error": result["error"],
                }
                rows.append(row)
                print(json.dumps(row, indent=2, default=str))

    if not rows:
        raise RuntimeError("No benchmark rows were produced.")

    write_results(rows, output_path)
    print(f"Wrote benchmark results to {output_path}")


if __name__ == "__main__":
    main()
