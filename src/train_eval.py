# src/train_eval.py
from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from src.config import load_config, make_run_dir, resolve_path, save_run_config

warnings.filterwarnings(
    "ignore",
    message=r"`sklearn\.utils\.parallel\.delayed` should be used with `sklearn\.utils\.parallel\.Parallel`.*",
    category=UserWarning,
    module=r"sklearn\.utils\.parallel",
)
warnings.filterwarnings(
    "ignore",
    message=r"The total space of parameters .* is smaller than n_iter=.*",
    category=UserWarning,
    module=r"sklearn\.model_selection\._search",
)
warnings.simplefilter("ignore", ConvergenceWarning)


# -------------------- logging --------------------
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("eurpln_train_eval")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# -------------------- split + metrics --------------------
def split_time_series(df: pd.DataFrame, val_size: int, test_size: int) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True)
    n = len(df)
    if val_size + test_size >= n:
        raise ValueError(f"Za mało danych na split: n={n}, val={val_size}, test={test_size}")
    train_end = n - (val_size + test_size)
    val_end = n - test_size
    return df.iloc[:train_end].copy(), df.iloc[train_end:val_end].copy(), df.iloc[val_end:].copy()


def mae(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_pred - y_true)))


def rmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    eps = 1e-12
    return float(np.mean(np.abs((y_pred - y_true) / (y_true + eps))))


def smape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred) + 1e-12)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom))


def directional_accuracy(y_t, y_true, y_pred) -> float:
    y_t = np.asarray(y_t, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    d_true = np.sign(y_true - y_t)
    d_pred = np.sign(y_pred - y_t)
    return float(np.mean(d_true == d_pred))


# -------------------- baselines --------------------
def sma_from_lags(df: pd.DataFrame, K: int) -> np.ndarray:
    cols = [f"lag_{i}" for i in range(K)]
    return df[cols].mean(axis=1).to_numpy(dtype=float)


def ema_from_lags(df: pd.DataFrame, K: int) -> np.ndarray:
    alpha = 2.0 / (K + 1.0)
    cols = [f"lag_{i}" for i in range(K - 1, -1, -1)]  # lag_{K-1} ... lag_0
    mat = df[cols].to_numpy(dtype=float)
    ema = mat[:, 0].copy()
    for j in range(1, mat.shape[1]):
        ema = alpha * mat[:, j] + (1.0 - alpha) * ema
    return ema


# -------------------- utils --------------------
def _get_or_create_run_dir(cfg: dict[str, Any], run: str | None) -> Path:
    if run:
        run_dir = resolve_path(run)
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    return make_run_dir(cfg["output"]["runs_dir"])


def _find_ds_csv(cfg: dict[str, Any], H: int) -> Path:
    data_dir = resolve_path(cfg["output"]["data_dir"])
    p = data_dir / f"ds_H{H}.csv"
    if not p.exists():
        raise FileNotFoundError(f"Nie znaleziono datasetu: {p}")
    return p


def _feature_cols(df: pd.DataFrame) -> List[str]:
    drop = {"date", "target_date", "y_t", "target"}
    return [c for c in df.columns if c not in drop]


def _add_eval_row(
    metrics_rows: List[Dict[str, Any]],
    H: int,
    model: str,
    split: str,
    y_t: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    row: Dict[str, Any] = {
        "H": H,
        "model": model,
        "split": split,
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
        "SMAPE": smape(y_true, y_pred),
    }
    if H == 1:
        row["DirAcc"] = directional_accuracy(y_t, y_true, y_pred)
    metrics_rows.append(row)


# -------------------- ML models + tuning --------------------
def make_models(seed: int) -> Dict[str, Any]:
    return {
        "Ridge": Pipeline([("scaler", StandardScaler()), ("model", Ridge())]),
        "ElasticNet": Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(max_iter=50000, random_state=seed))]),
        "RandomForest": RandomForestRegressor(random_state=seed, n_jobs=-1),
        "ExtraTrees": ExtraTreesRegressor(random_state=seed, n_jobs=-1),
        "HistGB": HistGradientBoostingRegressor(random_state=seed, early_stopping=False),
    }


def get_search_spaces(seed: int) -> dict[str, dict[str, list]]:
    # listy są OK dla RandomizedSearchCV (bez scipy)
    return {
        "Ridge": {
            "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0],
        },
        "ElasticNet": {
            "model__alpha": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1],
            "model__l1_ratio": [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.95],
        },
        "RandomForest": {
            "n_estimators": [400, 600, 900, 1200],
            "max_depth": [None, 4, 6, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.6, 0.8],
        },
        "ExtraTrees": {
            "n_estimators": [600, 900, 1200, 1500],
            "max_depth": [None, 6, 8, 12, 16],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2", 0.6, 0.8],
        },
        "HistGB": {
            "learning_rate": [0.01, 0.03, 0.05, 0.1],
            "max_depth": [3, 4, 6, 8, None],
            "max_iter": [200, 400, 600, 800],
            "min_samples_leaf": [10, 20, 40, 80],
            "l2_regularization": [0.0, 0.01, 0.1, 1.0],
        },
    }


def tune_with_timeseriessplit(
    cfg: dict[str, Any],
    model,
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    tuning = cfg.get("tuning", {}) or {}
    use_cv = bool(tuning.get("use_timeseries_cv", True))
    if not use_cv:
        return model, {}

    spaces = get_search_spaces(seed).get(name)
    if not spaces:
        return model, {}

    n_splits = int(tuning.get("tscv_splits", 5))
    n_iter = int(tuning.get("n_iter", 40))

    tss = TimeSeriesSplit(n_splits=n_splits)
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=spaces,
        n_iter=n_iter,
        cv=tss,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
        random_state=seed,
        refit=True,  # best_estimator_ fit na CAŁYM train
        verbose=0,
    )
    search.fit(X_train, y_train)

    info = {
        "cv_best_RMSE": float(-search.best_score_),
        "best_params": search.best_params_,
        "n_splits": n_splits,
        "n_iter": n_iter,
    }
    return search.best_estimator_, info


# -------------------- SARIMAX "fair" --------------------
def sarimax_forecast_from_lags(part: pd.DataFrame, W: int, H: int, order=(1, 1, 1)) -> np.ndarray:
    preds = np.empty(len(part), dtype=float)
    lag_cols = [f"lag_{i}" for i in range(W - 1, -1, -1)]
    lag_mat = part[lag_cols].to_numpy(dtype=float)

    for i in range(len(part)):
        y_window = lag_mat[i]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model = SARIMAX(
                    y_window,
                    order=order,
                    trend="n",
                    enforce_stationarity=False,
                    enforce_invertibility=False,
                    simple_differencing=True,
                )
                res = model.fit(disp=False, maxiter=80, method="lbfgs")

            if hasattr(res, "mle_retvals") and isinstance(res.mle_retvals, dict):
                if res.mle_retvals.get("converged") is False:
                    preds[i] = float(part.iloc[i]["y_t"])
                    continue

            fc = res.forecast(steps=H)
            preds[i] = float(fc[-1])
        except Exception:
            preds[i] = float(part.iloc[i]["y_t"])

    return preds


# -------------------- per-H evaluation --------------------
def eval_all_for_H(
    cfg: dict[str, Any],
    logger: logging.Logger,
    H: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:

    ds_path = _find_ds_csv(cfg, H)
    df = pd.read_csv(ds_path, parse_dates=["date", "target_date"])

    val_size = int(cfg["split"]["val_size"])
    test_size = int(cfg["split"]["test_size"])
    train, val, test = split_time_series(df, val_size=val_size, test_size=test_size)

    feat_cols = _feature_cols(df)

    X_train = train[feat_cols].to_numpy(dtype=float)
    y_train = train["target"].to_numpy(dtype=float)
    X_val = val[feat_cols].to_numpy(dtype=float)
    y_val = val["target"].to_numpy(dtype=float)
    X_test = test[feat_cols].to_numpy(dtype=float)
    y_test = test["target"].to_numpy(dtype=float)

    logger.info(f"H={H} | ds={ds_path.name} | train={len(train)} val={len(val)} test={len(test)} | feats={len(feat_cols)}")

    metrics_rows: list[dict[str, Any]] = []
    preds_rows: list[pd.DataFrame] = []

    # ---- Baselines: Naive + SMA/EMA (K wybieramy na val)
    candidate_K = [5, 10, 20, 60]
    best_baselines = {
        "SMA": {"K": None, "val_RMSE": float("inf")},
        "EMA": {"K": None, "val_RMSE": float("inf")},
    }

    # Naive
    for split_name, part in [("val", val), ("test", test)]:
        y_pred = part["y_t"].to_numpy(dtype=float)
        y_true = part["target"].to_numpy(dtype=float)
        y_t = part["y_t"].to_numpy(dtype=float)

        _add_eval_row(metrics_rows, H, "Naive", split_name, y_t, y_true, y_pred)
        preds_rows.append(
            pd.DataFrame(
                {
                    "date": part["date"],
                    "target_date": part["target_date"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "model": "Naive",
                    "H": H,
                    "split": split_name,
                }
            )
        )

    # SMA/EMA tuning na val
    for K in candidate_K:
        r = rmse(y_val, sma_from_lags(val, K))
        if r < best_baselines["SMA"]["val_RMSE"]:
            best_baselines["SMA"] = {"K": K, "val_RMSE": r}

    for K in candidate_K:
        r = rmse(y_val, ema_from_lags(val, K))
        if r < best_baselines["EMA"]["val_RMSE"]:
            best_baselines["EMA"] = {"K": K, "val_RMSE": r}

    for kind, fn in [("SMA", sma_from_lags), ("EMA", ema_from_lags)]:
        K = int(best_baselines[kind]["K"])
        name = f"{kind}(K={K})"
        for split_name, part in [("val", val), ("test", test)]:
            y_pred = fn(part, K)
            y_true = part["target"].to_numpy(dtype=float)
            y_t = part["y_t"].to_numpy(dtype=float)

            _add_eval_row(metrics_rows, H, name, split_name, y_t, y_true, y_pred)
            preds_rows.append(
                pd.DataFrame(
                    {
                        "date": part["date"],
                        "target_date": part["target_date"],
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "model": name,
                        "H": H,
                        "split": split_name,
                    }
                )
            )

    # ---- ML models (tuning na TRAIN przez TimeSeriesSplit)
    models = make_models(seed)
    best_ml_params: dict[str, Any] = {}

    for name, model in models.items():
        tuned_model, info = tune_with_timeseriessplit(cfg, model, name, X_train, y_train, seed=seed)
        best_ml_params[name] = info
        if info:
            logger.info(f"H={H} | {name} tuned(TSS): {info}")

        # VAL: (tuned_model jest fit na TRAIN)
        y_pred_val = tuned_model.predict(X_val)
        _add_eval_row(metrics_rows, H, name, "val", val["y_t"].to_numpy(dtype=float), y_val, y_pred_val)
        preds_rows.append(
            pd.DataFrame(
                {
                    "date": val["date"],
                    "target_date": val["target_date"],
                    "y_true": y_val,
                    "y_pred": y_pred_val,
                    "model": name,
                    "H": H,
                    "split": "val",
                }
            )
        )

        # TEST: refit na TRAIN+VAL najlepszymi parametrami
        refit_model = clone(tuned_model)
        X_dev = np.vstack([X_train, X_val])
        y_dev = np.concatenate([y_train, y_val])
        refit_model.fit(X_dev, y_dev)

        y_pred_test = refit_model.predict(X_test)
        _add_eval_row(metrics_rows, H, name, "test", test["y_t"].to_numpy(dtype=float), y_test, y_pred_test)
        preds_rows.append(
            pd.DataFrame(
                {
                    "date": test["date"],
                    "target_date": test["target_date"],
                    "y_true": y_test,
                    "y_pred": y_pred_test,
                    "model": name,
                    "H": H,
                    "split": "test",
                }
            )
        )

    # ---- SARIMAX "fair": fit na ostatnich W dla każdej prognozy
    W = int(cfg["window_W"])
    sarimax_order = (1, 1, 1)
    logger.info(f"H={H} | SARIMAX{sarimax_order} fair-fit per row (W={W}) na val+test ...")

    for split_name, part in [("val", val), ("test", test)]:
        y_pred = sarimax_forecast_from_lags(part, W=W, H=H, order=sarimax_order)
        y_true = part["target"].to_numpy(dtype=float)
        y_t = part["y_t"].to_numpy(dtype=float)

        model_name = f"SARIMAX{sarimax_order}"
        _add_eval_row(metrics_rows, H, model_name, split_name, y_t, y_true, y_pred)
        preds_rows.append(
            pd.DataFrame(
                {
                    "date": part["date"],
                    "target_date": part["target_date"],
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "model": model_name,
                    "H": H,
                    "split": split_name,
                }
            )
        )

    metrics_df = pd.DataFrame(metrics_rows)
    preds_df = pd.concat(preds_rows, ignore_index=True)
    return metrics_df, preds_df, best_baselines, best_ml_params


def main(config_path: str = "configs/config.json", run: str | None = None) -> None:
    cfg = load_config(config_path)

    run_dir = _get_or_create_run_dir(cfg, run)
    logger = setup_logger(run_dir / cfg["output"]["log_name"])
    save_run_config(cfg, run_dir)

    seed = int(cfg.get("seed", 123))
    Hs = [int(h) for h in cfg["horizons_H"]]

    all_metrics = []
    all_preds = []
    best_baselines_all: Dict[str, Any] = {}
    best_ml_all: Dict[str, Any] = {}

    for H in Hs:
        metrics_df, preds_df, best_b, best_ml = eval_all_for_H(cfg, logger, H, seed=seed)
        all_metrics.append(metrics_df)
        all_preds.append(preds_df)
        best_baselines_all[f"H{H}"] = best_b
        best_ml_all[f"H{H}"] = best_ml

    metrics_all = pd.concat(all_metrics, ignore_index=True)
    preds_all = pd.concat(all_preds, ignore_index=True)

    metrics_all.to_csv(run_dir / "metrics.csv", index=False)
    preds_all.to_csv(run_dir / "predictions.csv", index=False)
    (run_dir / "best_baselines.json").write_text(json.dumps(best_baselines_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "best_ml_params.json").write_text(json.dumps(best_ml_all, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "search_spaces.json").write_text(json.dumps(get_search_spaces(seed), ensure_ascii=False, indent=2), encoding="utf-8")

    logger.info(f"Zapisano: {run_dir/'metrics.csv'}")
    logger.info(f"Zapisano: {run_dir/'predictions.csv'}")
    logger.info(f"Zapisano: {run_dir/'best_baselines.json'}")
    logger.info(f"Zapisano: {run_dir/'best_ml_params.json'}")
    logger.info(f"Zapisano: {run_dir/'search_spaces.json'}")


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Trening + ewaluacja modeli + tuning (TimeSeriesSplit).")
    ap.add_argument("--config", default="configs/config.json", help="Ścieżka do config.json")
    ap.add_argument("--run", default=None, help="Istniejący runs/<timestamp> (opcjonalnie)")
    return ap.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(config_path=args.config, run=args.run)
