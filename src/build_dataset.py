# src/build_dataset.py
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import load_config, make_run_dir, save_run_config, resolve_path


# -------------------- logging --------------------
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ts_build_dataset")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, encoding="utf-8", mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def _pair_name(currency: str) -> str:
    return f"{currency.lower()}pln"


def _find_raw_csv(cfg: dict[str, Any], run_dir: Path | None) -> Path:
    """
    Szuka raw CSV:
      1) jeśli jest snapshot w run_dir -> bierzemy snapshot (reproducibility)
      2) jeśli nie ma -> bierzemy cache w data/
    """
    currency = str(cfg["currency"]).upper()
    pair = _pair_name(currency)

    data_dir = resolve_path(cfg["output"]["data_dir"])
    start = str(cfg["date_range"]["start"])
    end = str(cfg["date_range"]["end"])

    if run_dir is not None:
        snap = run_dir / f"raw_{pair}.csv"
        if snap.exists():
            return snap

    expected = data_dir / f"raw_{pair}_{start}_{end}.csv"
    if expected.exists():
        return expected

    # fallback: najnowszy raw_{pair}_*.csv
    cands = sorted(data_dir.glob(f"raw_{pair}_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not cands:
        raise FileNotFoundError(f"Nie znaleziono pliku raw_{pair}_*.csv w {data_dir}")
    return cands[0]


def _build_features(df_raw: pd.DataFrame, W: int) -> pd.DataFrame:
    """
    Buduje cechy:
      - lag_0..lag_{W-1} (poziom kursu)
      - delta_1 (y_t - y_{t-1})
      - logret_1 (log(y_t) - log(y_{t-1}))
      - sma_{5,10,20,60}, std_{5,10,20,60} (ddof=0)
      - ema_{10,20} (adjust=False)
    """
    df = df_raw.copy()
    if "date" not in df.columns or "mid" not in df.columns:
        raise ValueError("RAW CSV musi mieć kolumny: date, mid")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df["y"] = df["mid"].astype(float)

    y = df["y"]

    # lags: lag_0 = y_t, lag_1 = y_{t-1}, ...
    for k in range(W):
        df[f"lag_{k}"] = y.shift(k)

    # trendy
    df["delta_1"] = y.diff(1)
    df["logret_1"] = np.log(y).diff(1)

    # rolling
    for win in (5, 10, 20, 60):
        df[f"sma_{win}"] = y.rolling(win).mean()
        df[f"std_{win}"] = y.rolling(win).std(ddof=0)

    # EMA
    for span in (10, 20):
        df[f"ema_{span}"] = y.ewm(span=span, adjust=False).mean()

    return df


def _make_dataset_for_H(feat_df: pd.DataFrame, W: int, H: int) -> pd.DataFrame:
    """
    Dla każdego t: wektor cech z ostatnich W notowań, target = y_{t+H}.
    """
    df = feat_df.copy()
    df["y_t"] = df["y"]
    df["target_date"] = df["date"].shift(-H)
    df["target"] = df["y"].shift(-H)

    # stabilna kolejność kolumn
    meta_cols = ["date", "target_date", "y_t", "target"]
    lag_cols = [f"lag_{k}" for k in range(W)]
    other_cols = [
        "delta_1", "logret_1",
        "sma_5", "sma_10", "sma_20", "sma_60",
        "std_5", "std_10", "std_20", "std_60",
        "ema_10", "ema_20",
    ]
    keep = meta_cols + lag_cols + other_cols
    df = df[keep]

    # drop NA: początek (lagi/rolling), koniec (target)
    df = df.dropna().reset_index(drop=True)
    return df


def main(config_path: str = "configs/config.json", run_dir: str | None = None) -> None:
    cfg = load_config(config_path)

    # run_dir: jeśli podany -> użyj, jeśli nie -> utwórz nowy / weź najnowszy run (ułatwia ręczne odpalenie)
    if run_dir is None:
        run_path = make_run_dir(cfg["output"]["runs_dir"])
        save_run_config(cfg, run_path)
    else:
        run_path = resolve_path(run_dir)
        (run_path / "plots").mkdir(parents=True, exist_ok=True)
        if not (run_path / "config.json").exists():
            save_run_config(cfg, run_path)

    logger = setup_logger(run_path / cfg["output"]["log_name"])

    W = int(cfg["window_W"])
    Hs = [int(h) for h in cfg["horizons_H"]]
    currency = str(cfg["currency"]).upper()
    pair = _pair_name(currency)

    raw_csv = _find_raw_csv(cfg, run_path)
    logger.info(f"Wczytuję RAW: {raw_csv}")

    df_raw = pd.read_csv(raw_csv, parse_dates=["date"])
    feat_df = _build_features(df_raw, W=W)

    data_dir = resolve_path(cfg["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Buduję datasety: pair={pair} | W={W} | H={Hs}")

    manifest: dict[str, Any] = {
        "pair": pair,
        "W": W,
        "raw_csv_used": str(raw_csv),
        "datasets": {},
    }

    for H in Hs:
        ds = _make_dataset_for_H(feat_df, W=W, H=H)
        out_path = data_dir / f"ds_H{H}.csv"
        ds.to_csv(out_path, index=False)
        logger.info(f"Zapisano: {out_path} | rows={len(ds)} | cols={ds.shape[1]}")

        manifest["datasets"][f"H{H}"] = {
            "path": str(out_path),
            "rows": int(len(ds)),
            "cols": int(ds.shape[1]),
        }

    (run_path / "datasets_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info(f"Manifest: {run_path/'datasets_manifest.json'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--run", default=None, help="runs/<timestamp> (jeśli brak -> tworzy nowy)")
    args = ap.parse_args()
    main(config_path=args.config, run_dir=args.run)
