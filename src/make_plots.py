# src/make_plots.py
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from src.config import load_config, resolve_path


def _latest_run_dir(runs_dir: Path) -> Path:
    cands = [p for p in runs_dir.iterdir() if p.is_dir()]
    if not cands:
        raise FileNotFoundError(f"Brak runów w: {runs_dir}")
    return sorted(cands, key=lambda p: p.stat().st_mtime, reverse=True)[0]


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-\.]+", "_", s)[:120]


def _split_bounds(n: int, val_size: int, test_size: int) -> tuple[int, int]:
    train_end = n - (val_size + test_size)
    val_end = n - test_size
    if not (0 < train_end < val_end < n):
        raise ValueError(f"Zły split: n={n}, val={val_size}, test={test_size} -> train_end={train_end}, val_end={val_end}")
    return train_end, val_end


def _best_on_test(metrics: pd.DataFrame) -> pd.DataFrame:
    test = metrics[metrics["split"] == "test"].copy()
    idx = test.groupby("H")["RMSE"].idxmin()
    return test.loc[idx].sort_values("H").reset_index(drop=True)


def _maybe_log_x(values: np.ndarray) -> bool:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    if len(v) < 3:
        return False
    return (v.max() / np.median(v)) > 25


# -------------------- plots --------------------
def plot_series_with_split(ds_path: Path, out_path: Path, val_size: int, test_size: int, title: str) -> None:
    df = pd.read_csv(ds_path, parse_dates=["date", "target_date"]).sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end, val_end = _split_bounds(n, val_size, test_size)

    d0 = df.loc[0, "date"]
    d_train_end = df.loc[train_end, "date"]
    d_val_end = df.loc[val_end, "date"]
    dN = df.loc[n - 1, "date"]

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(df["date"], df["y_t"], linewidth=1.2, label="Kurs EUR/PLN")

    # tła: Train / Val / Test (różne alpha, żeby legenda miała sens)
    ax.axvspan(d0, d_train_end, alpha=0.06, color="gray")
    ax.axvspan(d_train_end, d_val_end, alpha=0.14, color="gray")
    ax.axvspan(d_val_end, dN, alpha=0.22, color="gray")

    # granice splitu
    ax.axvline(d_train_end, linestyle="--", linewidth=1.2, color="black")
    ax.axvline(d_val_end, linestyle="--", linewidth=1.2, color="black")

    ax.set_title(title)
    ax.set_xlabel("Data notowania")
    ax.set_ylabel("Kurs (PLN)")
    ax.grid(True, alpha=0.25)

    patches = [
        Patch(color="gray", alpha=0.06, label="Train"),
        Patch(color="gray", alpha=0.14, label="Val"),
        Patch(color="gray", alpha=0.22, label="Test"),
    ]
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + patches, labels + [p.get_label() for p in patches], loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_zoom_val_test(ds_path: Path, out_path: Path, val_size: int, test_size: int, title: str, pad_train: int = 220) -> None:
    df = pd.read_csv(ds_path, parse_dates=["date", "target_date"]).sort_values("date").reset_index(drop=True)
    n = len(df)
    train_end, val_end = _split_bounds(n, val_size, test_size)

    start = max(0, train_end - pad_train)
    z = df.iloc[start:].copy()

    d_train_end = df.loc[train_end, "date"]
    d_val_end = df.loc[val_end, "date"]
    dN = df.loc[n - 1, "date"]

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(z["date"], z["y_t"], linewidth=1.2, label="Kurs EUR/PLN")

    ax.axvspan(d_train_end, d_val_end, alpha=0.14, color="gray", label="Val")
    ax.axvspan(d_val_end, dN, alpha=0.22, color="gray", label="Test")
    ax.axvline(d_train_end, linestyle="--", linewidth=1.2, color="black")
    ax.axvline(d_val_end, linestyle="--", linewidth=1.2, color="black")

    ax.set_title(title)
    ax.set_xlabel("Data notowania")
    ax.set_ylabel("Kurs (PLN)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def bar_metric_test(metrics: pd.DataFrame, H: int, metric: str, out_path: Path) -> None:
    df = metrics[(metrics["H"] == H) & (metrics["split"] == "test")].copy().sort_values(metric, ascending=True)
    if df.empty:
        return

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.barh(df["model"], df[metric])

    if _maybe_log_x(df[metric].to_numpy()):
        ax.set_xscale("log")
        ax.set_title(f"{metric} na teście (H={H}) — skala log (outlier)")
    else:
        ax.set_title(f"{metric} na teście (H={H})")

    ax.set_xlabel(f"{metric} (PLN)")
    ax.set_ylabel("Model")
    ax.grid(True, axis="x", alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_pred_vs_true_topk(preds: pd.DataFrame, metrics: pd.DataFrame, H: int, out_path: Path, top_k: int = 3) -> None:
    test_metrics = metrics[(metrics["H"] == H) & (metrics["split"] == "test")].copy()
    if test_metrics.empty:
        return

    top = test_metrics.sort_values("RMSE").head(top_k)["model"].tolist()
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"].isin(top))].copy()
    if p.empty:
        return

    p = p.sort_values(["date", "model"])
    true_series = p.groupby("date")["y_true"].first().reset_index()

    fig, ax = plt.subplots(figsize=(13, 4.2))
    ax.plot(true_series["date"], true_series["y_true"], linewidth=1.6, label="TRUE")

    ymin = float(true_series["y_true"].min())
    ymax = float(true_series["y_true"].max())

    for m in top:
        pm = p[p["model"] == m]
        ax.plot(pm["date"], pm["y_pred"], linewidth=1.1, label=m)
        ymin = min(ymin, float(pm["y_pred"].min()))
        ymax = max(ymax, float(pm["y_pred"].max()))

    pad = 0.03 * (ymax - ymin + 1e-12)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(f"Pred vs True (test) | H={H} | top {top_k} wg RMSE")
    ax.set_xlabel("Data")
    ax.set_ylabel("Kurs (PLN)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def hist_error_best(preds: pd.DataFrame, best_row: pd.Series, out_path: Path) -> None:
    H = int(best_row["H"])
    model = str(best_row["model"])
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
    if p.empty:
        return

    err = (p["y_pred"].to_numpy(dtype=float) - p["y_true"].to_numpy(dtype=float))

    fig, ax = plt.subplots(figsize=(9, 4.4))
    ax.hist(err, bins=40)
    ax.axvline(0.0, linewidth=1.2)
    ax.set_title(f"Histogram błędu (y_pred - y_true) | H={H} | {model}")
    ax.set_xlabel("Błąd (PLN)")
    ax.set_ylabel("Liczność")
    ax.grid(True, alpha=0.20)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def error_series_best(preds: pd.DataFrame, best_row: pd.Series, out_path: Path) -> None:
    H = int(best_row["H"])
    model = str(best_row["model"])
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
    if p.empty:
        return

    p = p.sort_values("date")
    err = (p["y_pred"].to_numpy(dtype=float) - p["y_true"].to_numpy(dtype=float))

    max_abs = float(np.max(np.abs(err))) if len(err) else 1.0
    pad = 0.10 * max_abs

    fig, ax = plt.subplots(figsize=(13, 3.9))
    ax.plot(p["date"], err, linewidth=1.2, label="błąd")
    ax.axhline(0.0, linewidth=1.2)
    ax.set_ylim(-(max_abs + pad), (max_abs + pad))

    ax.set_title(f"Błąd w czasie (y_pred - y_true) | H={H} | {model}")
    ax.set_xlabel("Data")
    ax.set_ylabel("Błąd (PLN)")
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


# -------------------- main --------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="runs/<timestamp>. Jeśli brak: bierze najnowszy run.")
    ap.add_argument("--config", default=None, help="opcjonalnie: configs/config.json lub runs/<timestamp>/config.json")
    args = ap.parse_args()

    # wybór run_dir
    runs_dir = resolve_path("runs")
    run_dir = resolve_path(args.run) if args.run else _latest_run_dir(runs_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # config: jeśli nie podano, próbujemy run_dir/config.json
    cfg_path = resolve_path(args.config) if args.config else (run_dir / "config.json")
    cfg = load_config(cfg_path)

    val_size = int(cfg["split"]["val_size"])
    test_size = int(cfg["split"]["test_size"])
    W = int(cfg["window_W"])
    Hs = [int(h) for h in cfg["horizons_H"]]

    # dane
    data_dir = resolve_path(cfg["output"]["data_dir"])

    metrics_path = run_dir / "metrics.csv"
    preds_path = run_dir / "predictions.csv"
    if not metrics_path.exists() or not preds_path.exists():
        raise FileNotFoundError(f"W runie brakuje metrics.csv/predictions.csv: {run_dir}")

    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(preds_path, parse_dates=["date", "target_date"])

    # 1) kurs + split (bierzemy ds_H1 jako seria bazowa, a jak nie ma to pierwszy H z listy)
    base_H = 1 if (data_dir / "ds_H1.csv").exists() else Hs[0]
    ds_base = data_dir / f"ds_H{base_H}.csv"

    if ds_base.exists():
        plot_series_with_split(
            ds_base,
            plots_dir / "series_split.png",
            val_size=val_size,
            test_size=test_size,
            title=f"EUR/PLN + podział train/val/test (W={W})",
        )
        plot_zoom_val_test(
            ds_base,
            plots_dir / "series_split_zoom.png",
            val_size=val_size,
            test_size=test_size,
            title=f"EUR/PLN — zoom na końcówkę (val/test), W={W}",
        )

    # 2) barploty RMSE/MAE per H (test)
    for H in Hs:
        bar_metric_test(metrics, H, "RMSE", plots_dir / f"rmse_test_H{H}.png")
        bar_metric_test(metrics, H, "MAE", plots_dir / f"mae_test_H{H}.png")

    # 3) pred vs true top-3 per H
    for H in Hs:
        plot_pred_vs_true_topk(preds, metrics, H, plots_dir / f"pred_vs_true_top3_H{H}.png", top_k=3)

    # 4) best on test + błędy (hist + time series)
    best = _best_on_test(metrics)
    best.to_csv(run_dir / "best_on_test.csv", index=False)

    for _, row in best.iterrows():
        H = int(row["H"])
        model = str(row["model"])
        hist_error_best(preds, row, plots_dir / f"err_hist_best_H{H}_{_safe_name(model)}.png")
        error_series_best(preds, row, plots_dir / f"err_series_best_H{H}_{_safe_name(model)}.png")

    print(f"[OK] Wykresy: {plots_dir}")
    print(f"[OK] Best tabela: {run_dir / 'best_on_test.csv'}")


if __name__ == "__main__":
    main()
