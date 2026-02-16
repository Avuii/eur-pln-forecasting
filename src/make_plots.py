from __future__ import annotations

import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


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

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["date"], df["y_t"], linewidth=1.0, label="EUR/PLN")

    d0 = df["date"].iloc[0]
    d_train_end = df["date"].iloc[train_end - 1] if train_end > 0 else df["date"].iloc[0]
    d_val_end = df["date"].iloc[val_end - 1] if val_end > 0 else df["date"].iloc[0]
    d_last = df["date"].iloc[-1]

    # tła
    ax.axvspan(d0, d_train_end, alpha=0.08)
    ax.axvspan(d_train_end, d_val_end, alpha=0.14)
    ax.axvspan(d_val_end, d_last, alpha=0.20)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Rate (PLN)")
    ax.grid(True, alpha=0.25)

    legend_patches = [
        Patch(alpha=0.08, label="Train"),
        Patch(alpha=0.14, label="Val"),
        Patch(alpha=0.20, label="Test"),
    ]
    ax.legend(handles=[ax.lines[0], *legend_patches], loc="upper left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def bar_metric_test(metrics: pd.DataFrame, H: int, metric: str, out_path: Path) -> None:
    df = metrics[(metrics["H"] == H) & (metrics["split"] == "test")].copy().sort_values(metric, ascending=True)

    plt.figure(figsize=(10, 5))
    plt.barh(df["model"], df[metric])
    plt.xlabel(metric)
    plt.ylabel("Model")

    if _maybe_log_x(df[metric].to_numpy()):
        plt.xscale("log")
        plt.title(f"{metric} on test (H={H}) — log scale (outlier present)")
    else:
        plt.title(f"{metric} on test (H={H})")

    plt.grid(True, axis="x", alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def plot_pred_vs_true_topk(preds: pd.DataFrame, metrics: pd.DataFrame, H: int, out_path: Path, top_k: int = 3) -> None:
    test_metrics = metrics[(metrics["H"] == H) & (metrics["split"] == "test")].copy()
    top = test_metrics.sort_values("RMSE").head(top_k)["model"].tolist()

    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"].isin(top))].copy()
    p = p.sort_values(["date", "model"])

    plt.figure(figsize=(12, 4))
    true_series = p.groupby("date")["y_true"].first().reset_index()
    plt.plot(true_series["date"], true_series["y_true"], linewidth=1.4, label="TRUE")

    ymin = float(true_series["y_true"].min())
    ymax = float(true_series["y_true"].max())

    for m in top:
        pm = p[p["model"] == m]
        plt.plot(pm["date"], pm["y_pred"], linewidth=1.0, label=m)
        ymin = min(ymin, float(pm["y_pred"].min()))
        ymax = max(ymax, float(pm["y_pred"].max()))

    pad = 0.03 * (ymax - ymin + 1e-12)
    plt.ylim(ymin - pad, ymax + pad)

    plt.title(f"Pred vs True (test) | H={H} | top {top_k} by RMSE")
    plt.xlabel("Date")
    plt.ylabel("Rate (PLN)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def hist_error_best(preds: pd.DataFrame, best_row: pd.Series, out_path: Path) -> None:
    H = int(best_row["H"])
    model = str(best_row["model"])
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
    err = (p["y_pred"].to_numpy(dtype=float) - p["y_true"].to_numpy(dtype=float))

    plt.figure(figsize=(8, 4))
    plt.hist(err, bins=40)
    plt.title(f"Error histogram (y_pred - y_true) | H={H} | {model}")
    plt.xlabel("Error (PLN)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def error_series_best(preds: pd.DataFrame, best_row: pd.Series, out_path: Path) -> None:
    H = int(best_row["H"])
    model = str(best_row["model"])
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
    p = p.sort_values("date")
    err = (p["y_pred"].to_numpy(dtype=float) - p["y_true"].to_numpy(dtype=float))

    plt.figure(figsize=(12, 3.5))
    plt.plot(p["date"], err, linewidth=1.0, label="Residual")
    plt.axhline(0.0, linewidth=1.0)
    plt.title(f"Residuals vs time (y_pred - y_true) | H={H} | {model}")
    plt.xlabel("Date")
    plt.ylabel("Residual (PLN)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()


def abs_error_series_best(preds: pd.DataFrame, best_row: pd.Series, out_path: Path) -> float:
    H = int(best_row["H"])
    model = str(best_row["model"])
    p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
    p = p.sort_values("date")

    abs_err = np.abs(p["y_pred"].to_numpy(dtype=float) - p["y_true"].to_numpy(dtype=float))
    thr = float(np.percentile(abs_err, 95))

    plt.figure(figsize=(12, 3.5))
    plt.plot(p["date"], abs_err, linewidth=1.0, label="|error|")
    plt.axhline(thr, linewidth=1.0, label="p95(|error|)")
    plt.title(f"Absolute error vs time | H={H} | {model}")
    plt.xlabel("Date")
    plt.ylabel("|Error| (PLN)")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=220)
    plt.close()

    return thr


# -----------------------stability / outliers report --------------------
def _merge_with_ds(data_dir: Path, H: int, p: pd.DataFrame) -> pd.DataFrame:
    ds_path = data_dir / f"ds_H{H}.csv"
    if not ds_path.exists():
        return p.copy()

    ds = pd.read_csv(ds_path, parse_dates=["date", "target_date"])
    ds = ds[["date", "target_date", "y_t", "target"]].copy()
    merged = p.merge(ds, on=["date", "target_date"], how="left", suffixes=("", "_ds"))
    # target == y_true,  keep y_true from preds as source of truth
    return merged


def _find_bad_ranges(dates: pd.Series, mask: np.ndarray) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    if len(mask) == 0:
        return []

    ranges: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    start = None
    count = 0

    for i, flag in enumerate(mask):
        if flag and start is None:
            start = dates.iloc[i]
            count = 1
        elif flag and start is not None:
            count += 1
        elif (not flag) and start is not None:
            end = dates.iloc[i - 1]
            ranges.append((start, end, count))
            start = None
            count = 0

    if start is not None:
        ranges.append((start, dates.iloc[-1], count))

    return ranges


def write_stability_report(
    run_dir: Path,
    plots_dir: Path,
    preds: pd.DataFrame,
    best: pd.DataFrame,
    data_dir: Path,
) -> None:
    stats_rows = []
    md = []
    md.append("# Stability / outlier analysis (best-on-test)\n")

    for _, row in best.iterrows():
        H = int(row["H"])
        model = str(row["model"])

        p = preds[(preds["H"] == H) & (preds["split"] == "test") & (preds["model"] == model)].copy()
        p = p.sort_values("date")
        p = _merge_with_ds(data_dir, H, p)

        err = (p["y_pred"].to_numpy(float) - p["y_true"].to_numpy(float))
        abs_err = np.abs(err)

        p50 = float(np.percentile(abs_err, 50))
        p90 = float(np.percentile(abs_err, 90))
        p95 = float(np.percentile(abs_err, 95))
        p99 = float(np.percentile(abs_err, 99))
        mx = float(np.max(abs_err))
        mean_abs = float(np.mean(abs_err))
        mean_err = float(np.mean(err))
        pos_frac = float(np.mean(err > 0))

        stats_rows.append({
            "H": H,
            "model": model,
            "mean_abs_error": mean_abs,
            "p50_abs_error": p50,
            "p90_abs_error": p90,
            "p95_abs_error": p95,
            "p99_abs_error": p99,
            "max_abs_error": mx,
            "mean_error_bias": mean_err,
            "frac_error_positive": pos_frac,
        })

        # wykres |error| + próg
        abs_error_series_best(p, row, plots_dir / f"abs_err_series_best_H{H}_{_safe_name(model)}.png")

        # “kiedy siada”: odcinki powyżej p95
        bad_mask = abs_err > p95
        bad_ranges = _find_bad_ranges(p["date"], bad_mask)

        # top worst
        dfw = p.copy()
        dfw["error"] = err
        dfw["abs_error"] = abs_err
        if "y_t" in dfw.columns:
            dfw["jump"] = dfw["y_true"] - dfw["y_t"]
        dfw = dfw.sort_values("abs_error", ascending=False).head(10)

        md.append(f"## H={H} — **{model}**\n")
        md.append(f"- mean(|error|) = **{mean_abs:.6f} PLN**")
        md.append(f"- p90/p95/p99(|error|) = **{p90:.6f} / {p95:.6f} / {p99:.6f} PLN**")
        md.append(f"- max(|error|) = **{mx:.6f} PLN**")
        md.append(f"- bias mean(error) = **{mean_err:.6f} PLN** (fraction error>0: **{pos_frac:.2%}**)\n")

        if bad_ranges:
            md.append("**High-error periods (|error| > p95):**")
            for (a, b, ln) in bad_ranges[:8]:
                md.append(f"- {a.date()} → {b.date()}  (len={ln})")
            md.append("")
        else:
            md.append("**High-error periods:** none detected above p95.\n")

        md.append("**Worst 10 predictions (by |error|):**\n")
        cols = ["date", "target_date", "y_true", "y_pred", "error", "abs_error"]
        if "y_t" in dfw.columns:
            cols = ["date", "target_date", "y_t", "y_true", "y_pred", "jump", "error", "abs_error"]

        # markdown table
        md.append("| " + " | ".join(cols) + " |")
        md.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, rr in dfw.iterrows():
            vals = []
            for c in cols:
                v = rr.get(c)
                if isinstance(v, (pd.Timestamp,)):
                    vals.append(str(v.date()))
                elif isinstance(v, (float, np.floating, int, np.integer)):
                    vals.append(f"{float(v):.6f}")
                else:
                    vals.append("" if pd.isna(v) else str(v))
            md.append("| " + " | ".join(vals) + " |")
        md.append("")

        if "jump" in dfw.columns:
            # jeśli większość top błędów pokrywa się z dużymi jumpami -> “siada na skokach”
            big_jump_thr = float(np.percentile(np.abs((p["y_true"] - p["y_t"]).dropna()), 90)) if "y_t" in p.columns else np.nan
            frac_big_jump = float(np.mean(np.abs(dfw["jump"].to_numpy(float)) >= big_jump_thr)) if np.isfinite(big_jump_thr) else 0.0
            if np.isfinite(big_jump_thr) and frac_big_jump >= 0.6:
                md.append(f"> Comment: errors concentrate around **larger moves** (|jump| ≥ p90 ≈ {big_jump_thr:.6f} PLN). Model tends to “sit down” on jumps.\n")
            else:
                md.append("> Comment: no strong evidence that worst errors happen only on jumps; looks more spread out.\n")

    stats_df = pd.DataFrame(stats_rows).sort_values("H")
    stats_df.to_csv(run_dir / "stability_stats_best.csv", index=False)
    (run_dir / "stability_report.md").write_text("\n".join(md), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="Ścieżka do runs/<timestamp>. Jeśli brak: bierze najnowszy run.")
    ap.add_argument("--data_dir", default="data", help="Folder z ds_H*.csv (domyślnie: data).")
    ap.add_argument("--val_size", type=int, default=130)
    ap.add_argument("--test_size", type=int, default=260)
    ap.add_argument("--W", type=int, default=60)
    ap.add_argument("--Hs", default="1,7,30,60")
    args = ap.parse_args()

    runs_dir = Path("runs")
    run_dir = Path(args.run) if args.run else _latest_run_dir(runs_dir)
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = run_dir / "metrics.csv"
    preds_path = run_dir / "predictions.csv"
    if not metrics_path.exists() or not preds_path.exists():
        raise FileNotFoundError(f"W runie brakuje metrics.csv/predictions.csv: {run_dir}")

    metrics = pd.read_csv(metrics_path)
    preds = pd.read_csv(preds_path, parse_dates=["date", "target_date"])

    horizons = [int(x.strip()) for x in args.Hs.split(",") if x.strip()]
    data_dir = Path(args.data_dir)

    #kurs + split (ds_H{first}.csv jako seria bazowa)
    ds_base = data_dir / f"ds_H{horizons[0]}.csv"
    if ds_base.exists():
        plot_series_with_split(
            ds_base,
            plots_dir / "series_split.png",
            val_size=args.val_size,
            test_size=args.test_size,
            title=f"EUR/PLN with train/val/test split (W={args.W})",
        )

    #barplots RMSE/MAE per H (test)
    for H in horizons:
        bar_metric_test(metrics, H, "RMSE", plots_dir / f"rmse_test_H{H}.png")
        bar_metric_test(metrics, H, "MAE", plots_dir / f"mae_test_H{H}.png")

    #pred vs true top-3 per H
    for H in horizons:
        plot_pred_vs_true_topk(preds, metrics, H, plots_dir / f"pred_vs_true_top3_H{H}.png", top_k=3)

    #best on test + błędy (hist + residual series)
    best = _best_on_test(metrics)
    best.to_csv(run_dir / "best_on_test.csv", index=False)

    for _, row in best.iterrows():
        H = int(row["H"])
        model = str(row["model"])
        hist_error_best(preds, row, plots_dir / f"err_hist_best_H{H}_{_safe_name(model)}.png")
        error_series_best(preds, row, plots_dir / f"err_series_best_H{H}_{_safe_name(model)}.png")

    #stability report + abs error plots + percentiles/max
    write_stability_report(
        run_dir=run_dir,
        plots_dir=plots_dir,
        preds=preds,
        best=best,
        data_dir=data_dir,
    )

    print(f"[OK] Wykresy: {plots_dir}")
    print(f"[OK] Best tabela: {run_dir / 'best_on_test.csv'}")
    print(f"[OK] Stability: {run_dir / 'stability_report.md'} + {run_dir / 'stability_stats_best.csv'}")


if __name__ == "__main__":
    main()
