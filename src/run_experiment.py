from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime

from src.config import load_config, resolve_path, save_run_config


def main() -> None:
    ap = argparse.ArgumentParser(description="Runner: fetch -> build_dataset -> train_eval -> make_plots (jeden run_dir).")
    ap.add_argument("--config", default="configs/config.json", help="Bazowy config")
    ap.add_argument("--currency", default=None, help="Np. EUR")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD")
    ap.add_argument("-W", "--window", type=int, default=None, help="Okno wejściowe W")
    ap.add_argument("-H", "--horizons", nargs="+", type=int, default=None, help="Horyzonty H (np. 1 7 30 60)")
    ap.add_argument("--force_refresh", action="store_true", help="Wymuś ponowne pobranie danych (ignoruj cache)")
    args = ap.parse_args()

    cfg = load_config(args.config)

    # override z CLI
    if args.currency is not None:
        cfg["currency"] = args.currency
    if args.start is not None:
        cfg["date_range"]["start"] = args.start
    if args.end is not None:
        cfg["date_range"]["end"] = args.end
    if args.window is not None:
        cfg["window_W"] = int(args.window)
    if args.horizons is not None:
        cfg["horizons_H"] = [int(h) for h in args.horizons]

    # jeden wspólny run_dir
    runs_root = resolve_path(cfg["output"]["runs_dir"])
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = runs_root / ts
    run_dir.mkdir(parents=True, exist_ok=True)

    # snapshot config do run_dir
    save_run_config(cfg, run_dir)
    cfg_in_run = str(run_dir / "config.json")

    py = sys.executable

    cmds = [
        [py, "-m", "src.data_fetch", "--config", cfg_in_run, "--run", str(run_dir)] + (["--force_refresh"] if args.force_refresh else []),
        [py, "-m", "src.build_dataset", "--config", cfg_in_run, "--run", str(run_dir)],
        [py, "-m", "src.train_eval", "--config", cfg_in_run, "--run", str(run_dir)],
        [py, "-m", "src.make_plots", "--config", cfg_in_run, "--run", str(run_dir)],
    ]

    for c in cmds:
        print(">>", " ".join(c))
        subprocess.check_call(c)

    print(f"[OK] Gotowe. Wyniki: {run_dir}")


if __name__ == "__main__":
    main()
