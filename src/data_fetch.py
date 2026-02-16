from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import requests

from src.config import load_config, make_run_dir, save_run_config, resolve_path

NBP_BASE = "https://api.nbp.pl/api"
MAX_DAYS_PER_REQUEST = 93  # NBP limit: max 93 days per request


# -------------------- logging --------------------
def setup_logger(log_path: Path) -> logging.Logger:
    logger = logging.getLogger("ts_fetch")
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


# -------------------- helpers --------------------
def _to_date(s: str) -> date:
    return date.fromisoformat(str(s))


def _chunks_by_93_days(start: date, end: date) -> Iterable[tuple[date, date]]:
    cur = start
    step = timedelta(days=MAX_DAYS_PER_REQUEST - 1)  # inclusive
    while cur <= end:
        chunk_end = min(cur + step, end)
        yield cur, chunk_end
        cur = chunk_end + timedelta(days=1)


def _pair_name(currency: str) -> str:
    return f"{currency.lower()}pln"


def fetch_rates_range(
    table: str,
    code: str,
    start: date,
    end: date,
    session: requests.Session,
    logger: logging.Logger,
) -> list[dict[str, Any]]:

    url = f"{NBP_BASE}/exchangerates/rates/{table}/{code}/{start.isoformat()}/{end.isoformat()}/"
    headers = {"Accept": "application/json"}
    params = {"format": "json"}

    r = session.get(url, headers=headers, params=params, timeout=30)

    if r.status_code == 404:
        logger.warning(f"404 no data: {start}..{end}")
        return []

    if not r.ok:
        logger.error(f"HTTP {r.status_code} dla {url} | body: {r.text[:400]}")
        r.raise_for_status()

    payload = r.json()
    return payload.get("rates", [])


def compute_missing_days(df: pd.DataFrame, start: date, end: date) -> tuple[dict[str, Any], pd.DataFrame]:
    all_days = pd.date_range(start=start, end=end, freq="D")
    calendar_days = int((end - start).days + 1)

    obs_days = pd.to_datetime(df["date"]).dt.date
    obs_set = set(obs_days.tolist())

    missing = [d.date() for d in all_days if d.date() not in obs_set]
    missing_df = pd.DataFrame({"date": pd.to_datetime(missing)})

    # per-year: calendar/obs/missing
    all_days_df = pd.DataFrame({"date": all_days})
    all_days_df["year"] = all_days_df["date"].dt.year
    cal_per_year = all_days_df.groupby("year").size().to_dict()

    obs_df = df.copy()
    obs_df["date"] = pd.to_datetime(obs_df["date"])
    obs_df["year"] = obs_df["date"].dt.year
    obs_per_year = obs_df.groupby("year").size().to_dict()

    if missing_df.empty:
        per_year = {}
        for y in sorted(cal_per_year.keys()):
            cal = int(cal_per_year.get(y, 0))
            obs = int(obs_per_year.get(y, 0))
            per_year[str(int(y))] = {
                "calendar_days": cal,
                "observations": obs,
                "missing_total": int(cal - obs),
                "missing_weekend": None,
                "missing_weekday": None,
                "missing_ratio": float((cal - obs) / cal) if cal > 0 else 0.0,
            }

        summary = {
            "start": str(start),
            "end": str(end),
            "calendar_days": calendar_days,
            "observations": int(len(df)),
            "missing_days_total": int(calendar_days - len(df)),
            "missing_weekend": None,
            "missing_weekday": None,
            "missing_ratio": float((calendar_days - len(df)) / calendar_days) if calendar_days > 0 else 0.0,
            "per_year": per_year,
        }
        return summary, missing_df

    missing_df["weekday_idx"] = missing_df["date"].dt.weekday  # 0..6
    missing_df["is_weekend"] = missing_df["weekday_idx"] >= 5
    missing_df["year"] = missing_df["date"].dt.year

    missing_total = int(len(missing_df))
    missing_weekend = int(missing_df["is_weekend"].sum())
    missing_weekday = int(missing_total - missing_weekend)
    missing_ratio = float(missing_total / calendar_days) if calendar_days > 0 else 0.0

    miss_per_year = missing_df.groupby("year").size().to_dict()
    years = sorted(set(cal_per_year.keys()) | set(obs_per_year.keys()) | set(miss_per_year.keys()))

    per_year = {}
    for y in years:
        cal = int(cal_per_year.get(y, 0))
        obs = int(obs_per_year.get(y, 0))
        miss = int(miss_per_year.get(y, 0))
        g = missing_df[missing_df["year"] == y]
        w_end = int(g["is_weekend"].sum()) if not g.empty else 0
        w_day = int(miss - w_end)

        per_year[str(int(y))] = {
            "calendar_days": cal,
            "observations": obs,
            "missing_total": miss,
            "missing_weekend": w_end,
            "missing_weekday": w_day,
            "missing_ratio": float(miss / cal) if cal > 0 else 0.0,
        }

    summary = {
        "start": str(start),
        "end": str(end),
        "calendar_days": calendar_days,
        "observations": int(len(df)),
        "missing_days_total": missing_total,
        "missing_weekend": missing_weekend,
        "missing_weekday": missing_weekday,
        "missing_ratio": missing_ratio,
        "per_year": per_year,
    }
    return summary, missing_df


# -------------------- main --------------------
def main(config_path: str = "configs/config.json", run_dir: str | None = None, force_refresh: bool = False) -> Path:
    cfg = load_config(config_path)

    if run_dir is None:
        run_path = make_run_dir(cfg["output"]["runs_dir"])
        save_run_config(cfg, run_path)
    else:
        run_path = resolve_path(run_dir)
        (run_path / "plots").mkdir(parents=True, exist_ok=True)

        if not (run_path / "config.json").exists():
            save_run_config(cfg, run_path)

    logger = setup_logger(run_path / cfg["output"]["log_name"])

    data_dir = resolve_path(cfg["output"]["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)

    currency = str(cfg["currency"]).upper()     # EUR
    table = str(cfg["nbp_table"]).upper()       # A
    pair = _pair_name(currency)                 # eurpln

    start = _to_date(cfg["date_range"]["start"])
    end = _to_date(cfg["date_range"]["end"])
    if start > end:
        raise ValueError(f"start_date > end_date: {start} > {end}")

    out_csv = data_dir / f"raw_{pair}_{start}_{end}.csv"
    snap_csv = run_path / f"raw_{pair}.csv"

    # cache
    if out_csv.exists() and not force_refresh:
        logger.info(f"Cache hit: {out_csv} (force_refresh=False) -> wczytuję")
        df = pd.read_csv(out_csv, parse_dates=["date"])
        df.to_csv(snap_csv, index=False)
        logger.info(f"Snapshot zapisany: {snap_csv}")

        summary, missing_df = compute_missing_days(df, start, end)
        (run_path / "missing_days.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        missing_df.to_csv(run_path / "missing_dates.csv", index=False)
        logger.info(
            "Braki danych (kalendarzowo): missing_total=%d | weekend=%s | weekday=%s | ratio=%.3f",
            summary["missing_days_total"],
            str(summary["missing_weekend"]),
            str(summary["missing_weekday"]),
            summary["missing_ratio"],
        )
        return out_csv

    logger.info(f"Pobieram {currency}/PLN z NBP | tabela={table} | {start}..{end}")
    chunks = list(_chunks_by_93_days(start, end))
    logger.info(f"Liczba requestów (<=93 dni): {len(chunks)}")

    rows: list[tuple[str, float]] = []
    with requests.Session() as session:
        for i, (cs, ce) in enumerate(chunks, start=1):
            logger.info(f"[{i}/{len(chunks)}] GET {cs}..{ce}")
            rates = fetch_rates_range(table, currency, cs, ce, session, logger)
            for r in rates:
                d = r.get("effectiveDate")
                mid = r.get("mid")
                if d is None or mid is None:
                    continue
                rows.append((str(d), float(mid)))

    if not rows:
        raise RuntimeError("NBP API nie zwróciło żadnych danych dla zadanego zakresu.")

    df = pd.DataFrame(rows, columns=["date", "mid"])
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)

    df.to_csv(out_csv, index=False)
    df.to_csv(snap_csv, index=False)
    logger.info(
        f"Zapisano: {out_csv} | rekordów={len(df)} | {df['date'].min().date()}..{df['date'].max().date()}"
    )
    logger.info(f"Snapshot zapisany: {snap_csv}")

    # braki kalendarzowe
    summary, missing_df = compute_missing_days(df, start, end)
    (run_path / "missing_days.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    missing_df.to_csv(run_path / "missing_dates.csv", index=False)

    logger.info(
        "Braki danych (kalendarzowo): missing_total=%d | weekend=%d | weekday(święta/inne)=%d | ratio=%.3f",
        summary["missing_days_total"],
        summary["missing_weekend"],
        summary["missing_weekday"],
        summary["missing_ratio"],
    )
    logger.info(f"Zapisano: {run_path/'missing_days.json'} oraz {run_path/'missing_dates.csv'}")

    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.json")
    ap.add_argument("--run", default=None, help="runs/<timestamp> (jeśli brak -> tworzy nowy)")
    ap.add_argument("--force_refresh", action="store_true")
    args = ap.parse_args()

    main(config_path=args.config, run_dir=args.run, force_refresh=args.force_refresh)
