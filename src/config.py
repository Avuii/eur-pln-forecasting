# src/config.py
from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # .../project

def _resolve_today(s: str) -> str:
    return str(date.today()) if str(s).upper() == "TODAY" else str(s)

def resolve_path(p: str | Path) -> Path:
    p = Path(p)
    if p.is_absolute():
        return p

    # 1) najpierw próbujemy względem root projektu
    cand_project = PROJECT_ROOT / p
    if cand_project.exists():
        return cand_project

    # 2) potem względem bieżącego CWD (dla uruchomień z terminala)
    cand_cwd = Path.cwd() / p
    if cand_cwd.exists():
        return cand_cwd

    # jeśli nie istnieje, zwróć projektową (żeby błąd był czytelny)
    return cand_project

def load_config(path: str | Path = "configs/config.json") -> dict[str, Any]:
    path = resolve_path(path)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["date_range"]["start"] = _resolve_today(cfg["date_range"]["start"])
    cfg["date_range"]["end"] = _resolve_today(cfg["date_range"]["end"])
    return cfg

def make_run_dir(runs_dir: str | Path) -> Path:
    runs_dir = resolve_path(runs_dir)
    ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = runs_dir / ts
    (run_dir / "plots").mkdir(parents=True, exist_ok=True)
    return run_dir

def save_run_config(cfg: dict[str, Any], run_dir: Path) -> None:
    (run_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
