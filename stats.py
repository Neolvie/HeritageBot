"""
Persistent statistics tracker.

Data is stored in STATS_FILE (default: /app/data/stats.json).
Stats survive container restarts via the mounted volume.

Structure:
  - top-level totals (all time)
  - by_month["YYYY-MM"] — same sub-structure per calendar month
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATS_FILE = os.getenv("STATS_FILE", "/app/data/stats.json")


# ─────────────────────────────────────────────────────────────────────────────
# SCHEMA HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _empty_vision() -> dict:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "by_provider": {},
    }


def _empty_whisper() -> dict:
    return {
        "requests": 0,
        "seconds": 0.0,
        "cost_usd": 0.0,
    }


def _empty_period() -> dict:
    """One period (total or a single month)."""
    return {
        "requests": 0,
        "vision": _empty_vision(),
        "whisper": _empty_whisper(),
    }


def _empty_root() -> dict:
    return {
        **_empty_period(),
        "by_month": {},
        "first_request": None,
        "last_updated": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOAD / SAVE
# ─────────────────────────────────────────────────────────────────────────────

def _load() -> dict:
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return _empty_root()
    except Exception as e:
        logger.warning("Could not read stats file: %s", e)
        return _empty_root()


def _save(data: dict) -> None:
    try:
        Path(STATS_FILE).parent.mkdir(parents=True, exist_ok=True)
        data["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Could not save stats: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL MUTATORS
# ─────────────────────────────────────────────────────────────────────────────

def _add_vision(period: dict, provider: str, model: str,
                input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    period.setdefault("requests", 0)
    period["requests"] += 1

    v = period.setdefault("vision", _empty_vision())
    v["input_tokens"]  += input_tokens
    v["output_tokens"] += output_tokens
    v["cost_usd"]       = round(v["cost_usd"] + cost_usd, 6)

    key = f"{provider} / {model}"
    p = v["by_provider"].setdefault(key, {
        "requests": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
    })
    p["requests"]      += 1
    p["input_tokens"]  += input_tokens
    p["output_tokens"] += output_tokens
    p["cost_usd"]       = round(p["cost_usd"] + cost_usd, 6)


def _add_whisper(period: dict, seconds: float, cost_usd: float) -> None:
    w = period.setdefault("whisper", _empty_whisper())
    w["requests"] += 1
    w["seconds"]   = round(w["seconds"] + seconds, 1)
    w["cost_usd"]  = round(w["cost_usd"] + cost_usd, 6)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def record_vision(provider: str, model: str,
                  input_tokens: int, output_tokens: int, cost_usd: float) -> None:
    data = _load()
    now = datetime.now()
    if data["first_request"] is None:
        data["first_request"] = now.isoformat()

    month = now.strftime("%Y-%m")
    data["by_month"].setdefault(month, _empty_period())

    _add_vision(data,                    provider, model, input_tokens, output_tokens, cost_usd)
    _add_vision(data["by_month"][month], provider, model, input_tokens, output_tokens, cost_usd)
    _save(data)


def record_whisper(duration_seconds: float, cost_usd: float) -> None:
    data = _load()
    month = datetime.now().strftime("%Y-%m")
    data["by_month"].setdefault(month, _empty_period())

    _add_whisper(data,                    duration_seconds, cost_usd)
    _add_whisper(data["by_month"][month], duration_seconds, cost_usd)
    _save(data)


def reset_stats() -> None:
    _save(_empty_root())


# ─────────────────────────────────────────────────────────────────────────────
# FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def _fmt_period(period: dict, label: str) -> list[str]:
    """Format one period (total or month) into text lines."""
    v = period.get("vision", _empty_vision())
    w = period.get("whisper", _empty_whisper())
    total_cost = round(v["cost_usd"] + w["cost_usd"], 4)
    minutes = w["seconds"] / 60

    lines = [
        f"<b>{label}</b>  (запросов: {period.get('requests', 0)}, итого: <b>${total_cost:.4f}</b>)",
        f"  🎙 Whisper: {minutes:.1f} мин — ${w['cost_usd']:.4f}",
        f"  🔍 Vision: {v['input_tokens']:,} in / {v['output_tokens']:,} out — ${v['cost_usd']:.4f}",
    ]

    for key, p in v.get("by_provider", {}).items():
        lines.append(
            f"      • {key}: {p['requests']} зап., "
            f"{p['input_tokens']:,}/{p['output_tokens']:,} tok — ${p['cost_usd']:.4f}"
        )

    return lines


def format_stats() -> str:
    data = _load()

    first = (data.get("first_request") or "—")[:16].replace("T", " ")
    last  = (data.get("last_updated")  or "—")[:16].replace("T", " ")

    lines = [
        "📊 <b>Статистика HeritageBot</b>",
        f"Первый запрос: {first}   Обновлено: {last}",
        "",
    ]

    # ── Monthly breakdown (newest first) ─────────────────────────────────────
    months = sorted(data.get("by_month", {}).keys(), reverse=True)
    if months:
        lines.append("📅 <b>По месяцам:</b>")
        for month in months:
            lines += _fmt_period(data["by_month"][month], month)
            lines.append("")

    # ── All-time totals ───────────────────────────────────────────────────────
    lines += _fmt_period(data, "🗂 Всего за всё время")

    return "\n".join(lines)
