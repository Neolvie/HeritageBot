"""
Persistent statistics tracker.

Data is stored in STATS_FILE (default: /app/data/stats.json).
Stats survive container restarts via the mounted volume.
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

STATS_FILE = os.getenv("STATS_FILE", "/app/data/stats.json")


# ─────────────────────────────────────────────────────────────────────────────
# INTERNAL HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _empty() -> dict:
    return {
        "total_requests": 0,
        "vision": {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cost_usd": 0.0,
            "by_provider": {},
        },
        "whisper": {
            "total_requests": 0,
            "total_seconds": 0.0,
            "total_cost_usd": 0.0,
        },
        "first_request": None,
        "last_updated": None,
    }


def _load() -> dict:
    try:
        with open(STATS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return _empty()
    except Exception as e:
        logger.warning("Could not read stats file: %s", e)
        return _empty()


def _save(data: dict) -> None:
    try:
        Path(STATS_FILE).parent.mkdir(parents=True, exist_ok=True)
        data["last_updated"] = datetime.now().isoformat()
        with open(STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("Could not save stats: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────────────────────

def record_vision(
    provider: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    data = _load()
    data["total_requests"] += 1
    if data["first_request"] is None:
        data["first_request"] = datetime.now().isoformat()

    v = data["vision"]
    v["total_input_tokens"] += input_tokens
    v["total_output_tokens"] += output_tokens
    v["total_cost_usd"] = round(v["total_cost_usd"] + cost_usd, 6)

    key = f"{provider} / {model}"
    if key not in v["by_provider"]:
        v["by_provider"][key] = {
            "requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "cost_usd": 0.0,
        }
    p = v["by_provider"][key]
    p["requests"] += 1
    p["input_tokens"] += input_tokens
    p["output_tokens"] += output_tokens
    p["cost_usd"] = round(p["cost_usd"] + cost_usd, 6)

    _save(data)


def record_whisper(duration_seconds: float, cost_usd: float) -> None:
    data = _load()
    w = data["whisper"]
    w["total_requests"] += 1
    w["total_seconds"] = round(w["total_seconds"] + duration_seconds, 1)
    w["total_cost_usd"] = round(w["total_cost_usd"] + cost_usd, 6)
    _save(data)


def reset_stats() -> None:
    _save(_empty())


def format_stats() -> str:
    data = _load()
    v = data["vision"]
    w = data["whisper"]

    total_cost = round(v["total_cost_usd"] + w["total_cost_usd"], 4)
    total_minutes = w["total_seconds"] / 60

    first = data.get("first_request", "—")
    last = data.get("last_updated", "—")
    if first and first != "—":
        first = first[:16].replace("T", " ")
    if last and last != "—":
        last = last[:16].replace("T", " ")

    lines = [
        "📊 <b>Статистика HeritageBot</b>",
        "",
        f"Всего обработок: <b>{data['total_requests']}</b>",
        f"Первый запрос: {first}",
        f"Последнее обновление: {last}",
        "",
        "🎙 <b>Whisper (распознавание голоса)</b>",
        f"  Запросов: {w['total_requests']}",
        f"  Суммарно: {total_minutes:.1f} мин ({w['total_seconds']:.0f} сек)",
        f"  Стоимость: ${w['total_cost_usd']:.4f}",
        "",
        "🔍 <b>Анализ фотографий (vision)</b>",
        f"  Входящих токенов: {v['total_input_tokens']:,}",
        f"  Исходящих токенов: {v['total_output_tokens']:,}",
        f"  Стоимость: ${v['total_cost_usd']:.4f}",
    ]

    if v["by_provider"]:
        lines.append("")
        lines.append("  <i>По провайдерам:</i>")
        for key, p in v["by_provider"].items():
            lines.append(f"  • <b>{key}</b>")
            lines.append(f"    Запросов: {p['requests']}")
            lines.append(
                f"    Токены: {p['input_tokens']:,} in / {p['output_tokens']:,} out"
            )
            lines.append(f"    Стоимость: ${p['cost_usd']:.4f}")

    lines += [
        "",
        f"💰 <b>Итого: ${total_cost:.4f}</b>",
    ]

    return "\n".join(lines)
