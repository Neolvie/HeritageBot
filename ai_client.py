import os
import base64
import json
import re
import logging
from openai import AsyncOpenAI
import httpx

logger = logging.getLogger(__name__)

openai_client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "google/gemini-2.0-flash-lite")

# ─────────────────────────────────────────────────────────────────────────────
# VOICE TRANSCRIPTION
# ─────────────────────────────────────────────────────────────────────────────

async def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using OpenAI Whisper API."""
    with open(audio_path, "rb") as f:
        response = await openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            language="ru",
        )
    return response.text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# JSON EXTRACTION HELPER
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON from model response (handles markdown code blocks)."""
    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    for pattern in [r"```json\s*([\s\S]*?)\s*```", r"```\s*([\s\S]*?)\s*```"]:
        m = re.search(pattern, text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

    # Find outermost { ... } block
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Cannot extract JSON. Response start: {text[:300]}")


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS PROMPT
# ─────────────────────────────────────────────────────────────────────────────

_PROMPT_TEMPLATE = """\
Ты — ассистент по генеалогии. Твоя задача — структурировать информацию о семейной фотографии.

Пользователь описал фотографию голосом. Расшифровка (на русском языке):
\"\"\"{transcription}\"\"\"

Внимательно изучи фотографию и расшифровку. Извлеки информацию обо ВСЕХ упомянутых людях, предметах, животных и местах. Для каждого объекта определи его положение на изображении через нормализованный ограничивающий прямоугольник (координаты 0.0–1.0 от верхнего левого угла).

Верни ТОЛЬКО валидный JSON без markdown-разметки. Структура:

{{
  "photo_metadata": {{
    "date_taken": "дата съёмки или null",
    "approximate_year": "примерный год если точная дата неизвестна или null",
    "location": "место съёмки или null",
    "country": "страна или null",
    "city": "город или null",
    "photographer": "кто фотографировал или null",
    "camera_model": "марка/модель фотоаппарата или null",
    "occasion": "повод/событие или null",
    "significance": "почему фотография важна или null",
    "original_format": "плёнка/цифровое/скан и т.д. или null",
    "condition": "состояние снимка или null",
    "collection": "коллекция/альбом или null",
    "digitized_by": "кто оцифровал или null",
    "digitization_date": "когда оцифровано или null",
    "general_description": "общее описание содержимого фото",
    "additional_info": "дополнительная информация или null"
  }},
  "persons": [
    {{
      "id": "person_1",
      "last_name": "фамилия или null",
      "first_name": "имя или null",
      "patronymic": "отчество или null",
      "full_name_display": "полное ФИО для отображения или null",
      "nickname": "прозвище/псевдоним или null",
      "gender": "пол или null",
      "birth_date": "дата рождения или null",
      "birth_place": "место рождения или null",
      "death_date": "дата смерти или null",
      "death_place": "место смерти или null",
      "burial_place": "место захоронения или null",
      "relation_to_narrator": "степень родства с рассказчиком или null",
      "relations_to_others": ["родство с другими людьми на фото"],
      "nationality": "национальность или null",
      "occupation": "основная профессия/должность или null",
      "occupation_history": ["список мест работы"],
      "education": ["список учебных заведений"],
      "military_service": "военная служба или null",
      "awards_achievements": ["список наград и достижений"],
      "residence_history": ["список мест проживания"],
      "notable_facts": ["примечательные факты"],
      "hobbies_interests": ["хобби и интересы"],
      "personality_traits": ["черты характера"],
      "physical_description": "внешность/одежда на фото или null",
      "additional_info": "дополнительная информация или null",
      "bounding_box": {{"x_min": 0.05, "y_min": 0.05, "x_max": 0.45, "y_max": 0.90}}
    }}
  ],
  "objects": [
    {{
      "id": "object_1",
      "name": "название предмета или null",
      "type": "тип/категория предмета или null",
      "description": "описание или null",
      "significance": "значимость для семьи/истории или null",
      "year_created": "год создания/выпуска или null",
      "manufacturer_brand": "производитель/бренд или null",
      "model": "модель или null",
      "material": "материал изготовления или null",
      "purchase_date": "дата покупки или null",
      "purchase_place": "место покупки или null",
      "estimated_value": "приблизительная стоимость или null",
      "current_location": "текущее местонахождение или null",
      "condition": "состояние или null",
      "owner": "владелец или null",
      "provenance": "история происхождения или null",
      "additional_info": "дополнительная информация или null",
      "bounding_box": {{"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}}
    }}
  ],
  "animals": [
    {{
      "id": "animal_1",
      "name": "кличка или null",
      "species": "вид или null",
      "breed": "порода или null",
      "gender": "пол или null",
      "color": "окрас или null",
      "birth_year": "год рождения или null",
      "death_year": "год смерти или null",
      "owner": "владелец или null",
      "significance": "значимость или null",
      "additional_info": "дополнительная информация или null",
      "bounding_box": {{"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}}
    }}
  ],
  "locations_in_background": [
    {{
      "id": "location_1",
      "name": "название места или null",
      "type": "тип места (здание, парк, улица и т.д.) или null",
      "address": "адрес или null",
      "country": "страна или null",
      "region": "регион или null",
      "city": "город или null",
      "significance": "значимость или null",
      "year_of_photo_at_location": "год съёмки в этом месте или null",
      "additional_info": "дополнительная информация или null",
      "bounding_box": {{"x_min": 0.0, "y_min": 0.0, "x_max": 1.0, "y_max": 1.0}}
    }}
  ]
}}

ПРАВИЛА:
1. Bounding box — нормализованные координаты 0.0–1.0. Делай с небольшим отступом вокруг объекта.
2. Если bounding_box == {{"x_min":0.0,"y_min":0.0,"x_max":1.0,"y_max":1.0}} — значит объект не локализован.
3. Включай только сущности, упомянутые пользователем ИЛИ явно видимые на фото.
4. Все значения — на русском языке.
5. Верни ТОЛЬКО JSON, без пояснений.
"""


# ─────────────────────────────────────────────────────────────────────────────
# PHOTO ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_bbox(bbox) -> dict | None:
    """Normalize bounding box to float 0.0–1.0 dict format."""
    if not bbox:
        return None

    # Handle list format [x_min, y_min, x_max, y_max]
    if isinstance(bbox, list) and len(bbox) == 4:
        bbox = {"x_min": bbox[0], "y_min": bbox[1], "x_max": bbox[2], "y_max": bbox[3]}

    if not isinstance(bbox, dict):
        return None

    try:
        x_min = float(bbox.get("x_min", 0))
        y_min = float(bbox.get("y_min", 0))
        x_max = float(bbox.get("x_max", 1))
        y_max = float(bbox.get("y_max", 1))
    except (TypeError, ValueError):
        return None

    # If values look like pixels (> 1), assume max possible 4096 and normalize
    if any(v > 1 for v in [x_min, y_min, x_max, y_max]):
        max_val = max(x_min, y_min, x_max, y_max)
        scale = max_val if max_val > 0 else 4096
        x_min /= scale
        y_min /= scale
        x_max /= scale
        y_max /= scale

    return {
        "x_min": max(0.0, min(1.0, x_min)),
        "y_min": max(0.0, min(1.0, y_min)),
        "x_max": max(0.0, min(1.0, x_max)),
        "y_max": max(0.0, min(1.0, y_max)),
    }


def _normalize_result_bboxes(result: dict) -> dict:
    """Normalize all bounding boxes in the result."""
    for category in ("persons", "objects", "animals", "locations_in_background"):
        for item in result.get(category, []):
            raw_bbox = item.get("bounding_box")
            item["bounding_box"] = _normalize_bbox(raw_bbox)
    return result


async def analyze_photo(photo_path: str, transcription: str) -> dict:
    """Send photo + transcription to OpenRouter vision model and return structured data."""
    with open(photo_path, "rb") as f:
        image_bytes = f.read()

    ext = photo_path.lower().rsplit(".", 1)[-1]
    mime_map = {
        "jpg": "image/jpeg",
        "jpeg": "image/jpeg",
        "png": "image/png",
        "gif": "image/gif",
        "webp": "image/webp",
    }
    mime_type = mime_map.get(ext, "image/jpeg")
    image_b64 = base64.b64encode(image_bytes).decode()

    prompt = _PROMPT_TEMPLATE.format(transcription=transcription)

    payload: dict = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime_type};base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "temperature": 0.1,
    }

    # Enable JSON mode for models that support it
    if any(tag in OPENROUTER_MODEL for tag in ("gemini", "gpt-4o", "gpt-4")):
        payload["response_format"] = {"type": "json_object"}

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://heritagebot.local",
                "X-Title": "HeritageBot",
            },
            json=payload,
        )
        response.raise_for_status()
        data = response.json()

    if "error" in data:
        raise ValueError(f"OpenRouter error: {data['error']}")

    content = data["choices"][0]["message"]["content"]
    logger.debug("Raw model response: %s", content[:500])

    try:
        result = _extract_json(content)
    except ValueError as e:
        logger.error("JSON parse failed: %s", e)
        result = {
            "photo_metadata": {
                "general_description": "Не удалось разобрать ответ модели.",
                "additional_info": content[:800],
            },
            "persons": [],
            "objects": [],
            "animals": [],
            "locations_in_background": [],
        }

    return _normalize_result_bboxes(result)
