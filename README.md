# HeritageBot

Telegram-бот для структурированного описания семейных фотографий — proof-of-concept генеалогического архива.

Пользователь загружает фото и рассказывает голосом, кто и что на нём изображено. Бот распознаёт речь, анализирует снимок и возвращает структурированные карточки каждого человека/предмета, а также полный JSON-файл для дальнейшего построения генеалогического дерева.

---

## Как это работает

```
Фото → Голосовое сообщение → Whisper (расшифровка)
                                    ↓
                       OpenRouter vision-модель
                                    ↓
        ┌───────────────────────────────────────────────┐
        │  Сообщение 1: метаданные фотографии           │
        │  Сообщение 2…N: вырезанный фрагмент + карточка│
        │  Последнее: heritage_YYYYMMDD_HHMMSS.json     │
        └───────────────────────────────────────────────┘
```

---

## Быстрый старт

### 1. Получить токены

| Токен | Где взять |
|-------|-----------|
| `TELEGRAM_BOT_TOKEN` | [@BotFather](https://t.me/BotFather) |
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| `OPENROUTER_API_KEY` | [openrouter.ai/keys](https://openrouter.ai/keys) |

### 2. Настроить переменные окружения

```bash
cp .env.example .env
# Открыть .env и вписать токены
```

`.env`:
```
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
OPENAI_API_KEY=your_openai_api_key_here
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_MODEL=google/gemini-2.0-flash-lite
```

### 3. Запустить

```bash
docker-compose up -d
```

Проверить логи:
```bash
docker-compose logs -f
```

Остановить:
```bash
docker-compose down
```

---

## Сценарий использования

1. Откройте бот в Telegram, отправьте `/start`
2. Отправьте фотографию
3. Запишите голосовое сообщение, например:
   > *«Это фото сделано в 1978 году в Москве. Слева дедушка — Николаев Степан Михайлович, 1923 года рождения, ветеран войны. Справа бабушка — Николаева Анна Петровна. На столе стоит патефон, который дедушка купил в 1955 году»*
4. Получите:
   - Карточку с метаданными фото
   - Вырезанные портреты с ФИО, датами, фактами
   - Карточку патефона с историей
   - Файл `heritage_*.json`

Команды: `/reset` — сбросить текущую сессию

---

## Структура JSON

```json
{
  "photo_metadata": {
    "date_taken": "1978",
    "location": "Москва",
    "occasion": "Семейный обед",
    "significance": "...",
    ...
  },
  "persons": [
    {
      "id": "person_1",
      "last_name": "Николаев",
      "first_name": "Степан",
      "patronymic": "Михайлович",
      "birth_date": "1923",
      "military_service": "Ветеран ВОВ",
      "relation_to_narrator": "Дедушка",
      "bounding_box": { "x_min": 0.05, "y_min": 0.1, "x_max": 0.4, "y_max": 0.9 },
      ...
    }
  ],
  "objects": [ { "name": "Патефон", "year_created": "1955", ... } ],
  "animals": [],
  "locations_in_background": [],
  "processing_metadata": {
    "transcription": "...",
    "processed_at": "2025-08-01T14:23:00",
    "model_used": "google/gemini-2.0-flash-lite"
  }
}
```

Полный список полей — см. исходники `ai_client.py` (промпт) и `formatters.py`.

---

## Структура проекта

```
HeritageBot/
├── bot.py              # Точка входа, Telegram-хендлеры
├── ai_client.py        # Whisper (речь→текст) + OpenRouter (анализ фото)
├── image_utils.py      # Ресайз фото, нарезка регионов по bounding box
├── formatters.py       # HTML-форматирование карточек для Telegram
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

---

## Выбор vision-модели

Модель задаётся через `OPENROUTER_MODEL` в `.env`. Модель **обязана поддерживать vision (image input)**.

| Модель | Цена (вход) | Качество |
|--------|-------------|----------|
| `google/gemini-2.0-flash-lite` | ~$0.075/1M | ⭐⭐⭐ — рекомендуется |
| `google/gemini-flash-1.5` | ~$0.075/1M | ⭐⭐⭐ |
| `meta-llama/llama-3.2-90b-vision-instruct` | ~$0.35/1M | ⭐⭐⭐⭐ |
| `anthropic/claude-3.5-haiku` | ~$0.80/1M | ⭐⭐⭐⭐⭐ |

Актуальные цены: [openrouter.ai/models](https://openrouter.ai/models)

---

## Требования

- Docker + Docker Compose
- Токены: Telegram Bot, OpenAI, OpenRouter
