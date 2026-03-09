"""
Formatters: convert structured dicts to human-readable HTML strings
for Telegram messages (parse_mode="HTML").
"""


def esc(text) -> str:
    """Escape HTML special characters."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _row(label: str, value) -> str:
    """Single field row: <b>Label:</b> value"""
    if value is None or value == "" or value == []:
        return ""
    if isinstance(value, list):
        lines = [f"<b>{label}:</b> {esc(v)}" for v in value if v]
        return "\n".join(lines)
    return f"<b>{label}:</b> {esc(value)}"


def _block(pairs: list[tuple[str, object]]) -> str:
    rows = [_row(label, val) for label, val in pairs]
    return "\n".join(r for r in rows if r)


# ─────────────────────────────────────────────────────────────────────────────

def format_photo_metadata(meta: dict) -> str:
    text = _block([
        ("Дата съёмки",        meta.get("date_taken")),
        ("Примерный год",      meta.get("approximate_year")),
        ("Место съёмки",       meta.get("location")),
        ("Страна",             meta.get("country")),
        ("Город",              meta.get("city")),
        ("Фотограф",           meta.get("photographer")),
        ("Фотоаппарат",        meta.get("camera_model")),
        ("Повод / событие",    meta.get("occasion")),
        ("Значимость",         meta.get("significance")),
        ("Формат",             meta.get("original_format")),
        ("Состояние",          meta.get("condition")),
        ("Коллекция",          meta.get("collection")),
        ("Оцифровано",         meta.get("digitized_by")),
        ("Дата оцифровки",     meta.get("digitization_date")),
        ("Описание",           meta.get("general_description")),
        ("Доп. информация",    meta.get("additional_info")),
    ])
    return text or "Информация о фотографии не указана"


def format_person(person: dict) -> str:
    text = _block([
        ("Фамилия",               person.get("last_name")),
        ("Имя",                   person.get("first_name")),
        ("Отчество",              person.get("patronymic")),
        ("Псевдоним / прозвище",  person.get("nickname")),
        ("Пол",                   person.get("gender")),
        ("Дата рождения",         person.get("birth_date")),
        ("Место рождения",        person.get("birth_place")),
        ("Дата смерти",           person.get("death_date")),
        ("Место смерти",          person.get("death_place")),
        ("Место захоронения",     person.get("burial_place")),
        ("Степень родства",       person.get("relation_to_narrator")),
        ("Родственные связи",     person.get("relations_to_others")),
        ("Национальность",        person.get("nationality")),
        ("Профессия / должность", person.get("occupation")),
        ("История работы",        person.get("occupation_history")),
        ("Образование",           person.get("education")),
        ("Военная служба",        person.get("military_service")),
        ("Награды / достижения",  person.get("awards_achievements")),
        ("История проживания",    person.get("residence_history")),
        ("Примечательные факты",  person.get("notable_facts")),
        ("Хобби / интересы",      person.get("hobbies_interests")),
        ("Черты характера",       person.get("personality_traits")),
        ("Внешность на фото",     person.get("physical_description")),
        ("Доп. информация",       person.get("additional_info")),
    ])
    return text or "Информация не указана"


def format_object(obj: dict) -> str:
    text = _block([
        ("Тип",                      obj.get("type")),
        ("Описание",                 obj.get("description")),
        ("Значимость",               obj.get("significance")),
        ("Год создания",             obj.get("year_created")),
        ("Производитель / бренд",    obj.get("manufacturer_brand")),
        ("Модель",                   obj.get("model")),
        ("Материал",                 obj.get("material")),
        ("Дата покупки",             obj.get("purchase_date")),
        ("Место покупки",            obj.get("purchase_place")),
        ("Стоимость",                obj.get("estimated_value")),
        ("Текущее местонахождение",  obj.get("current_location")),
        ("Состояние",                obj.get("condition")),
        ("Владелец",                 obj.get("owner")),
        ("История происхождения",    obj.get("provenance")),
        ("Доп. информация",          obj.get("additional_info")),
    ])
    return text or "Информация не указана"


def format_animal(animal: dict) -> str:
    text = _block([
        ("Вид",              animal.get("species")),
        ("Порода",           animal.get("breed")),
        ("Пол",              animal.get("gender")),
        ("Окрас",            animal.get("color")),
        ("Год рождения",     animal.get("birth_year")),
        ("Год смерти",       animal.get("death_year")),
        ("Владелец",         animal.get("owner")),
        ("Значимость",       animal.get("significance")),
        ("Доп. информация",  animal.get("additional_info")),
    ])
    return text or "Информация не указана"


def format_location(loc: dict) -> str:
    text = _block([
        ("Тип",               loc.get("type")),
        ("Адрес",             loc.get("address")),
        ("Страна",            loc.get("country")),
        ("Регион",            loc.get("region")),
        ("Город",             loc.get("city")),
        ("Значимость",        loc.get("significance")),
        ("Год на фото",       loc.get("year_of_photo_at_location")),
        ("Доп. информация",   loc.get("additional_info")),
    ])
    return text or "Информация не указана"


def truncate_caption(caption: str, limit: int = 1024) -> tuple[str, str | None]:
    """
    If caption exceeds Telegram's limit, return (truncated_caption, overflow_text).
    Otherwise return (caption, None).
    """
    if len(caption) <= limit:
        return caption, None
    return caption[:limit - 3] + "...", caption
