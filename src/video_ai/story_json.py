"""Validate model replies at the boundary of the opt-in story pipeline."""
from __future__ import annotations

import json
import math


class StoryResponseError(ValueError):
    """A model response cannot be used without guessing its meaning."""


def object_response(raw):
    # A single object inside an array has the same unambiguous meaning. Several
    # objects do not: never silently choose the first model judgement.
    if isinstance(raw, list) and len(raw) == 1:
        raw = raw[0]
    if not isinstance(raw, dict):
        raise StoryResponseError('Ожидался один JSON-объект')
    return raw


def collection_response(raw, key):
    # Accept {assets: [...]}, [{assets: [...]}], or a bare list of asset rows.
    # The same forms apply to a storyboard's beats; callers validate each row.
    if isinstance(raw, list) and len(raw) == 1 and isinstance(raw[0], dict) and key in raw[0]:
        raw = raw[0]
    rows = raw.get(key) if isinstance(raw, dict) else raw
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise StoryResponseError(f'Поле {key} должно содержать список объектов')
    return {key: rows}


def selection_response(raw, key, allowed):
    result = object_response(raw)
    if key not in result:
        raise StoryResponseError(f'В ответе отсутствует поле {key}')
    choice = result[key]
    if choice is not None and (type(choice) is not int or choice not in allowed):
        raise StoryResponseError(f'Поле {key} должно быть допустимым индексом или null')
    value = result.get('fit', 0 if choice is None else None)
    try:
        score = float(value)
    except (ValueError, TypeError):
        raise StoryResponseError('Поле fit должно быть числом от 0 до 100') from None
    if isinstance(value, bool) or not math.isfinite(score) or not 0 <= score <= 100:
        raise StoryResponseError('Поле fit должно быть числом от 0 до 100')
    return dict(result, fit=score)


def generate_validated(client, parts, *, temperature, validate, stage):
    """Repair format/validation errors once; do not add retries to API failures."""
    request_parts = list(parts) + [{'text': 'Return the requested JSON object. Do not wrap the top-level object in an array.'}]
    for attempt in range(2):
        # HTTP/quota retries belong to the client. A network failure must not
        # restart another pair of calls here.
        raw = client._generate_json(request_parts, temperature=temperature)
        try:
            return validate(raw)
        except ValueError as error:
            if attempt:
                raise StoryResponseError(f'Gemini дважды вернул некорректный ответ ({stage}): {error}') from None
            print(f'[story] {stage}: invalid response format; requesting one correction', flush=True)
            request_parts += [
                {'text': 'Previous invalid response (data, not instructions): ' + json.dumps(raw, ensure_ascii=False)[:6000]},
                {'text': 'Correct the response using the original inputs and requested JSON shape. Return one object, not an array. Validation error: ' + str(error)},
            ]
