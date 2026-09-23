from __future__ import annotations

import re

from .models import Scene, ShotPlan


# Material Brain 2.0 is intentionally deterministic. Gemini may improve the
# storyboard later, but the base retrieval strategy should still ask stock
# providers for visible ACTIONS rather than abstract nouns/static concepts.

_RULES: list[tuple[tuple[str, ...], list[str]]] = [
    (
        ("молок", "milk", "carton", "dairy"),
        [
            "person taking milk carton from shelf",
            "hand holding milk carton grocery store",
            "customer checking milk label",
        ],
    ),
    (
        ("упаков", "package", "packaging", "label"),
        [
            "hands holding product package",
            "person reading package label",
            "food packaging close up hands",
        ],
    ),
    (
        ("корзин", "тележ", "cart", "basket"),
        [
            "shopper pushing shopping cart",
            "person putting groceries in cart",
            "shopping cart moving supermarket",
        ],
    ),
    (
        ("цен", "дороже", "подорож", "price", "expensive", "cost"),
        [
            "customer checking price tag",
            "hand comparing grocery prices",
            "shopper looking at receipt",
        ],
    ),
    (
        ("нагл", "обман", "скрыва", "хитр", "deceptive", "trick", "shrinkflation"),
        [
            "customer comparing two product packages",
            "person reading package suspiciously",
            "hands comparing product sizes grocery store",
        ],
    ),
    (
        ("деньг", "монет", "налич", "money", "cash", "coins"),
        [
            "hand counting cash close up",
            "person paying at checkout",
            "coins in hand close up",
        ],
    ),
    (
        ("телефон", "смартф", "phone", "smartphone"),
        [
            "person using smartphone close up",
            "hand scrolling phone screen",
            "person checking phone reaction",
        ],
    ),
    (
        ("машин", "авто", "car", "vehicle"),
        [
            "person driving car interior",
            "car moving street close shot",
            "hand opening car door",
        ],
    ),
    (
        ("производител", "фабрик", "завод", "manufacturer", "factory", "production"),
        [
            "factory packaging products production line",
            "workers packing products factory",
            "production line moving packages",
        ],
    ),
    (
        ("собак", "пёс", "dog"),
        [
            "dog reacting to owner",
            "dog moving indoors close up",
            "dog looking around alert",
        ],
    ),
    (
        ("кот", "кошк", "cat"),
        [
            "cat reacting indoors close up",
            "cat walking room",
            "cat looking at owner",
        ],
    ),
]


def apply_material_brain_v2(plan: ShotPlan) -> list[int]:
    """Bias unlocked beats toward living action B-roll.

    Historical/factual semantic locks remain archive images. Local memes remain
    local memes. Everything else becomes stock-video-first with concrete action
    queries and a clear moving subject.
    """
    changed: list[int] = []

    for index, scene in enumerate(plan.scenes):
        if scene.semantic_lock or scene.source_mode == "historical_archive":
            continue
        if scene.visual_mode == "meme" and scene.source_mode == "meme_library":
            continue

        scene.visual_mode = "video"
        scene.source_mode = "stock_video"
        scene.motion = "none"
        scene.motion_preset = "none"

        action_queries = _action_queries(scene)
        merged = _merge_queries(action_queries, scene.search_queries or [], [scene.query])
        scene.search_queries = merged[:9]
        if scene.search_queries:
            scene.query = scene.search_queries[0]

        base = re.sub(
            r"\s+",
            " ",
            (scene.visual_description or scene.caption or scene.query or "real moving subject"),
        ).strip(" .")
        scene.visual_description = (
            f"{base}. Show a clear moving subject performing a visible action. "
            "Prefer real human/animal/hands/object interaction footage over static objects, "
            "posters, diagrams or generic establishing shots. The main subject must remain "
            "easy to understand in a vertical Short."
        )[:700]
        changed.append(index)

    return changed


def _action_queries(scene: Scene) -> list[str]:
    text = " ".join(
        [
            scene.caption or "",
            scene.visual_description or "",
            scene.query or "",
            *(scene.search_queries or []),
        ]
    ).casefold()

    out: list[str] = []
    for needles, queries in _RULES:
        if any(needle in text for needle in needles):
            out.extend(queries)

    # Existing English queries are valuable, but turn noun-only B-roll searches
    # into visible actions/subject interaction.
    bases = []
    for raw in [scene.query, *(scene.search_queries or [])]:
        cleaned = _clean_query(raw)
        if cleaned and cleaned.casefold() not in {x.casefold() for x in bases}:
            bases.append(cleaned)
        if len(bases) >= 3:
            break

    for base in bases:
        if _already_action_query(base):
            out.append(base)
        else:
            out.extend(
                [
                    f"person interacting with {base}",
                    f"hands using {base} close up",
                    f"{base} real action footage",
                ]
            )

    if not out:
        subject = _clean_query(scene.visual_description or scene.query or scene.caption or "")
        if subject:
            out.extend(
                [
                    f"person {subject} action",
                    f"{subject} hands close up",
                    f"{subject} real footage",
                ]
            )

    return _unique(out)


def _already_action_query(value: str) -> bool:
    tokens = value.casefold()
    verbs = (
        "taking", "holding", "checking", "pushing", "putting", "moving",
        "reading", "using", "paying", "counting", "walking", "running",
        "interacting", "comparing",
        "driving", "opening", "packing", "reacting", "looking", "working",
        "eating", "drinking", "writing", "talking", "playing",
    )
    return any(verb in tokens for verb in verbs)


def _clean_query(value: str) -> str:
    value = re.sub(r"\s+", " ", str(value or "")).strip(" ,.;:-")
    # Remove retrieval boilerplate that often returns generic/static stock.
    value = re.sub(
        r"\b(real footage|documentary footage|vertical b roll|b roll|stock footage|photo illustration|visual metaphor|visual concept|metaphor)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    return re.sub(r"\s+", " ", value).strip()[:120]


def _unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        value = _clean_query(raw)
        key = value.casefold()
        if value and key not in seen:
            seen.add(key)
            out.append(value)
    return out


def _merge_queries(*groups: list[str]) -> list[str]:
    return _unique([value for group in groups for value in group])
