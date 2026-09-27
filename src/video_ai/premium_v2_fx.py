from __future__ import annotations

import json
import math
import zlib
from pathlib import Path

from .cache import DiskCache, cache_key, file_signature


def _attention_cache_key(plan, base_video, budget, avoid_scenes) -> str:
    captions = [
        {
            "start": round(float(scene.start), 3),
            "end": round(float(scene.end), 3),
            "caption": scene.caption or "",
        }
        for scene in plan.scenes
    ]
    return cache_key(
        "attention-v2",
        file_signature(base_video),
        budget,
        sorted(int(x) for x in avoid_scenes),
        json.dumps(captions, ensure_ascii=False, sort_keys=True),
    )


def _draw_attention_asset(item: dict, plan, out: Path) -> dict | None:
    box = item.get("target_box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x, y, w, h = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in (x, y, w, h)):
        return None

    kind = str(item.get("type") or "")
    out.mkdir(parents=True, exist_ok=True)
    seed = zlib.crc32(
        f"{item.get('scene')}:{kind}:{x:.3f}:{y:.3f}:{w:.3f}:{h:.3f}".encode()
    )

    try:
        from PIL import Image, ImageDraw

        if kind == "circle":
            margin_x = max(0.025, w * 0.16)
            margin_y = max(0.020, h * 0.16)
            lx = max(0.0, x - margin_x)
            ly = max(0.0, y - margin_y)
            lw = min(1.0 - lx, w + margin_x * 2)
            lh = min(1.0 - ly, h + margin_y * 2)

            canvas_w = max(180, int(plan.width * lw))
            canvas_h = max(120, int(plan.height * lh))
            image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
            draw = ImageDraw.Draw(image)
            stroke = max(7, int(min(canvas_w, canvas_h) * 0.055))
            for offset in (0, 3, -3):
                pad = stroke + abs(offset) + 3
                draw.ellipse(
                    (pad + offset, pad - offset, canvas_w - pad + offset, canvas_h - pad - offset),
                    outline=(255, 40, 40, 235),
                    width=stroke,
                )

            path = out / f"circle_{item.get('scene','x')}_{seed}.png"
            image.save(path)
            result = dict(item)
            result.update(
                type="png",
                asset=str(path.resolve()),
                layout_box=[lx, ly, lw, lh],
                animation="pop",
                size="hero",
                position="center",
                source="premium_attention_graphic",
                attention_kind="circle",
            )
            return result

        cx = x + w / 2
        cy = y + h / 2
        if cx < 0.35:
            lx = min(0.82, x + w + 0.02); ly = max(0.02, cy - 0.09); lw, lh = min(0.22, 0.98 - lx), 0.18; direction = "left"
        elif cx > 0.65:
            lx = max(0.02, x - 0.24); ly = max(0.02, cy - 0.09); lw, lh = 0.22, 0.18; direction = "right"
        elif cy < 0.36:
            lx = max(0.02, cx - 0.09); ly = min(0.78, y + h + 0.02); lw, lh = 0.18, min(0.22, 0.98 - ly); direction = "up"
        else:
            lx = max(0.02, cx - 0.09); ly = max(0.02, y - 0.24); lw, lh = 0.18, 0.22; direction = "down"

        canvas_w = max(160, int(plan.width * lw))
        canvas_h = max(160, int(plan.height * lh))
        image = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)
        stroke = max(8, int(min(canvas_w, canvas_h) * 0.055))

        if direction == "right":
            start = (int(canvas_w * 0.12), int(canvas_h * 0.50)); end = (int(canvas_w * 0.84), int(canvas_h * 0.50))
            head = [end,(int(canvas_w * 0.67), int(canvas_h * 0.30)),(int(canvas_w * 0.67), int(canvas_h * 0.70))]
        elif direction == "left":
            start = (int(canvas_w * 0.88), int(canvas_h * 0.50)); end = (int(canvas_w * 0.16), int(canvas_h * 0.50))
            head = [end,(int(canvas_w * 0.33), int(canvas_h * 0.30)),(int(canvas_w * 0.33), int(canvas_h * 0.70))]
        elif direction == "up":
            start = (int(canvas_w * 0.50), int(canvas_h * 0.88)); end = (int(canvas_w * 0.50), int(canvas_h * 0.16))
            head = [end,(int(canvas_w * 0.30), int(canvas_h * 0.33)),(int(canvas_w * 0.70), int(canvas_h * 0.33))]
        else:
            start = (int(canvas_w * 0.50), int(canvas_h * 0.12)); end = (int(canvas_w * 0.50), int(canvas_h * 0.84))
            head = [end,(int(canvas_w * 0.30), int(canvas_h * 0.67)),(int(canvas_w * 0.70), int(canvas_h * 0.67))]

        jitter = (-3, 0, 3)[seed % 3]
        for dx, dy, alpha in ((0, 0, 245), (jitter, -jitter, 170)):
            draw.line((start[0]+dx,start[1]+dy,end[0]+dx,end[1]+dy),fill=(255,35,35,alpha),width=stroke)
        draw.polygon(head, fill=(255, 35, 35, 245))

        path = out / f"arrow_{item.get('scene','x')}_{seed}.png"
        image.save(path)
        result = dict(item)
        result.update(
            type="png",
            asset=str(path.resolve()),
            layout_box=[lx, ly, lw, lh],
            animation="pop",
            size="hero",
            position="center",
            source="premium_attention_graphic",
            attention_kind="arrow",
        )
        return result
    except Exception as exc:
        print(f"[premium-v2] attention graphic fallback: {type(exc).__name__}", flush=True)
        return None


def _overlay_class(item: dict) -> str:
    if item.get("reaction_verified") is True or item.get("source") in {"local_reaction","giphy"}:
        return "reaction"
    kind = str(item.get("type") or "")
    if kind in {"arrow","circle"} and item.get("attention_verified") is True:
        return kind
    if kind in {"png","png_text","text"}:
        return "object"
    return "other"


def _allowed_for_scene(layout: str, overlay_class: str) -> bool:
    if layout == "reaction":
        return overlay_class == "reaction"
    if layout in {"arrow","circle"}:
        return overlay_class == layout
    if layout == "png_cutout":
        return overlay_class == "object"
    if layout in {
        "big_number","text_behind","parallax","split_screen","before_after",
        "freeze_frame","spotlight","blur_background","stacked_cards","focus_zoom",
    }:
        return False
    return True


def build_premium_overlays(
    plan,
    out_dir,
    *,
    max_overlays=0,
    use_gemini=True,
    sticker_dir=None,
    base_video=None,
):
    from . import viral_fx

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = DiskCache(out / "cache")
    original_attention = viral_fx._attention_accents

    def cached_attention(plan_arg, attention_out, *, base_video, budget, use_gemini, avoid_scenes):
        if not base_video:
            return []
        key = _attention_cache_key(plan_arg, base_video, budget, avoid_scenes)
        cached = cache.get_json(key)
        if isinstance(cached, dict) and isinstance(cached.get("items"), list):
            print("[premium-v2] attention cache hit", flush=True)
            return cached["items"]
        result = original_attention(
            plan_arg,
            attention_out,
            base_video=base_video,
            budget=budget,
            use_gemini=use_gemini,
            avoid_scenes=avoid_scenes,
        )
        cache.set_json(key, {"items": result})
        return result

    viral_fx._attention_accents = cached_attention
    try:
        raw = viral_fx.build_viral_overlays(
            plan,
            out / "viral",
            max_overlays=max_overlays,
            use_gemini=use_gemini,
            sticker_dir=sticker_dir,
            base_video=base_video,
        )
    finally:
        viral_fx._attention_accents = original_attention

    accepted: list[dict] = []
    last_hero_scene = -99
    used_scenes: set[int] = set()

    for item in sorted(
        [row for row in raw if isinstance(row, dict)],
        key=lambda row: (float(row.get("start", 0.0)), int(row.get("scene", 10**6)) if type(row.get("scene")) is int else 10**6),
    ):
        scene_index = item.get("scene")
        if type(scene_index) is not int or not 0 <= scene_index < len(plan.scenes):
            continue
        if scene_index in used_scenes:
            continue

        overlay_class = _overlay_class(item)
        layout = str(plan.scenes[scene_index].premium_layout or "clean")
        if not _allowed_for_scene(layout, overlay_class):
            continue

        if overlay_class in {"reaction","arrow","circle","object"}:
            if scene_index - last_hero_scene < 3:
                continue
            last_hero_scene = scene_index

        if overlay_class in {"arrow","circle"}:
            graphic = _draw_attention_asset(item, plan, out / "graphics")
            if graphic is None:
                continue
            item = graphic

        accepted.append(item)
        used_scenes.add(scene_index)

    report = {"style":"premium-v2","raw":len(raw),"accepted":len(accepted),"overlays":accepted}
    (out / "premium_v2_overlays.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"[premium-v2] overlays: {len(raw)} proposed -> {len(accepted)} accepted", flush=True)
    return accepted
