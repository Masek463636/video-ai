from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from .assets import MaterialRegistry, materialize_assets
from .broll_planner import build_donor_shot_plan
from .cli import _resolve_and_repair
from .io import load_shot_plan, save_shot_plan
from .material_brain import find_duplicate_scenes, prepare_diversity_repair
from .material_v2 import apply_material_brain_v2
from .moments import select_moments
from .premium_audio_v2 import mix_premium_audio
from .premium_pacing import apply_premium_pacing
from .premium_v2 import prepare_premium_v2_plan
from .premium_v2_fx import build_premium_overlays
from .premium_v2_renderer import render_premium_v2
from .qc import inspect_plan, save_qc
from .transcript import attach_caption_timings, load_transcript, save_transcript, transcribe_local
from .viral_style import apply_viral_motion
from .visual_diversity import assign_fingerprints, diversify_queries, soft_diversity_repair_indexes


def _quality_args(meme_dir: str | None, material_v2: bool) -> SimpleNamespace:
    return SimpleNamespace(
        limit=20,
        semantic=False,
        semantic_top_k=6,
        repair_passes=2,
        diversity_passes=3,
        meme_dir=meme_dir,
        material_v2=material_v2,
        material_v2_local=False,
        select_moments=True,
        premium_style=True,
    )


def _semantic_diversity_pass(plan, work: Path, args) -> list[int]:
    repairs = soft_diversity_repair_indexes(plan)
    if not repairs:
        return []

    print(f"[premium-v2] semantic visual diversity repair: {repairs}", flush=True)
    diversify_queries(plan, repairs)
    prepare_diversity_repair(plan, repairs, pass_index=6)

    materialize_assets(
        plan,
        work / "assets",
        limit=max(args.limit, 32),
        semantic=args.semantic,
        semantic_top_k=max(args.semantic_top_k, 8),
        replace_scenes=set(repairs),
        rank_offset=6,
        meme_dir=args.meme_dir,
        registry=MaterialRegistry(),
        allow_coverage_reuse=False,
        judge_with_gemini=True,
    )

    duplicates, _ = find_duplicate_scenes(plan)
    if duplicates:
        prepare_diversity_repair(plan, duplicates, pass_index=7)
        materialize_assets(
            plan,
            work / "assets",
            limit=max(args.limit, 40),
            semantic=args.semantic,
            semantic_top_k=max(args.semantic_top_k, 10),
            replace_scenes=set(duplicates),
            rank_offset=7,
            meme_dir=args.meme_dir,
            registry=MaterialRegistry(),
            allow_coverage_reuse=False,
            judge_with_gemini=False,
        )
        remaining, _ = find_duplicate_scenes(plan)
        if remaining:
            raise RuntimeError(
                "Premium v2 diversity gate refused exact duplicates: "
                + str(remaining)
            )

    assign_fingerprints(plan)
    return repairs


def _base(args) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    transcript = transcribe_local(
        args.audio,
        model_size=args.model,
        language=args.language,
    )
    save_transcript(transcript, work / "transcript.json")

    plan = build_donor_shot_plan(
        transcript,
        Path(args.audio),
        meme_dir=args.meme_dir,
        viral_style=True,
    )

    apply_premium_pacing(plan)
    apply_viral_motion(plan)
    save_shot_plan(plan, work / "shot_plan.json")

    quality = _quality_args(args.meme_dir, bool(args.material_v2))
    if args.material_v2:
        apply_material_brain_v2(plan)

    manifest, qc_results, diversity_repairs = _resolve_and_repair(
        plan,
        work,
        quality,
    )

    semantic_repairs = _semantic_diversity_pass(plan, work, quality)

    if args.select_moments:
        select_moments(plan, work / "moments")

    assign_fingerprints(plan)
    prepare_premium_v2_plan(
        plan,
        work / "premium-v2",
        use_gemini=True,
    )

    materialized = save_shot_plan(
        plan,
        work / "shot_plan.materialized.json",
    )
    qc_results = inspect_plan(plan)
    qc_path = save_qc(qc_results, work / "qc.json")

    output = render_premium_v2(
        plan,
        args.output,
        work_dir=work / "render-v2",
        captions=False,
        crf=20,
        editing_polish=True,
        overlays=[],
        reference_framing=bool(args.reference_framing),
        composition_review=False,
        base_video=None,
    )

    print(json.dumps({
        "ok": True,
        "style": "premium-v2",
        "output": str(output),
        "plan": str(materialized),
        "qc": str(qc_path),
        "scenes": len(plan.scenes),
        "downloaded": sum(item.get("status") == "downloaded" for item in manifest),
        "diversity_repairs": diversity_repairs,
        "semantic_diversity_repairs": semantic_repairs,
    }, ensure_ascii=False, indent=2))


def _final(args) -> None:
    plan = load_shot_plan(args.plan)
    if args.transcript:
        attach_caption_timings(plan, load_transcript(args.transcript))

    render_work = Path(args.work_dir)
    render_work.mkdir(parents=True, exist_ok=True)
    original_audio = plan.audio
    mixed = None

    if args.premium_audio:
        mixed = render_work / "premium_v2_audio.m4a"
        mix_premium_audio(
            plan.audio,
            plan,
            mixed,
            work_dir=render_work / "audio-v2",
            music=args.music,
            sfx_dir=args.sfx_dir,
        )
        plan.audio = mixed

    overlays = []
    if args.shorts_fx:
        overlays = build_premium_overlays(
            plan,
            render_work / "premium-v2-fx",
            max_overlays=max(0, args.max_overlays),
            use_gemini=True,
            sticker_dir=args.sticker_dir,
            base_video=args.base_video,
        )

    try:
        output = render_premium_v2(
            plan,
            args.output,
            work_dir=render_work,
            captions=True,
            crf=args.crf,
            editing_polish=True,
            overlays=overlays,
            reference_framing=bool(args.reference_framing),
            composition_review=bool(args.composition_review),
            base_video=args.base_video,
        )
    finally:
        plan.audio = original_audio

    print(json.dumps({
        "ok": True,
        "style": "premium-v2",
        "output": str(output),
        "overlays": len(overlays),
        "mixed_audio": str(mixed) if mixed else str(original_audio),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Isolated Viral Premium v2 pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base")
    base.add_argument("audio")
    base.add_argument("-o", "--output", required=True)
    base.add_argument("--work-dir", required=True)
    base.add_argument("--model", default="small")
    base.add_argument("--language", default="ru")
    base.add_argument("--meme-dir", default=None)
    base.add_argument("--material-v2", action="store_true")
    base.add_argument("--reference-framing", action="store_true")
    base.add_argument("--select-moments", action="store_true")
    base.add_argument("--no-sfx", action="store_true")
    base.add_argument("--no-captions", action="store_true")
    base.add_argument("--viral-style", action="store_true")
    base.add_argument("--premium-style", action="store_true")

    final = sub.add_parser("final")
    final.add_argument("plan")
    final.add_argument("-o", "--output", required=True)
    final.add_argument("--work-dir", required=True)
    final.add_argument("--transcript", default=None)
    final.add_argument("--base-video", required=True)
    final.add_argument("--editing-style", choices=("premium",), default="premium")
    final.add_argument("--editing-polish", action="store_true")
    final.add_argument("--reference-framing", action="store_true")
    final.add_argument("--composition-review", action="store_true")
    final.add_argument("--premium-audio", action="store_true")
    final.add_argument("--music", default=None)
    final.add_argument("--sfx-dir", default=None)
    final.add_argument("--shorts-fx", action="store_true")
    final.add_argument("--sticker-dir", default=None)
    final.add_argument("--max-overlays", type=int, default=0)
    final.add_argument("--crf", type=int, default=20)

    args = parser.parse_args()
    if args.command == "base":
        _base(args)
    else:
        _final(args)


if __name__ == "__main__":
    main()
