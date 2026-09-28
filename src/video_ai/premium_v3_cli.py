from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import MaterialRegistry, materialize_assets
from .io import load_shot_plan, save_shot_plan
from .material_brain import find_duplicate_scenes, prepare_diversity_repair
from .moments import select_moments
from .premium_audio_v2 import mix_premium_audio
from .premium_v2_fx import build_premium_overlays
from .premium_v2_renderer import render_premium_v2
from .premium_v3 import apply_callback_reuse, build_premium_v3_plan
from .premium_v3_qc import save_premium_v3_qc
from .premium_v3_sequence import apply_sequence_repairs, review_sequence
from .transcript import attach_caption_timings, load_transcript, save_transcript, transcribe_local
from .visual_diversity import assign_fingerprints


def _base(args) -> None:
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    transcript = transcribe_local(args.audio, model_size=args.model, language=args.language)
    save_transcript(transcript, work / "transcript.json")

    plan, blueprint = build_premium_v3_plan(
        transcript,
        args.audio,
        work / "premium-v3",
        use_gemini=True,
    )
    save_shot_plan(plan, work / "shot_plan.json")

    registry = MaterialRegistry()
    manifest = materialize_assets(
        plan,
        work / "assets",
        limit=30,
        semantic=False,
        semantic_top_k=6,
        meme_dir=args.meme_dir,
        registry=registry,
        allow_coverage_reuse=True,
        judge_with_gemini=False,
    )

    if args.select_moments:
        select_moments(plan, work / "moments")

    assign_fingerprints(plan)
    repairs = review_sequence(
        plan,
        blueprint,
        work / "premium-v3" / "sequence",
        use_gemini=True,
    )
    replacement_indexes = apply_sequence_repairs(plan, repairs)

    if replacement_indexes:
        materialize_assets(
            plan,
            work / "assets",
            limit=36,
            semantic=False,
            semantic_top_k=6,
            replace_scenes=set(replacement_indexes),
            rank_offset=5,
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=False,
            judge_with_gemini=False,
        )
        if args.select_moments:
            select_moments(plan, work / "moments-repair")

    duplicates, _ = find_duplicate_scenes(plan)
    if duplicates:
        prepare_diversity_repair(plan, duplicates, pass_index=9)
        materialize_assets(
            plan,
            work / "assets",
            limit=40,
            semantic=False,
            semantic_top_k=6,
            replace_scenes=set(duplicates),
            rank_offset=9,
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=False,
            judge_with_gemini=False,
        )

    callbacks = apply_callback_reuse(plan, blueprint)
    assign_fingerprints(plan)

    materialized = save_shot_plan(plan, work / "shot_plan.materialized.json")
    qc_path = save_premium_v3_qc(plan, work / "premium_v3_qc.json")

    output = render_premium_v2(
        plan,
        args.output,
        work_dir=work / "render-v3",
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
        "style": "premium-v3",
        "output": str(output),
        "plan": str(materialized),
        "qc": str(qc_path),
        "scenes": len(plan.scenes),
        "downloaded": sum(item.get("status") == "downloaded" for item in manifest),
        "sequence_repairs": replacement_indexes,
        "callbacks": callbacks,
        "gemini_design": "2 batch calls: whole-story director + whole-sequence review",
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
        mixed = render_work / "premium_v3_audio.m4a"
        mix_premium_audio(
            plan.audio,
            plan,
            mixed,
            work_dir=render_work / "audio-v3",
            music=args.music,
            sfx_dir=args.sfx_dir,
        )
        plan.audio = mixed

    overlays = []
    if args.shorts_fx:
        overlays = build_premium_overlays(
            plan,
            render_work / "premium-v3-fx",
            max_overlays=max(0, args.max_overlays),
            use_gemini=False,
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
            composition_review=False,
            base_video=args.base_video,
        )
    finally:
        plan.audio = original_audio

    print(json.dumps({
        "ok": True,
        "style": "premium-v3",
        "output": str(output),
        "overlays": len(overlays),
        "mixed_audio": str(mixed) if mixed else str(original_audio),
    }, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Premium v3 whole-story editing pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    base = sub.add_parser("base")
    base.add_argument("audio")
    base.add_argument("-o", "--output", required=True)
    base.add_argument("--work-dir", required=True)
    base.add_argument("--model", default="small")
    base.add_argument("--language", default="ru")
    base.add_argument("--meme-dir", default=None)
    base.add_argument("--reference-framing", action="store_true")
    base.add_argument("--select-moments", action="store_true")

    final = sub.add_parser("final")
    final.add_argument("plan")
    final.add_argument("-o", "--output", required=True)
    final.add_argument("--work-dir", required=True)
    final.add_argument("--transcript", default=None)
    final.add_argument("--base-video", required=True)
    final.add_argument("--reference-framing", action="store_true")
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
