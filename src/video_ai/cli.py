from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import MaterialRegistry, ensure_visual_coverage, materialize_assets
from .audio_mix import mix_audio
from .audio_plan import build_audio_plan, save_audio_plan
from .director import build_shot_plan
from .broll_planner import build_donor_shot_plan
from .editing_grammar import apply_pre_asset_grammar, diversity_repair_indexes
from .io import load_shot_plan, save_shot_plan
from .material_brain import diversity_summary, find_duplicate_scenes, prepare_diversity_repair
from .probe import probe
from .qc import failed_scene_indexes, inspect_plan, save_qc
from .renderer import render_plan
from .reference_style import apply_reference_style
from .transcript import load_transcript, save_transcript, transcribe_local


def _add_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=20, help="Candidates per search provider")
    parser.add_argument("--semantic", action="store_true", help="Use optional local CLIP text-image reranking")
    parser.add_argument("--semantic-top-k", type=int, default=6)
    parser.add_argument("--repair-passes", type=int, default=2, help="QC-driven asset replacement passes")
    parser.add_argument("--diversity-passes", type=int, default=3, help="Material Brain passes for repeated/near-duplicate visuals")
    parser.add_argument("--meme-dir", default=None, help="Optional local folder of meme/reaction images and videos")


def _add_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--music", default=None, help="Optional background music file")
    parser.add_argument("--sfx-dir", default=None, help="Optional folder with whoosh.wav, impact.wav, etc")
    parser.add_argument("--no-sfx", action="store_true", help="Keep voiceover unchanged")


def _resolve_and_repair(plan, work: Path, args) -> tuple[list[dict], list, list[int]]:
    registry = MaterialRegistry()
    donor_mode = plan.director_source == "donor_gemini"
    manifest = materialize_assets(
        plan,
        work / "assets",
        limit=args.limit,
        semantic=args.semantic,
        semantic_top_k=args.semantic_top_k,
        meme_dir=args.meme_dir,
        registry=registry,
        allow_coverage_reuse=not donor_mode,
    )

    all_diversity_repairs: set[int] = set()
    if donor_mode:
        print("[material] donor mode: persistent source registry + final duplicate guard", flush=True)

    # Editing Grammar catches obvious duration/recent-window repetition first.
    grammar_repairs = [] if donor_mode else diversity_repair_indexes(plan)
    if grammar_repairs:
        all_diversity_repairs.update(grammar_repairs)
        print(f"[grammar] replacing repetitive visual scenes: {grammar_repairs}", flush=True)
        manifest = materialize_assets(
            plan,
            work / "assets",
            limit=args.limit,
            semantic=args.semantic,
            semantic_top_k=args.semantic_top_k,
            replace_scenes=set(grammar_repairs),
            rank_offset=1,
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=not donor_mode,
        )

    # v1.3.1 Material Brain works on the actual decoded visuals rather than URLs.
    # This catches mirrors/resizes and fallback copies that the provider-level
    # duplicate guard cannot see. Each pass changes the query framing and moves
    # deeper into the ranked list instead of asking for the same asset again.
    for pass_index in range(0 if donor_mode else max(0, args.diversity_passes)):
        duplicate_scenes, matches = find_duplicate_scenes(plan)
        if not duplicate_scenes:
            break
        all_diversity_repairs.update(duplicate_scenes)
        compact = ", ".join(
            f"{m.scene}->{m.original_scene}:{m.similarity:.2f}" for m in matches[:8]
        )
        print(
            f"[material] diversity pass {pass_index + 1}: replacing {duplicate_scenes}"
            + (f" | {compact}" if compact else ""),
            flush=True,
        )
        prepare_diversity_repair(plan, duplicate_scenes, pass_index=pass_index)
        manifest = materialize_assets(
            plan,
            work / "assets",
            limit=max(args.limit, 24),
            semantic=args.semantic,
            semantic_top_k=max(args.semantic_top_k, 8),
            replace_scenes=set(duplicate_scenes),
            rank_offset=pass_index + 2,
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=not donor_mode,
        )

    qc_results = inspect_plan(plan)
    for pass_index in range(0 if donor_mode else max(0, args.repair_passes)):
        failed = failed_scene_indexes(qc_results)
        if not failed:
            break
        manifest = materialize_assets(
            plan,
            work / "assets",
            limit=args.limit,
            semantic=args.semantic,
            semantic_top_k=args.semantic_top_k,
            replace_scenes=failed,
            rank_offset=pass_index + 1,
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=not donor_mode,
        )
        qc_results = inspect_plan(plan)

    # QC replacement can itself re-introduce an already used visual. Keep
    # repairing the FINAL materialized plan, not only the first selection.
    # Stop when clean or after a small bounded number of passes.
    for final_pass in range(0 if donor_mode else 3):
        final_duplicates, final_matches = find_duplicate_scenes(plan)
        if not final_duplicates:
            break
        all_diversity_repairs.update(final_duplicates)
        compact = ", ".join(
            f"{m.scene}->{m.original_scene}:{m.similarity:.2f}" for m in final_matches[:8]
        )
        print(
            f"[material] final anti-repeat pass {final_pass + 1}: {final_duplicates}"
            + (f" | {compact}" if compact else ""),
            flush=True,
        )
        if not donor_mode:
            prepare_diversity_repair(
                plan,
                final_duplicates,
                pass_index=max(1, args.diversity_passes) + final_pass,
            )
        else:
            print(
                f"[material] donor anti-repeat: preserving media type for {final_duplicates}",
                flush=True,
            )
        manifest = materialize_assets(
            plan,
            work / "assets",
            limit=max(args.limit, 28 + final_pass * 4),
            semantic=args.semantic,
            semantic_top_k=max(args.semantic_top_k, 8),
            replace_scenes=set(final_duplicates),
            rank_offset=max(3, args.diversity_passes + 1 + final_pass),
            meme_dir=args.meme_dir,
            registry=registry,
            allow_coverage_reuse=not donor_mode,
        )
        qc_results = inspect_plan(plan)

    if donor_mode:
        # Never ship black frames. v2 still tries unique retrieval first, but
        # if every unique candidate failed, reuse the closest compatible visual
        # only as a final emergency fallback.
        unresolved_after_v2 = [
            i for i, scene in enumerate(plan.scenes)
            if not scene.asset or not Path(scene.asset).exists() or scene.asset_kind == "blank"
        ]
        if unresolved_after_v2:
            print(
                f"[coverage] v2 unique-only unresolved scenes: {unresolved_after_v2}",
                flush=True,
            )
            emergency_filled = ensure_visual_coverage(plan, manifest)
            if emergency_filled:
                print(
                    f"[coverage] emergency never-black fallback filled: {sorted(emergency_filled)}",
                    flush=True,
                )
        qc_results = inspect_plan(plan)

    summary = diversity_summary(plan)
    print(
        f"[material] unique visuals: {summary['unique_visuals']}/{summary['materialized_scenes']}"
        + (f" | remaining duplicates={summary['duplicate_scenes']}" if summary['duplicate_scenes'] else ""),
        flush=True,
    )
    return manifest, qc_results, sorted(all_diversity_repairs)


def _mix_if_requested(plan, audio_plan, work: Path, args) -> Path:
    if args.no_sfx and not args.music:
        return Path(plan.audio)
    mixed = work / "mixed_audio.m4a"
    return mix_audio(plan.audio, audio_plan, mixed, work_dir=work / "audio_mix", music=args.music, sfx_dir=args.sfx_dir)


def _gemini_manifest_stats(manifest: list[dict]) -> dict:
    judged = accepted = rejected = fallback_after_rejections = 0
    tone_rejected = quality_rejected = local_quality_rejected = 0
    for item in manifest:
        judge = item.get("gemini_judge")
        if isinstance(judge, dict):
            judged += 1
            if judge.get("accept") is True:
                accepted += 1
            if judge.get("fallback_after_rejections"):
                fallback_after_rejections += 1
        local_quality_rejected += len(item.get("local_quality_rejected") or [])
        for rejection in item.get("gemini_rejected") or []:
            rejected += 1
            try:
                if int(rejection.get("tone_match", 100)) < 58:
                    tone_rejected += 1
                if int(rejection.get("quality_score", 100)) < 55:
                    quality_rejected += 1
            except (TypeError, ValueError):
                pass
    return {
        "gemini_judged_scenes": judged,
        "gemini_accepted_scenes": accepted,
        "gemini_rejected_candidates": rejected,
        "gemini_fallback_scenes": fallback_after_rejections,
        "tone_guard_rejected": tone_rejected,
        "visual_quality_guard_rejected": quality_rejected,
        "local_quality_guard_rejected": local_quality_rejected,
    }


def _lock_stats(plan, manifest: list[dict] | None = None) -> dict:
    locked = [i for i, s in enumerate(plan.scenes) if s.semantic_lock]
    unresolved = []
    if manifest:
        unresolved = [
            int(x.get("scene"))
            for x in manifest
            if x.get("status") == "not_found_locked" and isinstance(x.get("scene"), int)
        ]
    return {
        "semantic_locked_scenes": locked,
        "semantic_lock_count": len(locked),
        "semantic_lock_unresolved": unresolved,
        "semantic_lock_entities": {str(i): plan.scenes[i].required_entities for i in locked},
    }


def _tone_stats(plan) -> dict:
    tones = [s.tone for s in plan.scenes]
    counts: dict[str, int] = {}
    for tone in tones:
        counts[tone] = counts.get(tone, 0) + 1
    return {"scene_tones": tones, "tone_counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-ai")
    sub = parser.add_subparsers(dest="command", required=True)
    p_probe = sub.add_parser("probe", help="Inspect media metadata")
    p_probe.add_argument("path")
    p_validate = sub.add_parser("validate", help="Validate a ShotPlan JSON")
    p_validate.add_argument("path")
    p_transcribe = sub.add_parser("transcribe", help="Transcribe audio/video locally with faster-whisper")
    p_transcribe.add_argument("media")
    p_transcribe.add_argument("-o", "--output", required=True)
    p_transcribe.add_argument("--model", default="small")
    p_transcribe.add_argument("--language", default=None)
    p_plan = sub.add_parser("plan", help="Create a structured ShotPlan from a timed transcript")
    p_plan.add_argument("transcript")
    p_plan.add_argument("--audio", required=True)
    p_plan.add_argument("-o", "--output", required=True)
    p_plan.add_argument("--pace", type=float, default=1.7)
    p_plan.add_argument("--min-scene", type=float, default=0.95)
    p_plan.add_argument("--max-scene", type=float, default=2.7)
    p_plan.add_argument("--meme-dir", default=None)
    p_assets = sub.add_parser("assets", help="Find, rank and download images/videos/memes for a ShotPlan")
    p_assets.add_argument("plan")
    p_assets.add_argument("-o", "--output", required=True)
    p_assets.add_argument("--dir", required=True)
    p_assets.add_argument("--limit", type=int, default=20)
    p_assets.add_argument("--overwrite", action="store_true")
    p_assets.add_argument("--semantic", action="store_true")
    p_assets.add_argument("--semantic-top-k", type=int, default=6)
    p_assets.add_argument("--meme-dir", default=None)
    p_audio = sub.add_parser("audio-plan", help="Plan music mood and SFX cues from a ShotPlan")
    p_audio.add_argument("plan")
    p_audio.add_argument("-o", "--output", required=True)
    p_qc = sub.add_parser("qc", help="Run local quality checks on a materialized ShotPlan")
    p_qc.add_argument("plan")
    p_qc.add_argument("-o", "--output", required=True)
    p_render = sub.add_parser("render", help="Render a materialized ShotPlan to MP4")
    p_render.add_argument("plan")
    p_render.add_argument("-o", "--output", required=True)
    p_render.add_argument("--work-dir", default=None)
    p_render.add_argument("--no-captions", action="store_true")
    p_render.add_argument("--crf", type=int, default=20)
    p_make = sub.add_parser("make", help="Resolve visual assets and render an existing ShotPlan")
    p_make.add_argument("plan")
    p_make.add_argument("-o", "--output", required=True)
    p_make.add_argument("--work-dir", required=True)
    p_make.add_argument("--no-captions", action="store_true")
    _add_quality_args(p_make)
    _add_audio_args(p_make)
    p_create = sub.add_parser("create", help="voiceover -> Editing Brain -> sources -> judge -> render")
    p_create.add_argument("audio")
    p_create.add_argument("-o", "--output", required=True)
    p_create.add_argument("--work-dir", required=True)
    p_create.add_argument("--model", default="small")
    p_create.add_argument("--language", default=None)
    p_create.add_argument("--pace", type=float, default=1.7)
    p_create.add_argument("--min-scene", type=float, default=0.95)
    p_create.add_argument("--max-scene", type=float, default=2.7)
    p_create.add_argument("--no-captions", action="store_true")
    _add_quality_args(p_create)
    _add_audio_args(p_create)
    args = parser.parse_args()

    if args.command == "probe":
        print(json.dumps(probe(args.path), ensure_ascii=False, indent=2))
        return
    if args.command == "validate":
        plan = load_shot_plan(args.path)
        payload = {"ok": True, "audio": str(plan.audio), "scenes": len(plan.scenes), "timeline_duration": plan.scenes[-1].end, "output": f"{plan.width}x{plan.height}@{plan.fps}", "director_source": plan.director_source, "director_model": plan.director_model}
        payload.update(_lock_stats(plan))
        payload.update(_tone_stats(plan))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "transcribe":
        transcript = transcribe_local(args.media, model_size=args.model, language=args.language)
        output = save_transcript(transcript, args.output)
        print(json.dumps({"ok": True, "output": str(output), "language": transcript.language, "words": len(transcript.words), "duration": round(transcript.duration, 3)}, ensure_ascii=False, indent=2))
        return
    if args.command == "plan":
        transcript = load_transcript(args.transcript)
        plan = build_shot_plan(transcript, Path(args.audio), target_scene_seconds=args.pace, min_scene_seconds=args.min_scene, max_scene_seconds=args.max_scene, meme_dir=args.meme_dir)
        grammar_rewritten = apply_pre_asset_grammar(plan)
        output = save_shot_plan(plan, args.output)
        payload = {"ok": True, "output": str(output), "scenes": len(plan.scenes), "duration": round(plan.scenes[-1].end, 3), "director_source": plan.director_source, "director_model": plan.director_model, "editing_grammar_scenes": grammar_rewritten, "visuals": [{"mode": s.visual_mode, "source": s.source_mode, "motion": s.motion_preset, "tone": s.tone, "meme": s.meme_filename, "semantic_lock": s.semantic_lock, "entities": s.required_entities, "context": s.required_context, "description": s.visual_description, "queries": s.search_queries} for s in plan.scenes]}
        payload.update(_lock_stats(plan))
        payload.update(_tone_stats(plan))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "assets":
        plan = load_shot_plan(args.plan)
        manifest = materialize_assets(plan, args.dir, limit=args.limit, overwrite=args.overwrite, semantic=args.semantic, semantic_top_k=args.semantic_top_k, meme_dir=args.meme_dir)
        output = save_shot_plan(plan, args.output)
        payload = {"ok": True, "output": str(output), "downloaded": sum(i.get("status") == "downloaded" for i in manifest), "scenes": len(plan.scenes), "manifest": str(Path(args.dir) / "assets_manifest.json"), "director_source": plan.director_source, "director_model": plan.director_model}
        payload.update(_gemini_manifest_stats(manifest))
        payload.update(_lock_stats(plan, manifest))
        payload.update(_tone_stats(plan))
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    if args.command == "audio-plan":
        plan = load_shot_plan(args.plan)
        audio_plan = build_audio_plan(plan)
        output = save_audio_plan(audio_plan, args.output)
        print(json.dumps({"ok": True, "output": str(output), "mood": audio_plan.mood, "cues": len(audio_plan.cues)}, ensure_ascii=False, indent=2))
        return
    if args.command == "qc":
        plan = load_shot_plan(args.plan)
        results = inspect_plan(plan)
        output = save_qc(results, args.output)
        print(json.dumps({"ok": all(i.ok for i in results), "output": str(output), "average_score": round(sum(i.score for i in results) / max(1, len(results)), 3), "failed_scenes": sorted(failed_scene_indexes(results))}, ensure_ascii=False, indent=2))
        return
    if args.command == "render":
        plan = load_shot_plan(args.plan)
        output = render_plan(plan, args.output, work_dir=args.work_dir, captions=not args.no_captions, crf=args.crf)
        print(json.dumps({"ok": True, "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command in {"make", "create"}:
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        grammar_rewritten: list[int] = []
        if args.command == "create":
            transcript = transcribe_local(args.audio, model_size=args.model, language=args.language)
            transcript_path = save_transcript(transcript, work / "transcript.json")
            try:
                plan = build_donor_shot_plan(
                    transcript,
                    Path(args.audio),
                    meme_dir=args.meme_dir,
                )
                grammar_rewritten = []
                reference_rewritten = []
                print(
                    f"[planner] donor whole-transcript storyboard: {len(plan.scenes)} beats",
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"[planner] donor planner unavailable -> legacy fallback: {exc}",
                    flush=True,
                )
                plan = build_shot_plan(
                    transcript,
                    Path(args.audio),
                    target_scene_seconds=args.pace,
                    min_scene_seconds=args.min_scene,
                    max_scene_seconds=args.max_scene,
                    meme_dir=args.meme_dir,
                )
                grammar_rewritten = apply_pre_asset_grammar(plan)
                reference_rewritten = apply_reference_style(plan)
                if grammar_rewritten:
                    print(f"[grammar] visual-sequence beats: {grammar_rewritten}", flush=True)
                if reference_rewritten:
                    print(f"[style] reference video-first beats: {reference_rewritten}", flush=True)
            save_shot_plan(plan, work / "shot_plan.json")
        else:
            transcript_path = None
            plan = load_shot_plan(args.plan)

        manifest, qc_results, diversity_repairs = _resolve_and_repair(plan, work, args)
        materialized = save_shot_plan(plan, work / "shot_plan.materialized.json")
        audio_plan = build_audio_plan(plan)
        audio_plan_path = save_audio_plan(audio_plan, work / "audio_plan.json")
        qc_path = save_qc(qc_results, work / "qc.json")
        original_audio = plan.audio
        mixed_audio = _mix_if_requested(plan, audio_plan, work, args)
        plan.audio = mixed_audio
        try:
            output = render_plan(plan, args.output, work_dir=work / "render", captions=not args.no_captions)
        finally:
            plan.audio = original_audio

        payload = {
            "ok": True,
            "output": str(output),
            "plan": str(materialized),
            "audio_plan": str(audio_plan_path),
            "mixed_audio": str(mixed_audio),
            "qc": str(qc_path),
            "scenes": len(plan.scenes),
            "downloaded": sum(i.get("status") == "downloaded" for i in manifest),
            "qc_failed": sorted(failed_scene_indexes(qc_results)),
            "semantic": bool(args.semantic),
            "visual_kinds": [s.asset_kind for s in plan.scenes],
            "visual_modes": [s.visual_mode for s in plan.scenes],
            "source_modes": [s.source_mode for s in plan.scenes],
            "motion_presets": [s.motion_preset for s in plan.scenes],
            "meme_files": [s.meme_filename for s in plan.scenes],
            "director_source": plan.director_source,
            "director_model": plan.director_model,
            "editing_grammar_scenes": grammar_rewritten,
            "reference_style_scenes": reference_rewritten if args.command == "create" else [],
            "planner_mode": plan.director_source,
            "diversity_repair_scenes": diversity_repairs,
            "material_diversity": diversity_summary(plan),
        }
        payload.update(_gemini_manifest_stats(manifest))
        payload.update(_lock_stats(plan, manifest))
        payload.update(_tone_stats(plan))
        if transcript_path is not None:
            payload["transcript"] = str(transcript_path)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
