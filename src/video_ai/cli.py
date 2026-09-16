from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import materialize_assets
from .audio_mix import mix_audio
from .audio_plan import build_audio_plan, save_audio_plan
from .director import build_shot_plan
from .io import load_shot_plan, save_shot_plan
from .probe import probe
from .qc import failed_scene_indexes, inspect_plan, save_qc
from .renderer import render_plan
from .transcript import load_transcript, save_transcript, transcribe_local


def _add_quality_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--limit", type=int, default=20, help="Candidates per search provider")
    parser.add_argument("--semantic", action="store_true", help="Use optional local CLIP text-image reranking")
    parser.add_argument("--semantic-top-k", type=int, default=6)
    parser.add_argument("--repair-passes", type=int, default=2, help="QC-driven asset replacement passes")
    parser.add_argument("--meme-dir", default=None, help="Optional local folder of meme/reaction images and videos")


def _add_audio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--music", default=None, help="Optional background music file")
    parser.add_argument("--sfx-dir", default=None, help="Optional folder with whoosh.wav, impact.wav, etc")
    parser.add_argument("--no-sfx", action="store_true", help="Keep voiceover unchanged")


def _resolve_and_repair(plan, work: Path, args) -> tuple[list[dict], list]:
    manifest = materialize_assets(
        plan, work / "assets", limit=args.limit, semantic=args.semantic,
        semantic_top_k=args.semantic_top_k, meme_dir=args.meme_dir,
    )
    qc_results = inspect_plan(plan)
    for pass_index in range(max(0, args.repair_passes)):
        failed = failed_scene_indexes(qc_results)
        if not failed:
            break
        manifest = materialize_assets(
            plan, work / "assets", limit=args.limit, semantic=args.semantic,
            semantic_top_k=args.semantic_top_k, replace_scenes=failed,
            rank_offset=pass_index + 1, meme_dir=args.meme_dir,
        )
        qc_results = inspect_plan(plan)
    return manifest, qc_results


def _mix_if_requested(plan, audio_plan, work: Path, args) -> Path:
    if args.no_sfx and not args.music:
        return Path(plan.audio)
    mixed = work / "mixed_audio.m4a"
    return mix_audio(plan.audio, audio_plan, mixed, work_dir=work / "audio_mix", music=args.music, sfx_dir=args.sfx_dir)


def _gemini_manifest_stats(manifest: list[dict]) -> dict:
    judged = accepted = rejected = fallback_after_rejections = 0
    for item in manifest:
        judge = item.get("gemini_judge")
        if isinstance(judge, dict):
            judged += 1
            if judge.get("accept") is True:
                accepted += 1
            if judge.get("fallback_after_rejections"):
                fallback_after_rejections += 1
        rejected += len(item.get("gemini_rejected") or [])
    return {
        "gemini_judged_scenes": judged,
        "gemini_accepted_scenes": accepted,
        "gemini_rejected_candidates": rejected,
        "gemini_fallback_scenes": fallback_after_rejections,
    }


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Inspect media metadata"); p_probe.add_argument("path")
    p_validate = sub.add_parser("validate", help="Validate a ShotPlan JSON"); p_validate.add_argument("path")

    p_transcribe = sub.add_parser("transcribe", help="Transcribe audio/video locally with faster-whisper")
    p_transcribe.add_argument("media"); p_transcribe.add_argument("-o", "--output", required=True)
    p_transcribe.add_argument("--model", default="small"); p_transcribe.add_argument("--language", default=None)

    p_plan = sub.add_parser("plan", help="Create a structured ShotPlan from a timed transcript")
    p_plan.add_argument("transcript"); p_plan.add_argument("--audio", required=True); p_plan.add_argument("-o", "--output", required=True)
    p_plan.add_argument("--pace", type=float, default=1.7); p_plan.add_argument("--min-scene", type=float, default=0.95); p_plan.add_argument("--max-scene", type=float, default=2.7)
    p_plan.add_argument("--meme-dir", default=None)

    p_assets = sub.add_parser("assets", help="Find, rank and download images/videos/memes for a ShotPlan")
    p_assets.add_argument("plan"); p_assets.add_argument("-o", "--output", required=True); p_assets.add_argument("--dir", required=True)
    p_assets.add_argument("--limit", type=int, default=20); p_assets.add_argument("--overwrite", action="store_true")
    p_assets.add_argument("--semantic", action="store_true"); p_assets.add_argument("--semantic-top-k", type=int, default=6)
    p_assets.add_argument("--meme-dir", default=None)

    p_audio = sub.add_parser("audio-plan", help="Plan music mood and SFX cues from a ShotPlan"); p_audio.add_argument("plan"); p_audio.add_argument("-o", "--output", required=True)
    p_qc = sub.add_parser("qc", help="Run local quality checks on a materialized ShotPlan"); p_qc.add_argument("plan"); p_qc.add_argument("-o", "--output", required=True)

    p_render = sub.add_parser("render", help="Render a materialized ShotPlan to MP4")
    p_render.add_argument("plan"); p_render.add_argument("-o", "--output", required=True); p_render.add_argument("--work-dir", default=None); p_render.add_argument("--no-captions", action="store_true"); p_render.add_argument("--crf", type=int, default=20)

    p_make = sub.add_parser("make", help="Resolve visual assets and render an existing ShotPlan")
    p_make.add_argument("plan"); p_make.add_argument("-o", "--output", required=True); p_make.add_argument("--work-dir", required=True); p_make.add_argument("--no-captions", action="store_true")
    _add_quality_args(p_make); _add_audio_args(p_make)

    p_create = sub.add_parser("create", help="voiceover -> Editing Brain -> sources -> judge -> render")
    p_create.add_argument("audio"); p_create.add_argument("-o", "--output", required=True); p_create.add_argument("--work-dir", required=True)
    p_create.add_argument("--model", default="small"); p_create.add_argument("--language", default=None); p_create.add_argument("--pace", type=float, default=1.7)
    p_create.add_argument("--min-scene", type=float, default=0.95); p_create.add_argument("--max-scene", type=float, default=2.7); p_create.add_argument("--no-captions", action="store_true")
    _add_quality_args(p_create); _add_audio_args(p_create)

    args = parser.parse_args()

    if args.command == "probe":
        print(json.dumps(probe(args.path), ensure_ascii=False, indent=2)); return
    if args.command == "validate":
        plan=load_shot_plan(args.path); print(json.dumps({"ok":True,"audio":str(plan.audio),"scenes":len(plan.scenes),"timeline_duration":plan.scenes[-1].end,"output":f"{plan.width}x{plan.height}@{plan.fps}","director_source":plan.director_source,"director_model":plan.director_model},ensure_ascii=False,indent=2)); return
    if args.command == "transcribe":
        transcript=transcribe_local(args.media,model_size=args.model,language=args.language); output=save_transcript(transcript,args.output)
        print(json.dumps({"ok":True,"output":str(output),"language":transcript.language,"words":len(transcript.words),"duration":round(transcript.duration,3)},ensure_ascii=False,indent=2)); return
    if args.command == "plan":
        transcript=load_transcript(args.transcript)
        plan=build_shot_plan(transcript,Path(args.audio),target_scene_seconds=args.pace,min_scene_seconds=args.min_scene,max_scene_seconds=args.max_scene,meme_dir=args.meme_dir)
        output=save_shot_plan(plan,args.output)
        print(json.dumps({"ok":True,"output":str(output),"scenes":len(plan.scenes),"duration":round(plan.scenes[-1].end,3),"director_source":plan.director_source,"director_model":plan.director_model,"visuals":[{"mode":s.visual_mode,"source":s.source_mode,"motion":s.motion_preset,"meme":s.meme_filename,"description":s.visual_description,"queries":s.search_queries} for s in plan.scenes]},ensure_ascii=False,indent=2)); return
    if args.command == "assets":
        plan=load_shot_plan(args.plan); manifest=materialize_assets(plan,args.dir,limit=args.limit,overwrite=args.overwrite,semantic=args.semantic,semantic_top_k=args.semantic_top_k,meme_dir=args.meme_dir); output=save_shot_plan(plan,args.output)
        payload={"ok":True,"output":str(output),"downloaded":sum(i.get("status")=="downloaded" for i in manifest),"scenes":len(plan.scenes),"manifest":str(Path(args.dir)/"assets_manifest.json"),"director_source":plan.director_source,"director_model":plan.director_model}
        payload.update(_gemini_manifest_stats(manifest)); print(json.dumps(payload,ensure_ascii=False,indent=2)); return
    if args.command == "audio-plan":
        plan=load_shot_plan(args.plan); audio_plan=build_audio_plan(plan); output=save_audio_plan(audio_plan,args.output); print(json.dumps({"ok":True,"output":str(output),"mood":audio_plan.mood,"cues":len(audio_plan.cues)},ensure_ascii=False,indent=2)); return
    if args.command == "qc":
        plan=load_shot_plan(args.plan); results=inspect_plan(plan); output=save_qc(results,args.output); print(json.dumps({"ok":all(i.ok for i in results),"output":str(output),"average_score":round(sum(i.score for i in results)/max(1,len(results)),3),"failed_scenes":sorted(failed_scene_indexes(results))},ensure_ascii=False,indent=2)); return
    if args.command == "render":
        plan=load_shot_plan(args.plan); output=render_plan(plan,args.output,work_dir=args.work_dir,captions=not args.no_captions,crf=args.crf); print(json.dumps({"ok":True,"output":str(output)},ensure_ascii=False,indent=2)); return

    if args.command in {"make","create"}:
        work=Path(args.work_dir); work.mkdir(parents=True,exist_ok=True)
        if args.command=="create":
            transcript=transcribe_local(args.audio,model_size=args.model,language=args.language); transcript_path=save_transcript(transcript,work/"transcript.json")
            plan=build_shot_plan(transcript,Path(args.audio),target_scene_seconds=args.pace,min_scene_seconds=args.min_scene,max_scene_seconds=args.max_scene,meme_dir=args.meme_dir); save_shot_plan(plan,work/"shot_plan.json")
        else:
            transcript_path=None; plan=load_shot_plan(args.plan)
        manifest,qc_results=_resolve_and_repair(plan,work,args); materialized=save_shot_plan(plan,work/"shot_plan.materialized.json")
        audio_plan=build_audio_plan(plan); audio_plan_path=save_audio_plan(audio_plan,work/"audio_plan.json"); qc_path=save_qc(qc_results,work/"qc.json")
        original_audio=plan.audio; mixed_audio=_mix_if_requested(plan,audio_plan,work,args); plan.audio=mixed_audio
        try: output=render_plan(plan,args.output,work_dir=work/"render",captions=not args.no_captions)
        finally: plan.audio=original_audio
        payload={
            "ok":True,"output":str(output),"plan":str(materialized),"audio_plan":str(audio_plan_path),"mixed_audio":str(mixed_audio),"qc":str(qc_path),
            "scenes":len(plan.scenes),"downloaded":sum(i.get("status")=="downloaded" for i in manifest),"qc_failed":sorted(failed_scene_indexes(qc_results)),"semantic":bool(args.semantic),
            "visual_kinds":[s.asset_kind for s in plan.scenes],"visual_modes":[s.visual_mode for s in plan.scenes],"source_modes":[s.source_mode for s in plan.scenes],"motion_presets":[s.motion_preset for s in plan.scenes],"meme_files":[s.meme_filename for s in plan.scenes],
            "director_source":plan.director_source,"director_model":plan.director_model,
        }
        payload.update(_gemini_manifest_stats(manifest))
        if transcript_path is not None: payload["transcript"]=str(transcript_path)
        print(json.dumps(payload,ensure_ascii=False,indent=2)); return


if __name__ == "__main__":
    main()
