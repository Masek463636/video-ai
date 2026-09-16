from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assets import materialize_assets
from .director import build_shot_plan
from .io import load_shot_plan, save_shot_plan
from .probe import probe
from .renderer import render_plan
from .transcript import load_transcript, save_transcript, transcribe_local


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

    p_plan = sub.add_parser("plan", help="Create a first-pass ShotPlan from a timed transcript")
    p_plan.add_argument("transcript")
    p_plan.add_argument("--audio", required=True)
    p_plan.add_argument("-o", "--output", required=True)
    p_plan.add_argument("--pace", type=float, default=2.1, help="Target seconds per scene")
    p_plan.add_argument("--min-scene", type=float, default=1.25)
    p_plan.add_argument("--max-scene", type=float, default=3.2)

    p_assets = sub.add_parser("assets", help="Find, rank and download visual assets for a ShotPlan")
    p_assets.add_argument("plan")
    p_assets.add_argument("-o", "--output", required=True, help="Output materialized ShotPlan JSON")
    p_assets.add_argument("--dir", required=True, help="Directory for downloaded assets")
    p_assets.add_argument("--limit", type=int, default=20, help="Commons candidates per search")
    p_assets.add_argument("--overwrite", action="store_true")

    p_render = sub.add_parser("render", help="Render a materialized ShotPlan to MP4")
    p_render.add_argument("plan")
    p_render.add_argument("-o", "--output", required=True)
    p_render.add_argument("--work-dir", default=None)
    p_render.add_argument("--no-captions", action="store_true")
    p_render.add_argument("--crf", type=int, default=20)

    p_make = sub.add_parser("make", help="Rank/download assets and render an existing ShotPlan")
    p_make.add_argument("plan")
    p_make.add_argument("-o", "--output", required=True)
    p_make.add_argument("--work-dir", required=True)
    p_make.add_argument("--limit", type=int, default=20)
    p_make.add_argument("--no-captions", action="store_true")

    p_create = sub.add_parser("create", help="One command: voiceover -> transcript -> scenes -> ranked assets -> MP4")
    p_create.add_argument("audio")
    p_create.add_argument("-o", "--output", required=True)
    p_create.add_argument("--work-dir", required=True)
    p_create.add_argument("--model", default="small")
    p_create.add_argument("--language", default=None)
    p_create.add_argument("--pace", type=float, default=2.1)
    p_create.add_argument("--min-scene", type=float, default=1.25)
    p_create.add_argument("--max-scene", type=float, default=3.2)
    p_create.add_argument("--limit", type=int, default=20)
    p_create.add_argument("--no-captions", action="store_true")

    args = parser.parse_args()

    if args.command == "probe":
        print(json.dumps(probe(args.path), ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        plan = load_shot_plan(args.path)
        duration = plan.scenes[-1].end
        print(json.dumps({
            "ok": True,
            "audio": str(plan.audio),
            "scenes": len(plan.scenes),
            "timeline_duration": duration,
            "output": f"{plan.width}x{plan.height}@{plan.fps}",
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "transcribe":
        transcript = transcribe_local(args.media, model_size=args.model, language=args.language)
        output = save_transcript(transcript, args.output)
        print(json.dumps({
            "ok": True, "output": str(output), "language": transcript.language,
            "words": len(transcript.words), "duration": round(transcript.duration, 3),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "plan":
        transcript = load_transcript(args.transcript)
        plan = build_shot_plan(
            transcript, Path(args.audio), target_scene_seconds=args.pace,
            min_scene_seconds=args.min_scene, max_scene_seconds=args.max_scene,
        )
        output = save_shot_plan(plan, args.output)
        print(json.dumps({
            "ok": True, "output": str(output), "scenes": len(plan.scenes),
            "duration": round(plan.scenes[-1].end, 3),
            "queries": [scene.query for scene in plan.scenes],
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "assets":
        plan = load_shot_plan(args.plan)
        manifest = materialize_assets(plan, args.dir, limit=args.limit, overwrite=args.overwrite)
        output = save_shot_plan(plan, args.output)
        downloaded = sum(item.get("status") == "downloaded" for item in manifest)
        print(json.dumps({
            "ok": True, "output": str(output), "downloaded": downloaded,
            "scenes": len(plan.scenes), "manifest": str(Path(args.dir) / "assets_manifest.json"),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "render":
        plan = load_shot_plan(args.plan)
        output = render_plan(
            plan, args.output, work_dir=args.work_dir,
            captions=not args.no_captions, crf=args.crf,
        )
        print(json.dumps({"ok": True, "output": str(output)}, ensure_ascii=False, indent=2))
        return

    if args.command == "make":
        plan = load_shot_plan(args.plan)
        work = Path(args.work_dir)
        manifest = materialize_assets(plan, work / "assets", limit=args.limit)
        materialized = save_shot_plan(plan, work / "shot_plan.materialized.json")
        output = render_plan(
            plan, args.output, work_dir=work / "render", captions=not args.no_captions
        )
        print(json.dumps({
            "ok": True, "output": str(output), "plan": str(materialized),
            "downloaded": sum(item.get("status") == "downloaded" for item in manifest),
        }, ensure_ascii=False, indent=2))
        return

    if args.command == "create":
        work = Path(args.work_dir)
        work.mkdir(parents=True, exist_ok=True)
        transcript = transcribe_local(args.audio, model_size=args.model, language=args.language)
        transcript_path = save_transcript(transcript, work / "transcript.json")
        plan = build_shot_plan(
            transcript, Path(args.audio), target_scene_seconds=args.pace,
            min_scene_seconds=args.min_scene, max_scene_seconds=args.max_scene,
        )
        save_shot_plan(plan, work / "shot_plan.json")
        manifest = materialize_assets(plan, work / "assets", limit=args.limit)
        materialized = save_shot_plan(plan, work / "shot_plan.materialized.json")
        output = render_plan(
            plan, args.output, work_dir=work / "render", captions=not args.no_captions
        )
        print(json.dumps({
            "ok": True,
            "output": str(output),
            "transcript": str(transcript_path),
            "plan": str(materialized),
            "scenes": len(plan.scenes),
            "downloaded": sum(item.get("status") == "downloaded" for item in manifest),
        }, ensure_ascii=False, indent=2))
        return


if __name__ == "__main__":
    main()
