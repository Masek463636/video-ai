from __future__ import annotations

import argparse
import json
from pathlib import Path

from .director import build_shot_plan
from .io import load_shot_plan, save_shot_plan
from .probe import probe
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

    args = parser.parse_args()

    if args.command == "probe":
        print(json.dumps(probe(args.path), ensure_ascii=False, indent=2))
        return

    if args.command == "validate":
        plan = load_shot_plan(args.path)
        duration = plan.scenes[-1].end
        print(
            json.dumps(
                {
                    "ok": True,
                    "audio": str(plan.audio),
                    "scenes": len(plan.scenes),
                    "timeline_duration": duration,
                    "output": f"{plan.width}x{plan.height}@{plan.fps}",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "transcribe":
        transcript = transcribe_local(args.media, model_size=args.model, language=args.language)
        output = save_transcript(transcript, args.output)
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(output),
                    "language": transcript.language,
                    "words": len(transcript.words),
                    "duration": round(transcript.duration, 3),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if args.command == "plan":
        transcript = load_transcript(args.transcript)
        plan = build_shot_plan(
            transcript,
            Path(args.audio),
            target_scene_seconds=args.pace,
            min_scene_seconds=args.min_scene,
            max_scene_seconds=args.max_scene,
        )
        output = save_shot_plan(plan, args.output)
        print(
            json.dumps(
                {
                    "ok": True,
                    "output": str(output),
                    "scenes": len(plan.scenes),
                    "duration": round(plan.scenes[-1].end, 3),
                    "queries": [scene.query for scene in plan.scenes],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return


if __name__ == "__main__":
    main()
