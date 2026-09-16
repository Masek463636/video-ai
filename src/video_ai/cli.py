from __future__ import annotations

import argparse
import json

from .io import load_shot_plan
from .probe import probe


def main() -> None:
    parser = argparse.ArgumentParser(prog="video-ai")
    sub = parser.add_subparsers(dest="command", required=True)

    p_probe = sub.add_parser("probe", help="Inspect media metadata")
    p_probe.add_argument("path")

    p_validate = sub.add_parser("validate", help="Validate a ShotPlan JSON")
    p_validate.add_argument("path")

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


if __name__ == "__main__":
    main()
