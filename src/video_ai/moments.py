"""Bounded optional selection of an action window in downloaded footage."""
from __future__ import annotations

import base64
import json
import math
import subprocess
from pathlib import Path

from .models import ShotPlan
from .renderer import _display_ranges, _safe_probe_duration


def window_starts(source_duration: float, duration: float) -> list[float]:
    if not math.isfinite(source_duration) or source_duration <= duration:
        return [0.0]
    last = source_duration - duration
    count = min(8, max(2, math.ceil(last / max(duration, 1.0)) + 1))
    return [round(last * i / (count - 1), 3) for i in range(count)]


def select_moments(plan: ShotPlan, work_dir: str | Path, *, client=None) -> list[dict]:
    """Inspect three chronological frames per window; retain timing on failure.

    At most one model call per eligible scene. Sampled frames are not proof of
    every action between samples. Opt-in because this spends additional quota.
    """
    if client is None:
        from .gemini_ai import get_gemini_client
        client = get_gemini_client()
    root = Path(work_dir)
    root.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []
    duration = _safe_probe_duration(plan.audio) or max(s.end for s in plan.scenes)
    for index, (scene, start, end) in enumerate(_display_ranges(plan, duration)):
        if scene.asset_kind != 'video' or not scene.asset or scene.source_mode == 'meme_library':
            continue
        row = {'scene': index, 'source_start': scene.source_start, 'status': 'unchanged'}
        report.append(row)
        if client is None:
            row['status'] = 'no_gemini'
            continue
        try:
            length = end - start
            source_duration = _safe_probe_duration(Path(scene.asset))
            starts = window_starts(source_duration, length)
            if len(starts) == 1:
                row['status'] = 'short_source'
                continue
            parts = [{'text': (
                'Select the source window that best SHOWS the narrated event. '
                'Each window has three chronological frames: beginning, middle, end. '
                'Prefer visible action/comparison, not merely the same noun. '
                'Do not claim unseen motion. Reject all if none demonstrates the intent. '
                f'Narration: {scene.caption}. Intent: {scene.visual_description or scene.query}. '
                'Return JSON {"window": integer or null, "fit": 0-100, "reason": "visible evidence"}.'
            )}]
            valid = set()
            for window, offset in enumerate(starts):
                frames = []
                for sample, fraction in enumerate((0.05, 0.5, 0.95)):
                    frame = root / f'scene_{index}_window_{window}_{sample}.jpg'
                    frame.unlink(missing_ok=True)
                    completed = subprocess.run([
                        'ffmpeg', '-y', '-v', 'error', '-ss', str(min(offset + length * fraction, max(0.0, source_duration - 0.2))),
                        '-i', scene.asset, '-frames:v', '1', '-vf', 'scale=384:-2', str(frame),
                    ], capture_output=True, timeout=20, check=False)
                    if completed.returncode == 0 and frame.exists():
                        frames.append({'inline_data': {'mime_type': 'image/jpeg',
                            'data': base64.b64encode(frame.read_bytes()).decode('ascii')}})
                if len(frames) == 3:
                    valid.add(window)
                    parts.append({'text': f'WINDOW {window}: {offset:.3f} to {offset + length:.3f} seconds'})
                    parts.extend(frames)
            if not valid:
                row['status'] = 'no_frames'
                continue
            data = client._generate_json(parts, temperature=0.01)
            chosen = data.get('window')
            fit = float(data.get('fit', 0))
            if type(chosen) is not int or chosen not in valid or not math.isfinite(fit) or not 70 <= fit <= 100:
                row['status'] = 'no_confident_match'
                continue
            scene.source_start = starts[chosen]
            row.update(status='selected', source_start=scene.source_start, fit=fit,
                       reason=str(data.get('reason', ''))[:300])
            print(f'[moments] scene {index + 1}: source {scene.source_start:.2f}s, fit={fit:.0f}', flush=True)
        except Exception as exc:
            row.update(status='failed', error=type(exc).__name__)
    (root / 'moments.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if client is None:
        print('[moments] Gemini unavailable; source timing unchanged', flush=True)
    return report
