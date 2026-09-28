"""Fourth-style primary-track meme cuts; emoji is the only overlay kind."""
from pathlib import Path
import json


def select_meme_shots(effects, duration, fps):
    from .dynamic_reactions import place_reactions
    shots = []
    # Reuse the existing asset verification contract and duplicate/overlap guard.
    for item in place_reactions([e for e in effects if isinstance(e, dict) and e.get("reaction_kind") == "meme"], duration):
        if item['reaction_kind'] != 'meme':
            continue
        start = max(0, round(item['start'] * fps))
        end = min(round(duration * fps), round(item['end'] * fps))
        if end - start < max(1, round(.65 * fps)):
            continue
        shots.append(dict(item, start=start / fps, end=end / fps,
                          start_frame=start, end_frame=end, role='primary_shot'))
    return shots


def emoji_overlays(effects, duration, shots, spans):
    from .dynamic_reactions import place_reactions
    from .viral_fx import pace_viral_accents
    # Never stack an emoji over a meme, or let it bridge the meme's cut.
    candidates = [e for e in effects if isinstance(e, dict) and e.get('reaction_kind') == 'emoji']
    candidates = pace_viral_accents(candidates, spans)
    result = []
    for e in place_reactions(candidates, duration):
        if any(e['start'] < shot['end'] + .15 and e['end'] > shot['start'] - .1 for shot in shots):
            continue
        result.append(dict(e, layout_box=[.34, .16, .32, .24]))
    return result


def compose_meme_shots(base, effects, plan, root, duration, *, crf=20):
    """Concatenate actual independent clips; no underlying stock during a meme.

    All segments share frame boundaries. Audio and captions are attached once
    by render_plan afterwards, so cutting the picture cannot move speech.
    """
    from .renderer import _run
    shots = select_meme_shots(effects, duration, plan.fps)
    root = Path(root)
    (root / 'meme-shots.json').write_text(json.dumps({'shots':shots}, ensure_ascii=False, indent=2), encoding='utf-8')
    if not shots:
        return Path(base), []
    folder = root / 'montage'
    folder.mkdir(exist_ok=True)
    fps, w, h = plan.fps, plan.width, plan.height
    total = round(duration * fps)
    segments, cursor = [], 0
    for shot in shots:
        if cursor < shot['start_frame']:
            segments.append((cursor, shot['start_frame'], None))
        segments.append((shot['start_frame'], shot['end_frame'], shot))
        cursor = shot['end_frame']
    if cursor < total:
        segments.append((cursor, total, None))
    clips = []
    for index, (start, end, shot) in enumerate(segments):
        clip = folder / f'{index:03d}.mp4'
        count = end - start
        cmd = ['ffmpeg', '-y', '-v', 'error']
        if shot is None:
            cmd += ['-ss', f'{start / fps:.9f}', '-i', str(base)]
            vf = f'fps={fps},setsar=1,setpts=PTS-STARTPTS'
        else:
            asset = Path(shot['asset'])
            if asset.suffix.lower() in {'.png', '.jpg', '.jpeg', '.webp', '.bmp'}:
                cmd += ['-loop', '1', '-framerate', str(fps)]
            else:
                cmd += ['-stream_loop', '-1']
            cmd += ['-i', str(asset)]
            # Static fill crop: this is a whole shot, no zoom/pop or stock behind it.
            vf = (f'scale={w}:{h}:force_original_aspect_ratio=increase,'
                  f'crop={w}:{h},setsar=1,fps={fps},setpts=PTS-STARTPTS')
        cmd += ['-vf', vf, '-frames:v', str(count), '-an', '-c:v', 'libx264',
                '-preset', 'veryfast', '-crf', str(crf), '-pix_fmt', 'yuv420p', str(clip)]
        _run(cmd)
        clips.append(clip)
    listing = folder / 'concat.txt'
    listing.write_text('\n'.join("file '" + str(p.resolve()).replace("'", "'\\''") + "'" for p in clips), encoding='utf-8')
    output = root / 'montage.mp4'
    _run(['ffmpeg', '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-i', str(listing),
          '-an', '-c:v', 'copy', str(output)])
    print(f'[viral] standalone meme shots={len(shots)}; no meme overlays', flush=True)
    return output, shots
