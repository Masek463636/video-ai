"""Execute the fifth director contract locally; never invoke an AI provider."""
from pathlib import Path
import hashlib
import json
import math
import re
import time

from .fifth_catalog import fingerprint
from .fifth_plan import validate_plan, validate_transcript
from .fifth_session import atomic_json
from .story_media import download_complete, IMAGE_EXTS, VIDEO_EXTS
from .story_render import _display_aspect, _input, write_story_captions
from .renderer import _run, _safe_probe_duration, _assert_duration, _filter_path

VERSION = 'fifth-render-v1'


def _emoji_for(beat, catalog):
    """Only an exact catalog tag match, with genuine transparency, may overlay."""
    if not beat.get('emoji') or beat['source'] == 'meme':
        return None
    from PIL import Image
    normalize = lambda s: ' '.join(re.findall(r'\w+', s.casefold()))
    query = normalize(beat['emoji']['query'])
    for entry in catalog.get('assets', []):
        desc = entry.get('description', {})
        if desc.get('kind') != 'emoji' or not desc.get('readable'):
            continue
        if query not in [normalize(t) for t in desc.get('tags', [])]:
            continue
        path = Path(entry['path'])
        if not path.is_file() or fingerprint(path) != entry['id']:
            continue
        try:
            with Image.open(path) as image:
                lo, hi = image.convert('RGBA').getchannel('A').getextrema()
                if lo == 255 or hi == 0:
                    continue
        except (OSError, ValueError):
            continue
        return dict(path=str(path.resolve()), kind='video' if path.suffix.lower() in VIDEO_EXTS else 'image',
                    source_start=0, fingerprint=entry['id'])
    return None


def materialize(plan, selection, transcript, audio, cache, *, catalog=None, fps=30):
    validate_transcript(transcript)
    original_plan = plan
    plan = validate_plan(plan, transcript)
    if original_plan.get("schema") == "fifth-director-v1":
        for old, new in zip(original_plan["beats"], plan["beats"]):
            if old.get("caption") != new["caption"] or abs(old.get("start", -1)-new["start"]) > .001 or abs(old.get("end", -1)-new["end"]) > .001:
                raise ValueError("Используй исходную расшифровку этого плана")
    duration = _safe_probe_duration(audio)
    if not math.isfinite(duration) or duration <= 0 or transcript.duration > duration + .1:
        raise ValueError('Озвучка не соответствует временным границам расшифровки')
    if selection.get('schema') != 'fifth-selection-v1':
        raise ValueError('Нужен selection.json пятого стиля')
    missing = [b['id'] for b in plan['beats'] if b['id'] not in selection.get('selections', {})]
    if missing or selection.get('unresolved'):
        raise ValueError('Сначала подбери кадры для всех сцен: ' + ', '.join(missing or selection['unresolved']))
    boundaries = [math.floor(b['start']*fps+.5) for b in plan['beats']] + [math.ceil(duration*fps)]
    if boundaries[0] != 0 or any(b <= a for a,b in zip(boundaries,boundaries[1:])):
        raise ValueError('Каждая сцена должна занимать хотя бы один кадр')
    beats, warnings = [], []
    for i, beat in enumerate(plan['beats']):
        chosen = selection['selections'][beat['id']]['selected']
        if chosen.get('source') != beat['source'] or beat['source'] not in ('stock','meme'):
            raise ValueError('Источник не соответствует плану: ' + beat['id'])
        path = Path(chosen['path'])
        if chosen.get('full_url') and beat['source'] == 'stock':
            key = hashlib.sha256(chosen['full_url'].encode()).hexdigest()
            path = Path(cache)/'originals'/(key+'.mp4')
            print(f"[fifth] full source: {beat['id']}", flush=True)
            download_complete(chosen['full_url'], path)
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTS | VIDEO_EXTS:
            raise ValueError('Не найден материал: ' + str(path))
        source_start = float(chosen.get('start', 0))
        if not math.isfinite(source_start) or source_start < 0:
            raise ValueError('Некорректное начало фрагмента')
        kind = 'video' if path.suffix.lower() in VIDEO_EXTS else 'image'
        length = (boundaries[i+1]-boundaries[i])/fps
        if kind == 'video':
            available = _safe_probe_duration(path)
            if available <= 0 or source_start >= available:
                raise ValueError('Невозможно прочитать выбранный фрагмент: ' + beat['id'])
            if beat['source'] == 'stock' and available-source_start < length-.05:
                raise ValueError('Стоковый фрагмент короче сцены: ' + beat['id'])
        asset = dict(path=str(path.resolve()),kind=kind,source_start=source_start,fingerprint=fingerprint(path))
        emoji = _emoji_for(beat, catalog or {})
        if beat.get('emoji') and emoji is None:
            warnings.append(dict(beat=beat['id'],reason='No exact verified transparent emoji; omitted'))
        row = dict(beat,start=boundaries[i]/fps,end=boundaries[i+1]/fps,
                   start_word=beat['first_word'],end_word=beat['last_word']+1,
                   kind='meme' if beat['source']=='meme' else 'footage',assets=[asset],emoji_asset=emoji)
        beats.append(row)
    return dict(schema=VERSION,audio=str(Path(audio).resolve()),duration=duration,beats=beats,warnings=warnings)


def render_shot(beat, output, *, width, height, fps):
    asset = beat['assets'][0]
    count = round((beat['end']-beat['start'])*fps)
    ratio = _display_aspect(asset)
    # Portrait fills the screen. Wide footage remains large in the upper 60%.
    # A meme owns the entire shot; fit it whole over a blur of itself.
    meme = beat['source'] == 'meme'
    panel_h = height if meme or ratio < 1 else round(height*.60/2)*2
    mode = 'decrease' if meme else 'increase'
    fit = f'scale={width}:{panel_h}:force_original_aspect_ratio={mode},setsar=1'
    if not meme:
        fit += f',crop={width}:{panel_h}'
    filters = [f'[0:v]setpts=PTS-STARTPTS,fps={fps},scale=trunc(ih*{ratio:.9f}/2)*2:ih,setsar=1,split=2[bg][fg]',
               f'[bg]scale={width}:{height}:force_original_aspect_ratio=increase,crop={width}:{height},gblur=sigma=22,eq=brightness=-0.15[back]',
               f'[fg]{fit}[front]',
               f'[back][front]overlay=(W-w)/2:({panel_h}-h)/2:shortest=1[layout]']
    current = 'layout'
    if beat['camera'] == 'gentle_push' and not meme:
        # Ease in/out from 1.00 to 1.045, evaluated once per output frame.
        phase = f'min(on/{max(1,count-1)},1)'
        zoom = f'1+0.045*({phase})*({phase})*(3-2*({phase}))'
        filters.append(f"[{current}]zoompan=z='{zoom}':x='iw/2-iw/zoom/2':y='ih/2-ih/zoom/2':d=1:s={width}x{height}:fps={fps}[pushed]")
        current = 'pushed'
    cmd = ['ffmpeg','-y','-v','error','-filter_complex_threads','1'] + _input(asset,count/fps,fps)
    emoji = beat.get('emoji_asset')
    if emoji:
        cmd += _input(emoji,count/fps,fps)
        size = max(2,round(width*.20/2)*2)
        start = max(.18,beat['emoji']['start']-beat['start'])
        end = min(count/fps-.12,start+1.1)
        if end-start >= .35:
            filters.append(f'[1:v]setpts=PTS-STARTPTS,scale={size}:{size}:force_original_aspect_ratio=decrease,format=rgba[emoji]')
            filters.append(f"[{current}][emoji]overlay=(W-w)/2:H*0.68-h/2:enable='between(t,{start:.4f},{end:.4f})':eof_action=repeat[reacted]")
            current = 'reacted'
    filters.append(f'[{current}]trim=end_frame={count},setpts=PTS-STARTPTS,setsar=1,format=yuv420p[v]')
    cmd += ['-filter_complex',';'.join(filters),'-map','[v]','-an','-frames:v',str(count),'-c:v','libx264',
            '-preset','veryfast','-crf','20','-r',str(fps),'-colorspace','bt709','-color_primaries','bt709',
            '-color_trc','bt709',str(output)]
    _run(cmd)
    _assert_duration(output,count/fps,label='fifth shot')


def sound_bed(beats, duration, output):
    # Cue vocabulary is explicit; no automatic sound at every cut.
    cues = [(b['start'], b['sound']) for b in beats if b['sound'] != 'none']
    cmd = ['ffmpeg','-y','-v','error','-f','lavfi','-i','anullsrc=r=48000:cl=stereo']
    filters=[]; labels=['[0:a]']; last=-10
    for start, sound in cues:
        if start-last < .8: continue
        last=start; index=len(labels)
        if sound == 'whoosh':
            source='anoisesrc=color=pink:duration=0.18:sample_rate=48000';effect='highpass=f=650,lowpass=f=4200,volume=0.07'
        else:
            source=f'sine=frequency={180 if sound=="soft_hit" else 1100}:duration=0.12:sample_rate=48000';effect='volume=0.12'
        cmd += ['-f','lavfi','-i',source]
        filters.append(f'[{index}:a]{effect},afade=t=in:d=0.01,afade=t=out:st=0.03:d=0.09,adelay={round(start*1000)}:all=1[s{index}]')
        labels.append(f'[s{index}]')
    filters.append(''.join(labels)+f'amix=inputs={len(labels)}:duration=first:normalize=0,atrim=duration={duration}[a]')
    _run(cmd+['-filter_complex',';'.join(filters),'-map','[a]','-t',str(duration),'-c:a','pcm_s16le',str(output)])


def render_fifth(plan, selection, transcript, audio, output, work, *, cache, catalog=None,
                 width=1080, height=1920, fps=30, effects=True):
    if width < 2 or height < 2 or width%2 or height%2 or fps < 1:
        raise ValueError('Invalid render dimensions or frame rate')
    started=time.monotonic();work=Path(work).resolve();output=Path(output).resolve()
    work.mkdir(parents=True,exist_ok=True);output.parent.mkdir(parents=True,exist_ok=True)
    timeline=materialize(plan,selection,transcript,audio,cache,catalog=catalog,fps=fps)
    inputs={Path(audio).resolve()}
    inputs.update(Path(a['path']).resolve() for b in timeline['beats'] for a in b['assets'] + ([b['emoji_asset']] if b['emoji_asset'] else []))
    if output in inputs or output.suffix.lower() != '.mp4':
        raise ValueError('Выбери отдельный выходной файл MP4')
    atomic_json(work/'timeline.json',timeline)
    clips=[];reused=0
    for index,beat in enumerate(timeline['beats']):
        identity=dict(version=VERSION,beat=beat,width=width,height=height,fps=fps)
        key=hashlib.sha256(json.dumps(identity,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        clip=work/'clips'/(key+'.mp4');clip.parent.mkdir(exist_ok=True)
        valid=False
        if clip.exists():
            try:
                _assert_duration(clip,beat['end']-beat['start'],label='cached fifth shot');valid=True
            except Exception: pass
        print(f'[fifth-render] scene {index+1}/{len(timeline["beats"])}'+(' cached' if valid else ''),flush=True)
        if valid: reused+=1
        else:
            pending=clip.with_suffix('.pending.mp4')
            render_shot(beat,pending,width=width,height=height,fps=fps);pending.replace(clip)
        clips.append(clip)
    listing=work/'concat.txt'
    listing.write_text(''.join("file '"+str(p).replace("'","'\\''")+"'\n" for p in clips),encoding='utf-8')
    base=work/'base.mp4'
    _run(['ffmpeg','-y','-v','error','-f','concat','-safe','0','-i',str(listing),'-c','copy',str(base)])
    captions=work/'captions.ass'
    beats=[dict(b,_layout={'caption_y':round(height*.84)}) for b in timeline['beats']]
    write_story_captions(transcript,beats,captions,width=width,height=height)
    cmd=['ffmpeg','-y','-v','error','-reinit_filter','0','-i',str(base),'-i',str(audio)]
    if effects and any(b['sound']!='none' for b in beats):
        bed=work/'sfx.wav';sound_bed(beats,timeline['duration'],bed)
        cmd+=['-i',str(bed),'-filter_complex','[1:a][2:a]amix=inputs=2:duration=first:normalize=0,alimiter=limit=0.95:level=0[a]', '-map','0:v','-map','[a]']
    else: cmd+=['-map','0:v','-map','1:a']
    pending=output.with_name(output.stem+'.pending.mp4')
    cmd+=['-vf',f"ass='{_filter_path(captions)}'",'-c:v','libx264','-preset','medium','-crf','20',
          '-pix_fmt','yuv420p','-c:a','aac','-b:a','192k','-movflags','+faststart','-t',str(timeline['duration']),str(pending)]
    _run(cmd);_assert_duration(pending,timeline['duration'],label='fifth final');pending.replace(output)
    report=dict(ok=True,output=str(output),scenes=len(beats),reused_scenes=reused,gemini_calls=0,
                elapsed_seconds=round(time.monotonic()-started,2),warnings=timeline['warnings'])
    atomic_json(work/'render-report.json',report)
    return report
