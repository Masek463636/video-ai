"""Bounded review of cropped footage and optional foreground inserts.

Uses actual renderer previews, not provider thumbnails. No credentials in reports.
A failed provider disables further review requests for this render.
"""
from __future__ import annotations
import base64
import json
import math
import subprocess
from dataclasses import replace
from pathlib import Path


def boxes(value):
    if not isinstance(value, list):
        raise ValueError('protected_boxes must be a list')
    result = []
    for box in value:
        if not isinstance(box, list) or len(box) != 4:
            raise ValueError('invalid box')
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in box):
            raise ValueError('invalid coordinate')
        x, y, w, h = box
        if not (0 <= x < 1 and 0 <= y < 1 and w > 0 and h > 0 and x+w <= 1.001 and y+h <= 1.001):
            raise ValueError('box outside frame')
        result.append(box)
    return result


def requirements(value):
    if not isinstance(value, list) or not value or len(value) > 12:
        raise ValueError('missing fixed visual requirements')
    if any(not isinstance(v, str) or not v.strip() or len(v) > 400 for v in value):
        raise ValueError('invalid visual requirement')
    return value


def repair_approved(data, fixed):
    checks = data.get('checks')
    return (data.get('usable') is True and data.get('preserves_visible_subjects') is True
            and isinstance(checks, list) and len(checks) == len(fixed)
            and all(isinstance(c, dict) and c.get('requirement') == r
                    and c.get('visible') is True and isinstance(c.get('evidence'), str)
                    and c['evidence'].strip() for c, r in zip(checks, fixed)))


def failure(row, exc, disabled):
    row.update(status='review_failed', error=type(exc).__name__,
               error_kind='invalid_review_response' if isinstance(exc, ValueError) else 'processing_or_provider_failure',
               remaining_review_disabled=disabled)
    # Do not persist provider exceptions: they may contain credential-bearing URLs.
    if isinstance(exc, subprocess.CalledProcessError):
        stderr = exc.stderr or b''
        row['ffmpeg_error'] = (stderr.decode('utf-8', errors='replace') if isinstance(stderr, bytes) else str(stderr))[-2000:]
    elif isinstance(exc, ValueError):
        row['validation_error'] = str(exc)[:300]


def choose_slot(protected):
    # Reserve captions and outer UI margins. A single fixed box bounds even
    # unusually tall/wide stickers. No fly-in path through protected content.
    blocked = protected + [[0, .50, 1, .14]]
    for size in (.28, .22):
        for y in (.68, .12, .32):
            for x in (.06, .94-size):
                rect = [x, y, size, size*.65]
                if all(not (x < b[0]+b[2]+.02 and x+rect[2] > b[0]-.02 and y < b[1]+b[3]+.02 and y+rect[3] > b[1]-.02) for b in blocked):
                    return rect
    return None


def frames(path, times, root, prefix):
    parts = []
    for i, time in enumerate(times):
        target = root / f'{prefix}_{i}.jpg'
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(max(0,time)),'-i',str(path),
                        '-frames:v','1','-vf','scale=270:-2',str(target)],check=True,capture_output=True,timeout=25)
        parts.append({'inline_data':{'mime_type':'image/jpeg','data':base64.b64encode(target.read_bytes()).decode('ascii')}})
    return parts


class CompositionReviewer:
    def __init__(self, root, client=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if client is None:
            from .gemini_ai import get_gemini_client
            client = get_gemini_client()
        self.client = client
        self.disabled = client is None
        self.report = {'scenes':[], 'overlays':[]}

    def save(self):
        (self.root/'review.json').write_text(json.dumps(self.report,ensure_ascii=False,indent=2),encoding='utf-8')

    def ask(self, parts):
        if self.disabled:
            raise RuntimeError('review unavailable')
        # A malformed model response is local to this item, not an outage.
        # Retry its format once; never pick an arbitrary element from a list.
        for attempt in range(2):
            try:
                request = parts if attempt == 0 else parts + [{'text':
                    'FORMAT CORRECTION: return exactly ONE JSON object matching the requested schema. '
                    'Do not return an array of separate frame or candidate judgements.'}]
                data = self.client._generate_json(request,temperature=0.01)
                if isinstance(data,list) and len(data)==1:
                    data=data[0]
                if not isinstance(data,dict):
                    raise ValueError('expected one review object')
                return data
            except ValueError:
                if attempt == 1:
                    raise ValueError('review response format invalid after one correction') from None
            except (RuntimeError, OSError, TimeoutError):
                self.disabled=True
                raise

    def scene(self, scene, plan, duration, clip, index, *, reference_framing, editing_polish):
        from .renderer import _render_scene, _safe_probe_duration
        row={'span':index,'status':'unchecked','asset':scene.asset,'source_start':scene.source_start}
        self.report['scenes'].append(row)
        if self.disabled:
            row['status']='review_unavailable'; self.save(); return
        try:
            prompt=(f'Review this FINAL CROPPED shot, three chronological frames. Narration: {scene.caption}. '
                    f'Required visible action: {scene.visual_description or scene.query}. '
                    'Blurred background is decoration. Judge the sharp foreground only. '
                    'Reject if the required subject/action is obscured, outside crop, or only the noun matches. '
                    'Do not require a face for hand/object closeups. Do not infer unseen actions. '
                    'Extract a fixed checklist from the required action and narration, including required objects and interactions. Do not invent extra requirements. Return JSON {"usable":true/false,"reason":"specific visible evidence","requirements":["one concrete visible requirement",...]}.')
            times=[duration*f for f in (.08,.5,.90)]
            data=self.ask([{'text':prompt}]+frames(clip,times,self.root,f'scene{index}'))
            row['reason']=str(data.get('reason',''))[:500]
            if data.get('usable') is True:
                row['status']='accepted'; self.save(); return
            if data.get('usable') is not False:
                raise ValueError('missing usable verdict')
            fixed=requirements(data.get('requirements'))
            row['requirements']=fixed
            row['status']='unresolved'
            if scene.asset_kind!='video':
                self.save(); return
            last=max(0,_safe_probe_duration(Path(scene.asset))-duration)
            options=[]
            for offset,focus in [(scene.source_start,.25),(scene.source_start,.75),(last*.5,.5),(last,.5)]:
                proposal=replace(scene,source_start=min(last,offset),focus_x=focus)
                if any(abs(p.source_start-proposal.source_start)<.05 and p.focus_x==focus for p,_ in options):
                    continue
                number=len(options)
                out=self.root/f'alternative_{index}_{number}.mp4'
                preview_plan=replace(plan,width=270,height=480,fps=12)
                _render_scene(proposal,duration,preview_plan,out,crf=26,editing_polish=editing_polish,reference_framing=reference_framing)
                options.append((proposal,out))
            contract = ('FIXED requirements, do not relax or reinterpret: '+json.dumps(fixed,ensure_ascii=False)+
                '. A facial expression cannot replace typing, holding or a visible phone. '
                'Keep narration-relevant objects AND people already visible in the original sharp foreground. '
                'Compare ORIGINAL with candidate. Every requirement must be visibly satisfied. '
                'Return JSON {"choice":integer or null,"usable":true/false,"preserves_visible_subjects":true/false,'
                '"checks":[{"requirement":"exact requirement text in original order","visible":true/false,"evidence":"visible evidence"}],"reason":"evidence"}.')
            original=frames(clip,times,self.root,f'original{index}')
            parts=[{'text':contract},{'text':'ORIGINAL'}]+original
            for number,(_,out) in enumerate(options):
                parts.append({'text':f'CHOICE {number}'})
                parts.extend(frames(out,times,self.root,f'alt{index}_{number}'))
            selected=self.ask(parts)
            row['selection']=selected
            choice=selected.get('choice')
            if repair_approved(selected,fixed) and type(choice) is int and 0<=choice<len(options):
                proposal,_=options[choice]
                # Verify the actual full-resolution replacement before touching the original.
                pending=Path(clip).with_name(Path(clip).stem+'.review-pending.mp4')
                try:
                    _render_scene(proposal,duration,plan,pending,crf=20,editing_polish=editing_polish,reference_framing=reference_framing)
                    verification=self.ask([{'text':contract},{'text':'ORIGINAL'}]+original+
                        [{'text':'CANDIDATE'}]+frames(pending,times,self.root,f'verified{index}'))
                    row['verification']=verification
                    if repair_approved(verification,fixed):
                        pending.replace(clip)
                        row.update(status='repaired',source_start=proposal.source_start,focus_x=proposal.focus_x)
                finally:
                    pending.unlink(missing_ok=True)
            print(f'[composition] scene {index+1}: {row["status"]}',flush=True)
        except Exception as exc:
            failure(row,exc,self.disabled)
        self.save()

    def overlays(self, base, overlays, plan):
        result=[]
        for index,item in enumerate(overlays):
            row={'overlay':index,'asset':item.get('asset'),'query':item.get('query'),'status':'skipped'}
            self.report['overlays'].append(row)
            if item.get('type')=='text':
                result.append(item); row['status']='text_preserved'; continue
            if self.disabled:
                row['reason']='Review unavailable; optional insert omitted'; continue
            try:
                start,end=float(item['start']),float(item['end'])
                caption=' '.join(s.caption or '' for s in plan.scenes if s.start<end and s.end>start)
                prompt=(f'Judge ONLY the attached OPTIONAL INSERT, without any background footage. Narration: {caption}. '
                    f'Requested insert: {item.get("query", "")}. '
                    'Describe what is actually visible in this insert. Approve only a clear illustration of the narrated '
                    'object or a fitting reaction. Do not imagine a phone merely because narration mentions one. '
                    'Unrelated restaurant people are not a smartphone. Reject uncertain matches. '
                    'Return JSON {"relevant":true/false,"reason":"visible evidence"}.')
                data=self.ask([{'text':prompt}]+frames(Path(item['asset']),[0],self.root,f'insert{index}'))
                row['insert_review']=data
                row['reason']=str(data.get('reason',''))[:500]
                if data.get('relevant') is not True:
                    row['status']='irrelevant'; continue
                placement_parts=[{'text':
                    'These are ONLY background frames, NOT the insert. Find boxes protecting all important sharp '
                    'foreground faces, objects and actions across all three frames. Ignore decorative blur. '
                    'Return one JSON object {"protected_boxes":[[x,y,width,height],...]}. '
                    'Use normalized numbers 0..1 relative to the ENTIRE image; x+width and y+height <=1. '
                    'Never use pixel coordinates, corner coordinates or named objects.'}]+frames(
                        base,[start+(end-start)*f for f in (.08,.5,.92)],self.root,f'overlaybg{index}')
                for attempt in range(2):
                    placement=self.ask(placement_parts)
                    row.setdefault('placement_reviews',[]).append(placement)
                    try:
                        protected=boxes(placement.get('protected_boxes'))
                        break
                    except ValueError:
                        if attempt: raise
                        placement_parts.append({'text':'FORMAT CORRECTION: protected_boxes must be arrays of '
                            'four normalized numbers [x,y,width,height], not pixels or corners. Return the complete object again.'})
                # Existing quantity callouts occupy the upper center.
                if any(other.get('type')=='text' and float(other['start'])<end and float(other['end'])>start for other in overlays):
                    protected.append([.1,.08,.8,.20])
                slot=choose_slot(protected)
                row['protected_boxes']=protected
                if slot is None:
                    row['status']='no_free_space'; continue
                placed=dict(item,layout_box=slot,animation='pop',label='')
                result.append(placed)
                row.update(status='placed',layout_box=slot)
            except Exception as exc:
                failure(row,exc,self.disabled)
        self.save()
        (self.root/'overlays.reviewed.json').write_text(json.dumps({'overlays':result},ensure_ascii=False,indent=2),encoding='utf-8')
        return result
