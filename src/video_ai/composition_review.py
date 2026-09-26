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
        try:
            data = self.client._generate_json(parts,temperature=0.01)
            if isinstance(data,list) and len(data)==1:
                data=data[0]
            if not isinstance(data,dict):
                raise ValueError('expected object')
            return data
        except Exception:
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
                    'Return JSON {"usable":true/false,"reason":"specific visible evidence"}.')
            times=[duration*f for f in (.08,.5,.90)]
            data=self.ask([{'text':prompt}]+frames(clip,times,self.root,f'scene{index}'))
            row['reason']=str(data.get('reason',''))[:500]
            if data.get('usable') is True:
                row['status']='accepted'; self.save(); return
            if data.get('usable') is not False:
                raise ValueError('missing usable verdict')
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
            parts=[{'text':prompt+' These are alternative crops/windows from the SAME source. Choose only a usable improvement. Return JSON {"choice":integer or null,"usable":true/false,"reason":"evidence"}.'}]
            for number,(_,out) in enumerate(options):
                parts.append({'text':f'CHOICE {number}'})
                parts.extend(frames(out,times,self.root,f'alt{index}_{number}'))
            selected=self.ask(parts)
            choice=selected.get('choice')
            row['repair_reason']=str(selected.get('reason',''))[:500]
            if selected.get('usable') is True and type(choice) is int and 0<=choice<len(options):
                proposal,_=options[choice]
                _render_scene(proposal,duration,plan,clip,crf=20,editing_polish=editing_polish,reference_framing=reference_framing)
                row.update(status='repaired',source_start=proposal.source_start,focus_x=proposal.focus_x)
            print(f'[composition] scene {index+1}: {row["status"]}',flush=True)
        except Exception as exc:
            row.update(status='review_failed',error=type(exc).__name__)
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
                prompt=(f'Check an OPTIONAL insert against narration: {caption}. Requested insert: {item.get("query","")}. '
                    'First three images are FINAL CROPPED background frames throughout the insert; last images are the insert. '
                    'Approve only if the insert clearly illustrates the narrated object or a fitting reaction. '
                    'Unrelated people at a table do not illustrate a phone or messaging app. Reject uncertain matches. '
                    'Find bounding boxes covering ALL important sharp-foreground faces, objects and actions across ALL three frames. '
                    'Ignore decorative blurred background. Coordinates normalized to the entire background frame. '
                    'Return JSON {"relevant":true/false,"reason":"visible evidence",'
                    '"protected_boxes":[[x,y,width,height],...]}.')
                parts=[{'text':prompt}]+frames(base,[start+(end-start)*f for f in (.08,.5,.92)],self.root,f'overlaybg{index}')
                parts.append({'text':'INSERT'})
                parts.extend(frames(Path(item['asset']),[0],self.root,f'insert{index}'))
                data=self.ask(parts)
                row['reason']=str(data.get('reason',''))[:500]
                if data.get('relevant') is not True:
                    row['status']='irrelevant'; continue
                protected=boxes(data.get('protected_boxes'))
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
                row.update(status='review_failed',error=type(exc).__name__)
        self.save()
        (self.root/'overlays.reviewed.json').write_text(json.dumps({'overlays':result},ensure_ascii=False,indent=2),encoding='utf-8')
        return result
