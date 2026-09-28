"""Bounded candidate preparation and contextual multi-scene selection."""
from pathlib import Path
import base64
import hashlib
import json
import re
from .fifth_catalog import fingerprint
from .fifth_session import atomic_json
from .story_media import download_complete, IMAGE_EXTS
from .stock_video import search_stock_videos


def window_preview(path, start, duration, cache):
    from .composition_review import frames
    key=hashlib.sha256(f'{fingerprint(path)}:{start:.3f}:{duration:.3f}'.encode()).hexdigest()
    root=Path(cache)/'windows'/key
    meta=root/'preview.json'
    try:
        data=json.loads(meta.read_text(encoding='utf-8'))
        if data['parts'] and Path(data['image']).is_file():return key,data
    except (OSError,ValueError,KeyError,TypeError):pass
    root.mkdir(parents=True,exist_ok=True)
    times=[0] if Path(path).suffix.lower() in IMAGE_EXTS else [start+duration*f for f in (.08,.5,.9)]
    parts=frames(path,times,root,'window')
    image=root/'rank.jpg'
    image.write_bytes(base64.b64decode(parts[len(parts)//2]['inline_data']['data']))
    data=dict(parts=parts,image=str(image.resolve()))
    atomic_json(meta,data)
    return key,data


def prepare_candidates(plan,catalog,cache,*,ranker=None):
    """No Gemini calls here. Provider search and CLIP are reused from old styles."""
    cache=Path(cache); result={}; search_cache={}; errors=[]
    for beat in plan['beats']:
        print(f"[fifth] preparing {beat['id']}: {beat['source']}",flush=True)
        duration=beat['end']-beat['start']; choices=[]; sources=[]
        if beat['source']=='meme':
            tokens=set(re.findall(r'\w+',' '.join(beat['queries']).lower()))
            entries=[e for e in catalog.get('assets',[]) if e.get('description',{}).get('kind')=='meme' and e['description'].get('readable')]
            def score(e):
                text=json.dumps(e['description'],ensure_ascii=False).lower()
                return sum(t in text for t in tokens)
            for e in sorted(entries,key=score,reverse=True)[:6]:
                path=Path(e['path'])
                if path.is_file() and fingerprint(path)==e['id']:
                    sources.append(dict(path=path,source='meme',duration=e['duration'],url='',full_url=''))
        elif beat['source']=='stock':
            seen=set()
            for query in beat['queries'][:2]:
                if query not in search_cache:search_cache[query]=search_stock_videos(query,limit=4)
                for item in search_cache[query]:
                    if item.page_url in seen or item.duration<duration or min(item.width,item.height)<320:continue
                    seen.add(item.page_url)
                    url=item.preview_url or item.download_url
                    key=hashlib.sha256(url.encode()).hexdigest()
                    path=cache/'downloads'/(key+'.mp4')
                    try:
                        download_complete(url,path)
                        sources.append(dict(path=path,source='stock',duration=item.duration,url=item.page_url,full_url=item.download_url))
                    except Exception as exc:errors.append(dict(beat=beat['id'],error=type(exc).__name__))
                    if len(sources)>=6:break
                if len(sources)>=6:break
        # Photo/graphic are intentionally unresolved until their dedicated retrievers are integrated.
        for source in sources:
            path=source['path']; available=source['duration']
            starts=[0.] if source['source']=='meme' else sorted({0.,round(max(0.,available-duration)/2,3)})
            for start in starts:
                # Short memes may loop in the renderer; preview only real frames.
                preview_duration=min(duration,available) if available>0 else duration
                try:
                    key,preview=window_preview(path,start,preview_duration,cache)
                    choices.append(dict(id=key,path=str(path.resolve()),source=source['source'],start=start,
                        duration=duration,url=source['url'],full_url=source['full_url'],preview=preview))
                except Exception as exc:errors.append(dict(beat=beat['id'],error=type(exc).__name__))
        choices=list({c['id']:c for c in choices}.values())
        if ranker is not None and choices:
            scores=ranker.score_images(beat['visual_goal'],[c['preview']['image'] for c in choices])
            choices=[c for _,c in sorted(zip(scores,choices),key=lambda x:x[0],reverse=True)]
        result[beat['id']]=choices[:3]
    atomic_json(cache/'last-retrieval.json',dict(errors=errors,counts={k:len(v) for k,v in result.items()},local_ranker=ranker is not None))
    return result


def choose_batches(plan,candidates,session,output,*,batch_size=2):
    """Joint choices are cached with actual preview contents, not just filenames."""
    if batch_size < 1: raise ValueError('batch_size must be positive')
    report=dict(schema='fifth-selection-v1',selections={},unresolved=[],errors=[])
    used=set()
    beats=plan['beats']; prior=[]
    for offset in range(0,len(beats),batch_size):
        group=beats[offset:offset+batch_size]
        print(f'[fifth] selecting beats {offset+1}–{offset+len(group)}/{len(beats)}',flush=True)
        parts=[{'text':'Choose a coherent visual SEQUENCE for these neighboring beats, not isolated matching nouns. '
            'Narrative role, visible action and continuity all matter. Candidate frames show the exact proposed window. '
            'Do not infer unseen actions. Prefer a grounded shot to unrelated grimaces. Memes are full shots. '
            'If no candidate demonstrates the action choose null. Never invent candidate IDs. '
            'Return {"choices":[{"beat":"exact beat ID","candidate":"candidate ID or null",'
            '"alternatives":["other acceptable candidate IDs"],"reason":"visible evidence and continuity"}]}. '
            'Exactly one entry per beat. Do not select the same candidate for two beats. '
            +json.dumps(dict(premise=plan['premise'],payoff=plan['payoff'],prior=prior[-2:],used_candidates=sorted(used),beats=group),ensure_ascii=False)}]
        for beat in group:
            for c in candidates.get(beat['id'],[]):
                parts += [{'text':f"BEAT {beat['id']} CANDIDATE {c['id']} SOURCE {c['source']}"}]+c['preview']['parts']
        if not any(candidates.get(b['id']) for b in group):
            report['unresolved'].extend(b['id'] for b in group)
            atomic_json(output,report);continue
        try:
            choices=session.ask('sequence_selection',parts,lambda raw:validate_choices(raw,group,candidates,used=used),version='fifth-selection-v1')
        except (RuntimeError,ValueError) as exc:
            report['errors'].append(dict(beats=[b['id'] for b in group],error=type(exc).__name__))
            report['unresolved'].extend(b['id'] for b in group)
            atomic_json(output,report);continue
        for choice in choices:
            bid=choice['beat'];lookup={c['id']:c for c in candidates.get(bid,[])}
            if choice['candidate'] is None:
                report['unresolved'].append(bid);continue
            def clean(c):return {k:v for k,v in c.items() if k!='preview'}
            row=dict(selected=clean(lookup[choice['candidate']]),
                     alternatives=[clean(lookup[c]) for c in choice['alternatives']],reason=choice['reason'])
            report['selections'][bid]=row
            used.add(choice['candidate'])
            prior.append(dict(beat=bid,candidate=choice['candidate'],reason=choice['reason']))
        atomic_json(output,report)
    return report


def validate_choices(raw,beats,candidates,*,used=()):
    rows=raw.get('choices') if isinstance(raw,dict) else None
    expected={b['id'] for b in beats}
    if not isinstance(rows,list) or len(rows)!=len(expected):raise ValueError('Incomplete selection batch')
    seen=set();used=set(used);result=[]
    for row in rows:
        if not isinstance(row,dict):raise ValueError('Invalid choice')
        bid=row.get('beat')
        if not isinstance(bid,str) or bid not in expected or bid in seen:raise ValueError('Unknown or duplicate beat')
        allowed={c['id'] for c in candidates.get(bid,[])};selected=row.get('candidate');alts=row.get('alternatives')
        if selected is not None and (not isinstance(selected,str) or selected not in allowed or selected in used):raise ValueError('Invalid selected candidate')
        if not isinstance(alts,list) or len(alts)>2 or any(not isinstance(c,str) or c not in allowed or c==selected for c in alts) or len(set(alts))!=len(alts):raise ValueError('Invalid alternatives')
        if selected is None and alts:raise ValueError('Rejected beat cannot have approved alternatives')
        reason=row.get('reason')
        if not isinstance(reason,str) or not reason.strip() or len(reason)>1200:raise ValueError('Missing selection evidence')
        result.append(dict(beat=bid,candidate=selected,alternatives=alts,reason=reason));seen.add(bid)
        if selected:used.add(selected)
    return sorted(result,key=lambda r:next(i for i,b in enumerate(beats) if b['id']==r['beat']))
