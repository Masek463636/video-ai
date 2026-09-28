"""Resumable fifth-style job and local scene edits for the studio."""
from pathlib import Path
import json
from .fifth_session import FifthSession, atomic_json
from .fifth_catalog import describe_pack, fingerprint
from .fifth_plan import plan_story, validate_plan
from .fifth_materials import prepare_candidates, choose_batches
from .fifth_render import render_fifth
from .transcript import load_transcript, save_transcript, transcribe_local


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def create_fifth(audio, output, work, *, cache, meme_dir='memes', sticker_dir='stickers',
                 effects=True, client=None, semantic=True, max_calls=16):
    work=Path(work);cache=Path(cache);work.mkdir(parents=True,exist_ok=True)
    identity=dict(audio=fingerprint(audio),effects=bool(effects))
    marker=work/'input.json'
    if marker.exists() and read(marker)!=identity:
        raise ValueError('Для другой озвучки или настроек нужна новая рабочая папка')
    atomic_json(marker,identity)
    transcript_file=work/'transcript.json'
    print('[fifth-stage] Расшифровка озвучки',flush=True)
    if not transcript_file.exists():
        save_transcript(transcribe_local(audio,model_size='small',language='ru'),transcript_file)
    transcript=load_transcript(transcript_file)
    if client is None:
        from .gemini_ai import get_gemini_client
        client=get_gemini_client()
    if client is None: raise RuntimeError('Нужен GEMINI_API_KEY')
    session=FifthSession(client,cache/'gemini',work/'pipeline-metrics.json',max_calls=max_calls)
    print('[fifth-stage] Каталог мемов и эмодзи',flush=True)
    catalog_file=work/'catalog.json'
    if catalog_file.exists(): catalog=read(catalog_file)
    else:
        catalog=describe_pack([meme_dir,sticker_dir],cache/'assets',session,max_new=24) if effects else {'assets': [], 'ready': 0, 'pending': 0, 'errors': []}
        atomic_json(catalog_file,catalog)
    print('[fifth-stage] План истории',flush=True)
    plan_file=work/'director-plan.json'
    if plan_file.exists(): plan=validate_plan(read(plan_file),transcript)
    else:
        sources={'stock'}
        if effects and any(e.get('description',{}).get('kind')=='meme' for e in catalog['assets']): sources.add('meme')
        plan=plan_story(transcript,work,cache_dir=cache/'gemini',client=client,session=session,allowed_sources=sources)
        if not effects:
            for beat in plan['beats']:beat.update(emoji=None,sound='none')
        atomic_json(plan_file,plan)
    selection_file=work/'selection.json'
    selection=read(selection_file) if selection_file.exists() else None
    selected=(selection or {}).get('selections',{})
    pending=[b for b in plan['beats'] if b['id'] not in selected]
    if pending:
        print('[fifth-stage] Подбор кадров по истории',flush=True)
        ranker=None
        if semantic:
            try:
                from .multimodal import get_clip_ranker
                ranker=get_clip_ranker()
            except Exception:
                print('[fifth] CLIP unavailable; using retrieval order',flush=True)
        candidates=prepare_candidates(dict(plan,beats=pending),catalog,cache/'media',ranker=ranker)
        selection=choose_batches(plan,candidates,session,selection_file,previous=selection)
        # One bounded repair for unresolved scenes. A failed provider is not retried here.
        unresolved=set(selection['unresolved'])
        if unresolved and not session.failed and session.stats['logical_calls']<max_calls:
            print('[fifth-stage] Повторный подбор только нерешённых сцен',flush=True)
            repair=[]
            for beat in plan['beats']:
                if beat['id'] not in unresolved:continue
                row=dict(beat)
                row['queries']=beat['queries'][1:]+beat['queries'][:1]
                # Use stock as an alternative full shot if the local pack has no usable meme.
                if row['source']=='meme': row.update(source='stock')
                repair.append(row)
            candidates=prepare_candidates(dict(plan,beats=repair),catalog,cache/'media',ranker=ranker)
            repair_by_id={b['id']:b for b in repair}
            repaired_plan=dict(plan,beats=[repair_by_id.get(b['id'],b) for b in plan['beats']])
            selection=choose_batches(repaired_plan,candidates,session,selection_file,previous=selection)
            # Persist the source actually selected, retaining original wording and timing.
            for beat in plan['beats']:
                if beat['id'] in selection['selections']:
                    beat['source']=selection['selections'][beat['id']]['selected']['source']
            atomic_json(plan_file,plan)
    session.save()
    if selection is None or selection.get('unresolved'):
        ids=', '.join((selection or {}).get('unresolved',[]))
        raise RuntimeError('Подбор не завершён: '+ids+'. Прогресс сохранён. Нажми «Продолжить».')
    print('[fifth-stage] Сборка сцен и финальный рендер',flush=True)
    return render_fifth(plan,selection,transcript,audio,output,work/'render',cache=cache/'media',catalog=catalog,effects=effects)


def edit_fifth(work, output, *, cache, beat_id, action):
    """Operate only on already approved alternatives; no API calls or arbitrary paths."""
    work=Path(work);plan=read(work/'director-plan.json');selection=read(work/'selection.json')
    beat=next((b for b in plan['beats'] if b['id']==beat_id),None)
    if beat is None:raise ValueError('Неизвестная сцена')
    if action=='alternative':
        row=selection['selections'].get(beat_id,{})
        alternatives=row.get('alternatives',[])
        if not alternatives:raise ValueError('Для этой сцены нет сохранённой альтернативы')
        previous=row['selected'];row['selected']=alternatives[0]
        row['alternatives']=alternatives[1:]+[previous]
    elif action=='calmer':beat['camera']='none'
    elif action=='no-emoji':beat['emoji']=None
    else:raise ValueError('Неизвестное действие')
    atomic_json(work/'director-plan.json',plan);atomic_json(work/'selection.json',selection)
    timeline=read(work/'render'/'timeline.json')
    return render_fifth(plan,selection,load_transcript(work/'transcript.json'),timeline['audio'],output,work/'render',
                        cache=Path(cache)/'media',catalog=read(work/'catalog.json'),effects=read(work/'input.json')['effects'])
