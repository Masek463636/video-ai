from __future__ import annotations

import html
import json
import re
import shutil
import tempfile
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .meme_library import search_memes
from .models import Scene, ShotPlan
from .openverse import search_openverse
from .quality_guard import infer_tone, local_quality_guard
from .stock_video import search_stock_videos
from .vision import detect_focus

COMMONS_API = "https://commons.wikimedia.org/w/api.php"
USER_AGENT = "video-ai/1.2 (+https://github.com/Masek463636/video-ai)"
_MAX_DOWNLOAD_BYTES = 120 * 1024 * 1024
_TAG_RE = re.compile(r"<[^>]+>")
_TOKEN_RE = re.compile(r"[\w-]+", flags=re.UNICODE)


@dataclass(slots=True)
class AssetCandidate:
    title: str; page_url: str; download_url: str; mime: str; width: int; height: int; size: int; kind: str
    license: str = ""; license_url: str = ""; artist: str = ""; credit: str = ""; description: str = ""
    score: float = 0.0; semantic_score: float | None = None; source: str = "commons"; local_path: str | None = None


def search_commons(query: str, *, limit: int = 20) -> list[AssetCandidate]:
    query = query.strip()
    if not query: return []
    params = {"action":"query","generator":"search","gsrsearch":query,"gsrnamespace":6,"gsrlimit":max(1,min(limit,30)),"prop":"imageinfo","iiprop":"url|mime|size|extmetadata","iiurlwidth":1600,"format":"json","formatversion":2}
    payload = _json_get(COMMONS_API + "?" + urllib.parse.urlencode(params)); out: list[AssetCandidate] = []
    for page in payload.get("query", {}).get("pages", []):
        infos = page.get("imageinfo") or []
        if not infos: continue
        info = infos[0]; mime = str(info.get("mime") or "").lower(); kind = _kind_from_mime(mime)
        if kind is None: continue
        size = int(info.get("size") or 0)
        if kind == "video" and size > _MAX_DOWNLOAD_BYTES: continue
        url = str(info.get("thumburl") or info.get("url") or "") if kind == "image" else str(info.get("url") or "")
        if not url: continue
        meta = info.get("extmetadata") or {}
        description = " ".join(x for x in (_clean_html(_meta(meta,"ImageDescription")),_clean_html(_meta(meta,"ObjectName")),_clean_html(_meta(meta,"Categories"))) if x)
        out.append(AssetCandidate(str(page.get("title") or "").removeprefix("File:"),str(info.get("descriptionurl") or ""),url,mime,int(info.get("width") or 0),int(info.get("height") or 0),size,kind,_meta(meta,"LicenseShortName"),_meta(meta,"LicenseUrl"),_clean_html(_meta(meta,"Artist")),_clean_html(_meta(meta,"Credit")),description,source="commons"))
    return out


def search_all(query: str, *, limit: int = 20, include_stock_video: bool = False, archive_only: bool = False) -> list[AssetCandidate]:
    pool: dict[str, AssetCandidate] = {}
    try:
        for c in search_commons(query, limit=limit): pool.setdefault(c.download_url,c)
    except Exception: pass
    try:
        for item in search_openverse(query, limit=limit):
            c=AssetCandidate(item.title,item.page_url,item.download_url,"image/jpeg",item.width,item.height,0,"image",item.license,item.license_url,item.artist,item.artist,item.description,source="openverse"); pool.setdefault(c.download_url,c)
    except Exception: pass
    if include_stock_video and not archive_only:
        try:
            for item in search_stock_videos(query, limit=min(limit,16)):
                c=AssetCandidate(item.title,item.page_url,item.download_url,"video/mp4",item.width,item.height,0,"video",item.license,description=query,source=item.source); pool.setdefault(c.download_url,c)
        except Exception: pass
    return list(pool.values())


def materialize_assets(plan: ShotPlan, out_dir: str | Path, *, limit: int = 20, overwrite: bool = False, semantic: bool = False, semantic_top_k: int = 6, replace_scenes: set[int] | None = None, rank_offset: int = 0, meme_dir: str | Path | None = None) -> list[dict]:
    out_dir=Path(out_dir); out_dir.mkdir(parents=True,exist_ok=True); replace_scenes=replace_scenes or set(); used_urls:set[str]=set(); used_titles:list[str]=[]; manifest:list[dict]=[]
    try:
        from .gemini_ai import get_gemini_client
        gemini=get_gemini_client()
    except Exception: gemini=None

    for scene in plan.scenes:
        # Local tone inference is authoritative fallback. Gemini may plan the
        # visual, but this makes tone protection work even if Director omitted it.
        scene.tone = infer_tone(scene.caption)

    for index,scene in enumerate(plan.scenes):
        if index not in replace_scenes and scene.asset and Path(scene.asset).exists(): used_titles.append(Path(scene.asset).stem)

    for index,scene in enumerate(plan.scenes):
        force_replace=index in replace_scenes
        if scene.asset and not overwrite and not force_replace and Path(scene.asset).exists():
            _apply_focus(scene,Path(scene.asset)); manifest.append({"scene":index,"status":"existing","path":scene.asset,"tone":scene.tone}); continue

        pool: dict[str, tuple[AssetCandidate,str]] = {}; variants=list(_query_variants(scene))
        if (scene.visual_mode=="meme" or scene.source_mode=="meme_library") and not scene.semantic_lock:
            memes=search_memes(meme_dir,scene.visual_description or scene.query,limit=20)
            if scene.meme_filename: memes.sort(key=lambda m:0 if m.path.name==scene.meme_filename else 1)
            for meme in memes:
                key=f"local:{meme.path.resolve()}"; exact=100.0 if scene.meme_filename and meme.path.name==scene.meme_filename else 0.0
                pool[key]=(AssetCandidate(meme.title,"",key,"video/mp4" if meme.kind=="video" else "image/jpeg",0,0,meme.path.stat().st_size,meme.kind,"local user library",source="local_meme",local_path=str(meme.path.resolve()),score=30.0+meme.score+exact),"local meme library")

        archive_only=scene.semantic_lock or scene.source_mode=="historical_archive"
        include_stock=(scene.source_mode=="stock_video" or (scene.source_mode=="auto" and scene.visual_mode=="video")) and not scene.semantic_lock
        allow_public=scene.source_mode!="meme_library" or not pool or scene.semantic_lock
        if allow_public:
            for query_index,variant in enumerate(variants):
                for candidate in search_all(variant,limit=limit,include_stock_video=include_stock,archive_only=archive_only):
                    if candidate.download_url in used_urls or not _source_allowed(scene,candidate): continue
                    ranking_query=" ".join(x for x in (variant,scene.visual_description or "") if x).strip()
                    candidate.score=_score(candidate,ranking_query,scene.visual_description or "")+_visual_mode_bonus(scene,candidate)+_source_mode_bonus(scene,candidate)+_semantic_lock_bonus(scene,candidate)-query_index*0.32-_repeat_penalty(candidate.title,used_titles)
                    existing=pool.get(candidate.download_url)
                    if existing is None or candidate.score>existing[0].score: pool[candidate.download_url]=(candidate,variant)

        ranked=sorted(pool.values(),key=lambda item:item[0].score,reverse=True)
        if semantic and ranked:
            _semantic_rerank(scene,ranked,top_k=semantic_top_k); ranked.sort(key=lambda item:item[0].score,reverse=True)

        chosen=None; target=None; search_used=""; judge_info=None; rejected:list[dict]=[]; local_rejected:list[dict]=[]; gemini_checks=0
        max_checks=6 if scene.semantic_lock else 5
        for candidate,search in ranked[max(0,rank_offset):]:
            suffix=_suffix(candidate); candidate_target=out_dir/f"scene_{index:03d}{suffix}"
            try:
                if candidate.local_path: shutil.copy2(candidate.local_path,candidate_target)
                else: _download(candidate.download_url,candidate_target)
                if not candidate_target.exists() or candidate_target.stat().st_size<1024: candidate_target.unlink(missing_ok=True); continue
            except Exception:
                candidate_target.unlink(missing_ok=True); continue

            local_guard=local_quality_guard(candidate_target,title=candidate.title,description=candidate.description,width=candidate.width,height=candidate.height,kind=candidate.kind)
            if not local_guard.accept:
                local_rejected.append({"title":candidate.title,"source":candidate.source,"quality_score":local_guard.score,"issues":local_guard.issues})
                candidate_target.unlink(missing_ok=True); continue

            # If Gemini is available, every candidate that can become final must
            # pass it. v1.1's unjudged fallback is intentionally removed.
            if gemini is not None:
                if gemini_checks>=max_checks:
                    candidate_target.unlink(missing_ok=True); continue
                gemini_checks+=1
                judgement=gemini.judge_visual(scene,candidate_target,candidate_title=candidate.title,source=candidate.source)
                if judgement is None:
                    candidate_target.unlink(missing_ok=True); continue
                judge_info={"score":judgement.score,"accept":judgement.accept,"reason":judgement.reason,"mismatch":judgement.mismatch,"tone_match":judgement.tone_match,"quality_score":judgement.quality_score,"quality_issues":judgement.quality_issues or []}
                if not judgement.accept:
                    rejected.append({"title":candidate.title,"source":candidate.source,"score":judgement.score,"tone_match":judgement.tone_match,"quality_score":judgement.quality_score,"quality_issues":judgement.quality_issues or [],"reason":judgement.reason,"mismatch":judgement.mismatch})
                    candidate_target.unlink(missing_ok=True); continue
                candidate.score+=(judgement.score+judgement.tone_match+judgement.quality_score)/36.0
            chosen,target,search_used=candidate,candidate_target,search; break

        if chosen is None or target is None:
            scene.asset=None; scene.asset_kind="blank"; scene.focus_x=scene.focus_y=None; scene.focus_source=None; scene.asset_score=scene.semantic_score=None
            manifest.append({"scene":index,"status":"not_found_locked" if scene.semantic_lock else "not_found_guarded","query":scene.query,"tone":scene.tone,"visual_mode":scene.visual_mode,"source_mode":scene.source_mode,"semantic_lock":scene.semantic_lock,"required_entities":scene.required_entities,"required_context":scene.required_context,"semantic_fallback":scene.semantic_fallback,"motion_preset":scene.motion_preset,"visual_description":scene.visual_description,"queries_tried":variants,"local_quality_rejected":local_rejected,"gemini_rejected":rejected}); continue

        used_urls.add(chosen.download_url); used_titles.append(chosen.title); scene.asset=str(target.resolve()); scene.asset_kind=chosen.kind  # type: ignore[assignment]
        scene.asset_score=round(chosen.score,4); scene.semantic_score=round(chosen.semantic_score,4) if chosen.semantic_score is not None else None; _apply_focus(scene,target)
        manifest.append({"scene":index,"status":"downloaded","query":scene.query,"tone":scene.tone,"visual_mode":scene.visual_mode,"source_mode":scene.source_mode,"semantic_lock":scene.semantic_lock,"required_entities":scene.required_entities,"required_context":scene.required_context,"semantic_fallback":scene.semantic_fallback,"motion_preset":scene.motion_preset,"meme_filename":scene.meme_filename,"visual_description":scene.visual_description,"queries_tried":variants,"search_used":search_used,"path":str(target),"source":chosen.source,"kind":chosen.kind,"gemini_judge":judge_info,"local_quality_rejected":local_rejected,"gemini_rejected":rejected,"top_candidates":[{"title":c.title,"score":round(c.score,3),"search":q,"source":c.source,"kind":c.kind,"semantic_score":round(c.semantic_score,4) if c.semantic_score is not None else None} for c,q in ranked[:8]],**asdict(chosen)})

    ensure_visual_coverage(plan,manifest); (out_dir/"assets_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); return manifest


def ensure_visual_coverage(plan: ShotPlan, manifest: list[dict] | None = None) -> set[int]:
    good=[i for i,s in enumerate(plan.scenes) if s.asset and Path(s.asset).exists() and s.asset_kind!="blank"]
    if not good: return set()
    filled:set[int]=set(); by_scene={int(i.get("scene")):i for i in (manifest or []) if isinstance(i.get("scene"),int)}
    for index,scene in enumerate(plan.scenes):
        if scene.asset and Path(scene.asset).exists() and scene.asset_kind!="blank": continue
        if scene.semantic_lock:
            compatible=[i for i in good if plan.scenes[i].semantic_lock and _lock_compatible(scene,plan.scenes[i])]
            if not compatible: continue
        else:
            # Tone-safe fallback only. Tragic/violent/negative scenes may borrow
            # from each other, but never from cheerful/neutral beats by default.
            compatible=[i for i in good if _tone_compatible(scene.tone,plan.scenes[i].tone) and (plan.scenes[i].source_mode==scene.source_mode or scene.source_mode in {"auto","generic_image"})]
            if not compatible: continue
        source_index=min(compatible,key=lambda other:(abs(other-index),0 if other<index else 1,other)); source=plan.scenes[source_index]
        scene.asset=source.asset; scene.asset_kind=source.asset_kind; scene.focus_x=source.focus_x; scene.focus_y=source.focus_y; scene.focus_source=f"fallback_nearest:{source_index}"; scene.asset_score=source.asset_score; scene.semantic_score=source.semantic_score; scene.motion_preset="micro_push" if source.asset_kind=="video" else "slow_push"; filled.add(index)
        if index in by_scene: by_scene[index].update({"status":"fallback_nearest_tone_safe","fallback_from_scene":source_index,"path":scene.asset})
    return filled


def _tone_compatible(a: str,b: str)->bool:
    dark={"tragic","negative","violent","tense","shocking"}; light={"positive","funny","victorious","absurd"}
    if a in dark:return b in dark or b in {"emotional","mysterious"}
    if a in light:return b in light or b=="neutral"
    return b not in ({"positive","funny","victorious"} if a in dark else set())


def _lock_compatible(target: Scene, source: Scene) -> bool:
    a={x.lower() for x in target.required_entities}; b={x.lower() for x in source.required_entities}
    ca={x.lower() for x in target.required_context}; cb={x.lower() for x in source.required_context}
    return (not a or bool(a & b)) and (not ca or bool(ca & cb))


def _source_allowed(scene: Scene, candidate: AssetCandidate) -> bool:
    if scene.semantic_lock: return candidate.source in {"commons","openverse"} and candidate.kind=="image"
    if scene.source_mode=="historical_archive": return candidate.source in {"commons","openverse"} and candidate.kind=="image"
    if scene.source_mode=="stock_video": return candidate.source in {"pexels","pixabay","commons","openverse"}
    if scene.source_mode=="meme_library": return candidate.source=="local_meme"
    if scene.source_mode=="generic_image": return candidate.kind=="image"
    return True


def _query_variants(scene: Scene) -> Iterable[str]:
    seen:set[str]=set(); locked=" ".join([*scene.required_entities,*scene.required_context]).strip(); base=[]
    tone_hint=_tone_query_hint(scene.tone)
    if scene.semantic_lock and locked:
        base += [f"{locked} {scene.visual_description or ''}",f"{locked} historical illustration",f"{locked} archival portrait",scene.semantic_fallback or ""]
    base += [f"{q} {tone_hint}".strip() for q in (scene.search_queries or [])]
    base += [f"{scene.visual_description or ''} {tone_hint}".strip(),scene.query]
    if scene.source_mode=="historical_archive": base += [f"{scene.visual_description or scene.query} archival engraving",f"{scene.visual_description or scene.query} historical painting"]
    elif scene.visual_mode=="video": base += [f"{scene.visual_description or scene.query} {tone_hint} documentary footage",f"{scene.visual_description or scene.query} b roll"]
    elif scene.visual_mode=="meme": base += [f"{scene.visual_description or scene.query} reaction"]
    else: base += [f"{scene.visual_description or scene.query} {tone_hint} photo illustration"]
    for value in base:
        value=re.sub(r"\s+"," ",value).strip()
        if value and value.lower() not in seen: seen.add(value.lower()); yield value


def _tone_query_hint(tone:str)->str:
    return {"tragic":"somber aftermath destruction mourning","violent":"conflict destruction tense","negative":"somber disappointed dark","tense":"tense anxious dramatic","shocking":"dramatic shocking serious","mysterious":"mysterious surreal atmospheric","religious":"religious sacred painting","positive":"uplifting hopeful","victorious":"victory triumphant","funny":"funny reaction","emotional":"emotional expressive"}.get(tone,"")


def _semantic_lock_bonus(scene: Scene, candidate: AssetCandidate) -> float:
    if not scene.semantic_lock: return 0.0
    hay=" ".join([candidate.title,candidate.description]).lower(); bonus=0.0
    for entity in scene.required_entities:
        if all(tok in hay for tok in _tokens(entity)): bonus+=14.0
        else: bonus-=8.0
    for ctx in scene.required_context:
        toks=_tokens(ctx)
        if toks and any(tok in hay for tok in toks): bonus+=5.0
    return bonus


def _source_mode_bonus(scene: Scene,candidate: AssetCandidate)->float:
    if scene.source_mode=="historical_archive": return 30.0 if candidate.source=="commons" else 16.0 if candidate.source=="openverse" else -100.0
    if scene.source_mode=="stock_video": return 22.0 if candidate.source in {"pexels","pixabay"} and candidate.kind=="video" else 2.0
    if scene.source_mode=="meme_library": return 100.0 if candidate.source=="local_meme" else -100.0
    if scene.source_mode=="generic_image": return 8.0 if candidate.kind=="image" else -10.0
    return 0.0


def _visual_mode_bonus(scene: Scene,candidate: AssetCandidate)->float:
    if scene.visual_mode=="video": return 18.0 if candidate.kind=="video" else -5.0
    if scene.visual_mode=="image": return 4.0 if candidate.kind=="image" else -1.0
    if scene.visual_mode=="meme": return 30.0 if candidate.source=="local_meme" else -10.0
    return 0.0


def _semantic_rerank(scene: Scene, ranked: list[tuple[AssetCandidate,str]], *, top_k:int)->None:
    pairs=[(c,q) for c,q in ranked if c.kind=="image"][:max(1,top_k)]
    if not pairs:return
    try:
        from .multimodal import ClipRanker
        ranker=ClipRanker()
    except Exception:return
    prompt=" ".join([*(scene.required_entities if scene.semantic_lock else []),*(scene.required_context if scene.semantic_lock else []),scene.visual_description or scene.query or "documentary scene",_tone_query_hint(scene.tone)]).strip()
    with tempfile.TemporaryDirectory(prefix="video-ai-clip-") as d:
        paths=[]; valid=[]
        for idx,(c,_) in enumerate(pairs):
            p=Path(d)/f"candidate_{idx:02d}{_suffix(c)}"
            try:
                if c.local_path: shutil.copy2(c.local_path,p)
                else:_download(c.download_url,p)
            except Exception: continue
            paths.append(p); valid.append(c)
        if not paths:return
        try:scores=ranker.score_images(prompt,paths)
        except Exception:return
        for c,sim in zip(valid,scores): c.semantic_score=float(sim); c.score+=float(sim)*28.0


def _apply_focus(scene: Scene,path: Path)->None:
    if scene.asset_kind!="image":return
    x,y,source=detect_focus(path); scene.focus_x=round(x,4); scene.focus_y=round(y,4); scene.focus_source=source


def _score(candidate:AssetCandidate,query:str,caption:str)->float:
    wanted=_tokens(query+" "+caption); title=_tokens(candidate.title); desc=_tokens(candidate.description); score=len(wanted&title)*7.0+len(wanted&desc)*2.5
    if candidate.width>=1200 or candidate.height>=1200: score+=2.5
    elif candidate.width>=800 or candidate.height>=800: score+=1.0
    if candidate.height and candidate.width:
        ratio=candidate.width/candidate.height
        if 0.45<=ratio<=0.8: score+=3.5
        elif ratio<=1.1: score+=1.5
    if candidate.license:score+=0.5
    return score


def _repeat_penalty(title:str,previous_titles:list[str])->float:
    current=_tokens(title)
    if not current:return 0.0
    worst=0.0
    for previous in previous_titles[-5:]:
        other=_tokens(previous)
        if other: worst=max(worst,len(current&other)/max(1,len(current|other)))
    return worst*8.0


def _tokens(value:str)->set[str]: return {t.lower() for t in _TOKEN_RE.findall(value) if len(t)>1}

def _json_get(url:str)->dict:
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT,"Accept":"application/json"})
    with urllib.request.urlopen(req,timeout=25) as response:return json.load(response)

def _download(url:str,target:Path)->None:
    req=urllib.request.Request(url,headers={"User-Agent":USER_AGENT})
    with urllib.request.urlopen(req,timeout=60) as response,target.open("wb") as output:
        total=0
        while True:
            chunk=response.read(1024*1024)
            if not chunk:break
            total+=len(chunk)
            if total>_MAX_DOWNLOAD_BYTES:raise RuntimeError("asset too large")
            output.write(chunk)

def _kind_from_mime(mime:str)->str|None:
    if mime in {"image/jpeg","image/png","image/webp"}:return "image"
    if mime in {"video/webm","video/mp4","video/ogg"}:return "video"
    return None

def _suffix(candidate:AssetCandidate)->str:
    suffix=Path(candidate.local_path).suffix.lower() if candidate.local_path else Path(urllib.parse.urlparse(candidate.download_url).path).suffix.lower()
    if candidate.kind=="image" and suffix not in {".jpg",".jpeg",".png",".webp"}:return ".jpg"
    if candidate.kind=="video" and suffix not in {".webm",".mp4",".ogv",".ogg",".mov",".mkv",".avi"}:return ".mp4"
    return suffix or (".jpg" if candidate.kind=="image" else ".mp4")

def _meta(meta:dict,key:str)->str:
    raw=meta.get(key) or {}; return str(raw.get("value") or "") if isinstance(raw,dict) else str(raw or "")

def _clean_html(value:str)->str:return html.unescape(_TAG_RE.sub(" ",value)).strip()
