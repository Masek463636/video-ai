"""Stock photo search for story mode, using the existing provider keys.

Provider contracts: https://www.pexels.com/api/documentation/
and https://pixabay.com/api/docs/ . Search responses are cached for 24 hours.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.parse
import urllib.request


def _search(provider, query, key, limit):
    if provider == 'pexels':
        url = 'https://api.pexels.com/v1/search?' + urllib.parse.urlencode({'query':query, 'per_page':limit})
        headers = {'Authorization':key}
    else:
        url = 'https://pixabay.com/api/?' + urllib.parse.urlencode({
            'key':key, 'q':query[:100], 'image_type':'photo', 'safesearch':'true', 'per_page':max(3,limit),
        })
        headers = {}
    headers['User-Agent'] = 'video-ai/2.0.0'
    with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=25) as response:
        payload = json.load(response)
    rows = []
    for item in payload.get('photos' if provider == 'pexels' else 'hits', []):
        if provider == 'pexels':
            sizes = item.get('src') or {}
            download = sizes.get('large2x') or sizes.get('large') or sizes.get('original')
            preview = sizes.get('medium') or download
            title = str(item.get('alt') or f'Pexels photo {item.get("id", "")}')
            page, artist = item.get('url'), item.get('photographer')
            width, height = item.get('width',0), item.get('height',0)
        else:
            download = item.get('largeImageURL') or item.get('webformatURL')
            preview = item.get('webformatURL') or download
            title = str(item.get('tags') or f'Pixabay photo {item.get("id", "")}')
            page, artist = item.get('pageURL'), item.get('user')
            width, height = item.get('imageWidth',0), item.get('imageHeight',0)
        if not isinstance(download,str) or not download.startswith('https://'):
            continue
        rows.append({'kind':'image','source':provider,'title':title,'description':title,
                     'download_url':download,'preview_url':preview,'page_url':page or '',
                     'artist':artist or '', 'width':width,'height':height,
                     'license':'Pexels License' if provider=='pexels' else 'Pixabay Content License',
                     'license_url':'https://www.pexels.com/license/' if provider=='pexels' else 'https://pixabay.com/service/license-summary/'})
    return rows


def _cached_search(provider, query, key, limit, cache_dir):
    cached = None
    if cache_dir is not None:
        # The key itself and the authenticated request URL are never persisted.
        signature = hashlib.sha256(json.dumps([provider,query,limit,key]).encode()).hexdigest()
        cached = Path(cache_dir) / (signature + '.json')
        try:
            data = json.loads(cached.read_text(encoding='utf-8'))
            age = time.time()-float(data['created'])
            if 0 <= age < 86400 and isinstance(data['items'],list):
                return data['items']
        except (OSError,ValueError,KeyError,TypeError):
            pass
    rows = _search(provider,query,key,limit)
    if cached is not None:
        try:
            cached.parent.mkdir(parents=True,exist_ok=True)
            temporary = cached.with_suffix('.tmp')
            temporary.write_text(json.dumps({'created':time.time(),'items':rows},ensure_ascii=False),encoding='utf-8')
            temporary.replace(cached)
        except OSError:
            print('[story] photo search cache unavailable; using current results',flush=True)
    return rows


def search_stock_images(query, *, limit=6, cache_dir=None):
    if cache_dir is None:
        # Share the 24-hour response cache across studio jobs and CLI runs.
        cache_dir = Path(os.getenv('LOCALAPPDATA') or Path.home()/'.cache')/'video-ai'/'stock-images'
    providers = [(p,os.getenv(f'{p.upper()}_API_KEY','').strip()) for p in ('pexels','pixabay')]
    groups = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        pending = [(p,pool.submit(_cached_search,p,query,k,max(3,min(limit,20)),cache_dir)) for p,k in providers if k]
        for provider, result in pending:
            try:
                groups.append(result.result())
            except Exception as error:
                # HTTPError.__str__ can contain Pixabay's key-bearing URL.
                status = getattr(error,'code',None)
                print(f'[story] {provider} photo search failed: {"HTTP "+str(status) if status else type(error).__name__}',flush=True)
    return [row for group in groups for row in group]
