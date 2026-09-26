import io
import json
import urllib.error
import urllib.parse

import pytest

from video_ai import stock_images as photos


@pytest.mark.parametrize('provider',['pexels','pixabay'])
def test_photo_api_contract_and_24_hour_cache(tmp_path,monkeypatch,provider):
    key='fixture-secret-key'
    monkeypatch.setenv(f'{provider.upper()}_API_KEY',key)
    monkeypatch.delenv('PIXABAY_API_KEY' if provider=='pexels' else 'PEXELS_API_KEY',raising=False)
    now=[100000.0];requests=[]
    monkeypatch.setattr(photos.time,'time',lambda:now[0])
    def urlopen(request,**kw):
        requests.append(request)
        query=urllib.parse.parse_qs(urllib.parse.urlparse(request.full_url).query)
        if provider=='pexels':
            assert request.get_header('Authorization')==key and key not in request.full_url
            assert query['query']==['getting dressed']
            payload={'photos':[{'id':1,'alt':'Getting dressed','photographer':'Creator','url':'https://example.org/page',
                                 'width':2000,'height':3000,'src':{'medium':'https://example.org/preview.jpg',
                                                               'large2x':'https://example.org/large.jpg'}}]}
        else:
            assert query['key']==[key] and query['q']==['getting dressed']
            assert query['image_type']==['photo'] and query['safesearch']==['true']
            payload={'hits':[{'id':1,'tags':'Getting dressed','user':'Creator','pageURL':'https://example.org/page',
                              'imageWidth':2000,'imageHeight':3000,'webformatURL':'https://example.org/preview.jpg',
                              'largeImageURL':'https://example.org/large.jpg'}]}
        return io.BytesIO(json.dumps(payload).encode())
    monkeypatch.setattr(photos.urllib.request,'urlopen',urlopen)
    first=photos.search_stock_images('getting dressed',cache_dir=tmp_path)
    second=photos.search_stock_images('getting dressed',cache_dir=tmp_path)
    assert first==second and len(requests)==1
    row=first[0]
    assert row['download_url']=='https://example.org/large.jpg'
    assert row['preview_url']=='https://example.org/preview.jpg'
    assert row['artist']=='Creator' and row['source']==provider and row['height']==3000
    assert row['license_url'] and row['page_url']=='https://example.org/page'
    assert key not in ''.join(p.read_text() for p in tmp_path.glob('*.json'))
    now[0]+=86401
    photos.search_stock_images('getting dressed',cache_dir=tmp_path)
    assert len(requests)==2


def test_provider_failure_keeps_other_results_without_logging_key(tmp_path,monkeypatch,capsys):
    monkeypatch.setenv('PEXELS_API_KEY','pexels-private')
    monkeypatch.setenv('PIXABAY_API_KEY','pixabay-private')
    def search(provider,*args):
        if provider=='pixabay':
            raise urllib.error.HTTPError('https://example.org/?key=pixabay-private',429,'key=pixabay-private',{},None)
        return [{'download_url':'https://example.org/photo.jpg','source':'pexels'}]
    monkeypatch.setattr(photos,'_search',search)
    rows=photos.search_stock_images('dressing',cache_dir=tmp_path)
    assert len(rows)==1 and rows[0]['source']=='pexels'
    log=capsys.readouterr().out
    assert 'HTTP 429' in log and 'pixabay-private' not in log and 'pexels-private' not in log
