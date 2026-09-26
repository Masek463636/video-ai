from types import SimpleNamespace
from unittest.mock import patch
import pytest
from video_ai.shorts_fx import _score_overlay_candidates, _find_png

@pytest.mark.parametrize('accept,score,quality,expected', [(False,99,99,0),(True,50,99,0),(True,90,30,0),(True,90,90,1)])
def test_metadata_cannot_override_visual_judgement(tmp_path, accept, score, quality, expected):
    candidate = SimpleNamespace(title='phone phone smartphone')
    class Client:
        def judge_visual(self, scene, *args, **kwargs):
            assert 'incoming call' in scene.caption
            return SimpleNamespace(accept=accept, score=score, quality_score=quality)
    result = _score_overlay_candidates('phone', [(candidate,tmp_path/'a.png',999)], client=Client(), prompt='incoming call')
    assert len(result) == expected

def test_outage_skips_inserts_without_retrying_entire_pool(tmp_path):
    class Client:
        calls = 0
        def judge_visual(self, *args, **kwargs):
            self.calls += 1
            raise RuntimeError('unavailable')
    client = Client()
    rows = [(SimpleNamespace(title='phone'), tmp_path/'a.png',99)]*8
    assert _score_overlay_candidates('phone',rows,client=client,prompt='phone') == []
    assert client.calls == 1
    assert _score_overlay_candidates('phone',rows,client=None,prompt='phone') == []

def test_rejected_regular_images_are_not_resurrected(tmp_path):
    candidate = SimpleNamespace(kind='image',mime='image/jpeg',download_url='https://example.invalid/a.jpg',width=800,height=600,title='phone',description='phone')
    def download(url,path):
        path.write_bytes(b'preview')
    with patch('video_ai.shorts_fx.search_commons',return_value=[candidate]), patch('video_ai.shorts_fx._download',side_effect=download), patch('video_ai.shorts_fx._score_overlay_candidates',return_value=[]), patch('video_ai.shorts_fx._meme_cardize',side_effect=AssertionError('rejected image rendered')):
        assert _find_png('phone',tmp_path/'result.png',None,caption='incoming call') is None
    assert not list(tmp_path.glob('*'))
