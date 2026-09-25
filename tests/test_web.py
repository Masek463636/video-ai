import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest
from video_ai.web import Studio, make_handler


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', 'test-private-key')
    monkeypatch.setenv('PEXELS_API_KEY', 'test-pexels')
    studio = Studio(tmp_path)
    server = ThreadingHTTPServer(('127.0.0.1', 0), make_handler(studio))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield studio, f'http://127.0.0.1:{server.server_port}'
    server.shutdown()
    server.server_close()


def request(url, data=None, **headers):
    req = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            return response.status, response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.read()


def test_http_boundaries(app):
    studio, url = app
    assert request(url + '/')[0] == 200
    assert request(url + '/../../pyproject.toml')[0] == 404
    assert request(url + '/api/state', Origin='http://evil.example')[0] == 403
    assert request(url + '/api/state', Host='evil.example')[0] == 403
    assert request(url + '/api/jobs', b'bad')[0] == 403
    status, body = request(url + '/api/jobs', b'bad', **{'X-Studio-Request':'1'})
    assert status == 400
    assert not studio.busy.locked()
    assert not list(studio.storage.iterdir())
    studio.busy.acquire()
    try:
        assert request(url + '/api/jobs', b'bad', **{'X-Studio-Request':'1'})[0] == 409
    finally:
        studio.busy.release()


def test_upload_pipeline_history_and_download(app, monkeypatch, tmp_path):
    studio, url = app
    audio = tmp_path / 'test.mp3'
    subprocess.run(['ffmpeg','-v','error','-f','lavfi','-i','sine=frequency=440:duration=1','-y',str(audio)],check=True)
    commands = []
    real_popen = subprocess.Popen

    def fake_renderer(command, **kwargs):
        if '-m' not in command:
            return real_popen(command, **kwargs)
        commands.append(command)
        # Real child process for log streaming and lifecycle, controlled output only.
        output = command[command.index('-o') + 1]
        return real_popen([command[0], '-c', "from pathlib import Path; import sys; Path(sys.argv[1]).write_bytes(b'test-video'); print('test-private-key')", output], **kwargs)

    monkeypatch.setattr(subprocess, 'Popen', fake_renderer)
    status, body = request(url + '/api/jobs', audio.read_bytes(), **{'X-Studio-Request':'1', 'X-Effects':'1'})
    assert status == 202
    job_id = json.loads(body)['id']
    deadline = time.monotonic() + 10
    while studio.busy.locked() and time.monotonic() < deadline:
        time.sleep(.02)
    assert not studio.busy.locked()
    job = studio.jobs[job_id]
    assert job['status'] == 'done'
    assert len(commands) == 2
    assert '--select-moments' in commands[0]
    assert '--shorts-fx' in commands[1]
    assert '--max-overlays' in commands[1]
    assert request(url + '/download/' + job_id) == (200, b'test-video')
    assert b'test-private-key' not in request(url + '/api/state')[1]
    assert Studio(tmp_path).jobs[job_id]['status'] == 'done'
    studio.update(job, status='running')
    assert Studio(tmp_path).jobs[job_id]['status'] == 'error'


def test_renderer_failure_releases_slot(tmp_path, monkeypatch):
    studio = Studio(tmp_path)
    folder = studio.storage / 'failed'
    folder.mkdir()
    job = {'id':'failed','created':time.time(),'status':'queued','stage':'','step':0,'duration':1,'effects':False,'logs':[]}
    studio.jobs['failed'] = job
    studio.busy.acquire()
    real_popen = subprocess.Popen
    def failing(command, **kwargs):
        return real_popen([command[0], '-c', "print('API unavailable'); raise SystemExit(1)"], **kwargs)
    monkeypatch.setattr(subprocess, 'Popen', failing)
    studio.run(job, {})
    assert job['status'] == 'error'
    assert not studio.busy.locked()
    assert 'API unavailable' in job['logs']
    assert not (folder / 'final.mp4').exists()
