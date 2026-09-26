"""Loopback-only UI for the existing CLI. Not a public hosting server."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
import webbrowser

KEYS = ('GEMINI_API_KEY', 'PEXELS_API_KEY', 'PIXABAY_API_KEY')
MAX_UPLOAD = 100 * 1024 * 1024


class Studio:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.storage = self.root / 'work-web'
        self.storage.mkdir(exist_ok=True)
        self.lock = threading.Lock()
        self.busy = threading.Lock()
        self.jobs = {}
        self.process = None
        self.stopping = threading.Event()
        for file in self.storage.glob('*/job.json'):
            try:
                job = json.loads(file.read_text(encoding='utf-8'))
                if job['id'] != file.parent.name:
                    continue
                if job['status'] in ('running', 'queued'):
                    job.update(status='error', stage='Прервано закрытием приложения')
                self.jobs[job['id']] = job
            except (ValueError, KeyError, OSError):
                continue

    def save(self, job):
        path = self.storage / job['id'] / 'job.json'
        temp = path.with_suffix('.tmp')
        temp.write_text(json.dumps(job, ensure_ascii=False), encoding='utf-8')
        temp.replace(path)

    def update(self, job, **values):
        with self.lock:
            job.update(values)
            self.save(job)

    def run(self, job, env):
        folder = self.storage / job['id']
        work = folder / 'base-work'
        common = [sys.executable, '-u', '-m', 'video_ai.cli']
        steps = [
            ('Подбор кадров и сборка основы', common + ['create', str(folder / 'voice.mp3'), '-o', str(folder / 'base.mp4'), '--work-dir', str(work), '--language', 'ru', '--material-v2', '--reference-framing', '--select-moments', '--no-sfx']),
            ('Оформление и финальный рендер', common + ['render', str(work / 'shot_plan.materialized.json'), '--transcript', str(work / 'transcript.json'), '--editing-polish', '--reference-framing', '-o', str(folder / 'final.mp4'), '--work-dir', str(folder / 'final-work')]),
        ]
        if job['effects']:
            steps[1][1].extend(['--shorts-fx', '--sticker-dir', str(self.root / 'stickers'), '--max-overlays', '2'])
        if job.get('style') == 'story':
            command = common + ['story', str(folder / 'voice.mp3'), '-o', str(folder / 'final.mp4'),
                                '--work-dir', str(folder / 'story-work'), '--language', 'ru',
                                '--meme-dir', str(self.root / 'memes'), '--elements-dir', str(self.root / 'elements')]
            if not job['effects']:
                command.append('--no-effects')
            steps = [('Истории и мемы: подготовка материалов', command)]
        try:
            for index, (stage, command) in enumerate(steps):
                if self.stopping.is_set():
                    raise RuntimeError('Приложение закрывается')
                self.update(job, status='running', stage=stage, step=index + 1)
                with self.lock:
                    if self.stopping.is_set():
                        raise RuntimeError('Приложение закрывается')
                    self.process = subprocess.Popen(command, cwd=self.root, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', errors='replace')
                    process = self.process
                for line in process.stdout:
                    # Logs remain bounded; API key values never go to disk or browser.
                    for key in KEYS:
                        if env.get(key):
                            line = line.replace(env[key], '[ключ скрыт]')
                    if job.get('style') == 'story' and line.startswith('[story'):
                        if 'render' in line:
                            self.update(job, stage='Сборка сцен и финальный рендер', step=2)
                        elif '[story] scene ' in line:
                            self.update(job, stage='Поиск материалов · ' + line.split(':', 1)[0].replace('[story] scene ', 'сцена '))
                        elif 'describing' in line:
                            self.update(job, stage='Разбор паков мемов и элементов')
                        elif 'planning' in line:
                            self.update(job, stage='Планирование истории')
                    with self.lock:
                        job['logs'] = (job['logs'] + [line.rstrip()])[-120:]
                if process.wait() != 0:
                    raise RuntimeError('Этап завершился с ошибкой. Подробности — в журнале.')
            if not (folder / 'final.mp4').is_file():
                raise RuntimeError('Рендер не создал итоговый файл')
            self.update(job, status='done', stage='Ролик готов', step=2)
        except Exception as error:
            self.update(job, status='error', stage=str(error))
        finally:
            with self.lock:
                self.process = None
            self.busy.release()

    def stop(self):
        self.stopping.set()
        with self.lock:
            process = self.process
        if process and process.poll() is None:
            if os.name == 'nt':
                subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True)
            else:
                process.terminate()


def make_handler(studio):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def send_data(self, data, status=200, content_type='application/json; charset=utf-8'):
            if not isinstance(data, bytes):
                data = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', content_type)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; img-src 'self' data:; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def allowed(self, mutation=False):
            expected = f'127.0.0.1:{self.server.server_port}'
            if self.headers.get('Host') != expected:
                self.send_data({'error': 'Недопустимый адрес'}, 403)
                return False
            origin = self.headers.get('Origin')
            if origin and origin != 'http://' + expected:
                self.send_data({'error': 'Недопустимый источник'}, 403)
                return False
            if mutation and self.headers.get('X-Studio-Request') != '1':
                self.send_data({'error': 'Откройте приложение в браузере'}, 403)
                return False
            return True

        def do_GET(self):
            if not self.allowed():
                return
            path = urlsplit(self.path).path
            if path == '/api/state':
                with studio.lock:
                    jobs = sorted(studio.jobs.values(), key=lambda j: j['created'], reverse=True)
                    snapshot = json.loads(json.dumps(jobs))
                self.send_data({'jobs': snapshot, 'busy': studio.busy.locked(), 'keys': {k: bool(os.environ.get(k)) for k in KEYS}, 'tools': {x: bool(shutil.which(x)) for x in ('ffmpeg', 'ffprobe')}})
            elif path.startswith('/download/'):
                job_id = path.removeprefix('/download/')
                job = studio.jobs.get(job_id)
                if not job or job['status'] != 'done':
                    self.send_data({'error': 'Результат недоступен'}, 404)
                    return
                file = studio.storage / job_id / 'final.mp4'
                if not file.is_file():
                    self.send_data({'error': 'Файл был удалён'}, 404)
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'video/mp4')
                self.send_header('Content-Length', str(file.stat().st_size))
                self.send_header('Content-Disposition', f'attachment; filename="shorts-{job_id}.mp4"')
                self.end_headers()
                with file.open('rb') as source:
                    shutil.copyfileobj(source, self.wfile)
            else:
                names = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript; charset=utf-8'), '/style.css': ('style.css', 'text/css; charset=utf-8')}
                if path not in names:
                    self.send_data({'error': 'Не найдено'}, 404)
                    return
                name, mime = names[path]
                self.send_data((Path(__file__).parent / 'web_static' / name).read_bytes(), content_type=mime)

        def do_POST(self):
            if not self.allowed(mutation=True):
                return
            if self.path != '/api/jobs':
                self.send_data({'error': 'Не найдено'}, 404)
                return
            try:
                length = int(self.headers.get('Content-Length', '0'))
            except ValueError:
                length = 0
            if not 0 < length <= MAX_UPLOAD:
                self.send_data({'error': 'Выберите MP3 размером до 100 МБ'}, 413)
                return
            if not all(shutil.which(x) for x in ('ffmpeg', 'ffprobe')):
                self.send_data({'error': 'Установите FFmpeg и добавьте его в PATH'}, 400)
                return
            style = self.headers.get('X-Style', 'classic')
            if style not in ('classic', 'story'):
                self.send_data({'error': 'Неизвестный стиль'}, 400)
                return
            if not os.environ.get('GEMINI_API_KEY') or (style == 'classic' and not any(os.environ.get(k) for k in KEYS[1:])):
                required = 'Нужен сохранённый ключ Gemini.' if style == 'story' else 'Нужны сохранённый ключ Gemini и хотя бы один ключ Pexels или Pixabay.'
                self.send_data({'error': required + ' Перезапустите приложение после настройки ключей.'}, 400)
                return
            if not studio.busy.acquire(blocking=False):
                self.send_data({'error': 'Дождитесь завершения текущего ролика'}, 409)
                return
            folder = studio.storage / secrets.token_hex(8)
            folder.mkdir()
            launched = False
            try:
                self.connection.settimeout(60)
                with (folder / 'voice.mp3').open('wb') as target:
                    remaining = length
                    while remaining:
                        chunk = self.rfile.read(min(65536, remaining))
                        if not chunk:
                            raise ValueError('Загрузка прервалась')
                        target.write(chunk)
                        remaining -= len(chunk)
                check = subprocess.run(['ffprobe', '-v', 'error', '-show_entries', 'format=duration:stream=codec_type', '-of', 'json', str(folder / 'voice.mp3')], capture_output=True, text=True, timeout=30)
                if check.returncode:
                    raise ValueError('Не удалось прочитать аудио. Загрузите исправный MP3.')
                info = json.loads(check.stdout)
                duration = float(info.get('format', {}).get('duration', 0))
                if not math.isfinite(duration) or not 0 < duration <= 180 or not any(s.get('codec_type') == 'audio' for s in info.get('streams', [])):
                    raise ValueError('Нужна озвучка длительностью от 1 до 180 секунд')
                job = {'id': folder.name, 'created': time.time(), 'status': 'queued', 'stage': 'Озвучка загружена', 'step': 0, 'duration': round(duration, 1), 'style': style, 'effects': self.headers.get('X-Effects') == '1', 'logs': []}
                with studio.lock:
                    studio.jobs[job['id']] = job
                    studio.save(job)
                threading.Thread(target=studio.run, args=(job, dict(os.environ, PYTHONIOENCODING='utf-8', PYTHONUNBUFFERED='1')), daemon=True).start()
                launched = True
                self.send_data({'id': job['id']}, 202)
            except (ValueError, OSError, subprocess.TimeoutExpired) as error:
                if not launched:
                    shutil.rmtree(folder, ignore_errors=True)
                self.send_data({'error': str(error)}, 400)
            finally:
                if not launched:
                    studio.busy.release()
                    shutil.rmtree(folder, ignore_errors=True)
    return Handler


def main():
    parser = argparse.ArgumentParser(description='Локальная студия video-ai')
    parser.add_argument('--port', type=int, default=8765)
    parser.add_argument('--project-dir', type=Path, default=Path.cwd())
    parser.add_argument('--no-browser', action='store_true')
    args = parser.parse_args()
    studio = Studio(args.project_dir)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), make_handler(studio))
    url = f'http://127.0.0.1:{server.server_port}'
    print(f'Video AI Studio: {url}\nДля остановки нажмите Ctrl+C.', flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        studio.stop()
        server.server_close()


if __name__ == '__main__':
    main()
