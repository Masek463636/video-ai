'use strict';
const $ = id => document.getElementById(id);
let selectedFile, previewUrl, selectedJob, state, sending = false;
function choose(file) {
  if (!file) return;
  if (!file.name.toLowerCase().endsWith('.mp3') || file.size > 100 * 1024 * 1024) {
    selectedFile = null; $('audio').value = ''; $('audio').required = true;
    $('filename').textContent = 'Выбрать озвучку'; $('audio-preview').hidden = true;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    $('message').textContent = 'Выбери MP3 размером до 100 МБ.'; return;
  }
  selectedFile = file;
  $('filename').textContent = file.name;
  $('message').textContent = '';
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  $('audio-preview').src = previewUrl; $('audio-preview').hidden = false;
}
$('audio').addEventListener('change', e => choose(e.target.files[0]));
for (const type of ['dragover', 'dragleave', 'drop']) $('drop').addEventListener(type, e => {
  e.preventDefault(); $('drop').classList.toggle('dragging', type === 'dragover');
  if (type === 'drop') { choose(e.dataTransfer.files[0]); $('audio').required = !selectedFile; }
});
function showJob(job) {
  if (!job) return;
  $('empty').hidden = true; $('job').hidden = false;
  $('stage').textContent = job.stage;
  const running = ['running', 'queued'].includes(job.status);
  $('status-badge').textContent = ({done:'Готово',error:'Ошибка',running:'В работе',queued:'Запуск'})[job.status];
  $('step1').classList.toggle('active', job.step >= 1);
  $('step2').classList.toggle('active', job.step >= 2);
  if (running) $('progress').removeAttribute('value'); else $('progress').value = job.status === 'done' ? 2 : Math.max(0, job.step - 1);
  $('working-note').hidden = !running;
  $('elapsed').textContent = running ? `Прошло ${Math.floor((Date.now()/1000 - job.created)/60)} мин · озвучка ${job.duration} с` : `Озвучка ${job.duration} с`;
  $('download').hidden = job.status !== 'done'; $('download').href = '/download/' + job.id;
  const log = $('logs'), atBottom = log.scrollHeight - log.scrollTop - log.clientHeight < 30;
  log.textContent = job.logs.join('\n') || 'Запускаем обработку…';
  if (atBottom) log.scrollTop = log.scrollHeight;
  if (job.status === 'error') $('log-details').open = true;
}
async function refresh() {
  try {
    const response = await fetch('/api/state');
    if (!response.ok) throw new Error('Сервер недоступен');
    state = await response.json();
    $('create').disabled = state.busy || sending;
    $('create').textContent = state.busy ? 'Ролик обрабатывается…' : 'Создать ролик ↗';
    const names = {GEMINI_API_KEY:'Gemini',PEXELS_API_KEY:'Pexels',PIXABAY_API_KEY:'Pixabay',ffmpeg:'FFmpeg',ffprobe:'Проверка аудио'};
    $('connections').replaceChildren();
    for (const [key, ok] of Object.entries({...state.keys,...state.tools})) {
      const row = document.createElement('div'); row.className = 'key';
      const name = document.createElement('span'); name.textContent = names[key];
      const status = document.createElement('span'); status.className = ok ? 'good' : 'missing'; status.textContent = ok ? (key.endsWith('KEY') ? 'Ключ найден' : 'Готово') : 'Не найден';
      row.append(name,status); $('connections').append(row);
    }
    $('connection-count').textContent = `${Object.values(state.keys).filter(Boolean).length}/3 ключей`;
    if (!selectedJob && state.jobs.length) selectedJob = state.jobs[0].id;
    showJob(state.jobs.find(j=>j.id === selectedJob));
    $('history-count').textContent = state.jobs.length ? String(state.jobs.length) : '';
    if (state.jobs.length) {
      $('history-list').replaceChildren();
      for (const job of state.jobs) {
        const row = document.createElement('div'); row.className = 'history-row';
        const button = document.createElement('button'); button.textContent = new Date(job.created*1000).toLocaleString('ru-RU');
        const sub = document.createElement('small'); sub.textContent = `${job.duration} с · ${job.effects ? 'Со стикерами' : 'Без стикеров'}`; button.append(sub);
        button.onclick = ()=>{selectedJob=job.id; showJob(job);};
        const badge = document.createElement('span'); badge.className='badge'; badge.textContent=({done:'Готово',error:'Ошибка',running:'В работе',queued:'Запуск'})[job.status];
        row.append(button,badge);
        if (job.status==='done') {const a=document.createElement('a'); a.href='/download/'+job.id; a.textContent='Скачать ↓'; row.append(a);}
        $('history-list').append(row);
      }
    }
  } catch (error) { $('message').textContent = 'Нет связи со студией. Проверь, что окно запуска открыто.'; }
}
$('form').addEventListener('submit', async e => {
  e.preventDefault(); if (!selectedFile || sending || state?.busy) return;
  sending=true; $('create').disabled=true; $('message').textContent='Загружаем озвучку…';
  try {
    const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'audio/mpeg','X-Studio-Request':'1','X-Effects':$('effects').checked?'1':'0'},body:selectedFile});
    const result=await response.json(); if (!response.ok) throw new Error(result.error || 'Ошибка запуска');
    selectedJob=result.id; $('message').textContent=''; $('log-details').open=false;
  } catch (error) { $('message').textContent=error.message; }
  finally {sending=false; await refresh();}
});
async function poll(){ await refresh(); setTimeout(poll,2000); }
poll();
