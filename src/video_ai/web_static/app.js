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
  $('step1').textContent = job.style === 'story' ? '1. История и материалы' : '1. Кадры и основа';
  $('step2').textContent = job.style === 'story' ? '2. Сборка сцен' : '2. Оформление';
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
    const names = {GEMINI_API_KEY:'Gemini',PEXELS_API_KEY:'Pexels',PIXABAY_API_KEY:'Pixabay',GIPHY_API_KEY:'Giphy (необязательно)',ffmpeg:'FFmpeg',ffprobe:'Проверка аудио'};
    $('connections').replaceChildren();
    for (const [key, ok] of Object.entries({...state.keys,...state.tools})) {
      const row = document.createElement('div'); row.className = 'key';
      const name = document.createElement('span'); name.textContent = names[key];
      const status = document.createElement('span'); status.className = ok ? 'good' : 'missing'; status.textContent = ok ? (key.endsWith('KEY') ? 'Ключ найден' : 'Готово') : 'Не найден';
      row.append(name,status); $('connections').append(row);
    }
    $('connection-count').textContent = `${['GEMINI_API_KEY','PEXELS_API_KEY','PIXABAY_API_KEY'].filter(k=>state.keys[k]).length}/3 ключей`;
    if (!selectedJob && state.jobs.length) selectedJob = state.jobs[0].id;
    showJob(state.jobs.find(j=>j.id === selectedJob));
    $('history-count').textContent = state.jobs.length ? String(state.jobs.length) : '';
    if (state.jobs.length) {
      $('history-list').replaceChildren();
      for (const job of state.jobs) {
        const row = document.createElement('div'); row.className = 'history-row';
        const button = document.createElement('button'); button.textContent = new Date(job.created*1000).toLocaleString('ru-RU');
        const sub = document.createElement('small'); sub.textContent = `${job.duration} с · ${({story:'Истории и мемы',dynamic:'Динамичный',viral:'Viral / Darwin',classic:'Классический'})[job.style || 'classic']} · ${job.effects ? 'С реакциями' : 'Без реакций'}`; button.append(sub);
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
    const response=await fetch('/api/jobs',{method:'POST',headers:{'Content-Type':'audio/mpeg','X-Studio-Request':'1','X-Effects':$('effects').checked?'1':'0','X-Style':$('style').value},body:selectedFile});
    const result=await response.json(); if (!response.ok) throw new Error(result.error || 'Ошибка запуска');
    selectedJob=result.id; $('message').textContent=''; $('log-details').open=false;
  } catch (error) { $('message').textContent=error.message; }
  finally {sending=false; await refresh();}
});
async function poll(){ await refresh(); setTimeout(poll,2000); }
poll();

$('style').addEventListener('change', () => {
  const story = $('style').value === 'story';
  const dynamic = $('style').value === 'dynamic';
  const viral = $('style').value === 'viral';
  $('style-note').textContent = story ? 'Фото, видео и сравнения по смыслу. Использует папки memes и elements; первый раз разбор паков займёт больше времени.' : dynamic ? 'Крупные реакции по центру и пружинящие субтитры. На основе твоего удачного варианта Shorts.' : viral ? 'Стоки и мемы сменяют друг друга отдельными кадрами. Без зумов; смайлики — поверх видео.' : 'Знакомый монтаж, как в предыдущих роликах.';
  $('effects-note').textContent = story ? 'Реакции и элементы из твоих паков по смыслу истории' : dynamic ? 'По смыслу примерно каждые 3–4 секунды. Сначала твои паки; Giphy — если подключён. Без текстовых плашек.' : viral ? 'Мемы — отдельными кадрами по смыслу фразы. Смайлики — короткими реакциями поверх видео.' : 'Количество реакций — по смыслу, без текстовых плашек';
});


let chatConversationId = null;
let chatSending = false;

function chatHeaders() {
  return {'Content-Type':'application/json','X-Studio-Request':'1'};
}

function renderChatMessages(messages) {
  const box = $('chat-messages');
  box.replaceChildren();
  if (!messages.length) {
    const p = document.createElement('p');
    p.className = 'muted';
    p.textContent = 'Начни разговор. Монтажёр будет помнить историю локально.';
    box.append(p);
    return;
  }
  for (const item of messages) {
    const row = document.createElement('div');
    row.className = 'chat-bubble ' + (item.role === 'assistant' ? 'assistant' : 'user');
    const who = document.createElement('strong');
    who.textContent = item.role === 'assistant' ? 'AI монтажёр' : 'Ты';
    const text = document.createElement('div');
    text.textContent = item.content;
    row.append(who, text);
    box.append(row);
  }
  box.scrollTop = box.scrollHeight;
}

function renderMemories(memories) {
  const box = $('chat-memory');
  box.replaceChildren();
  if (!memories.length) {
    const p = document.createElement('p');
    p.className = 'muted';
    p.textContent = 'Пока пусто.';
    box.append(p);
    return;
  }
  for (const item of memories) {
    const p = document.createElement('p');
    p.textContent = '• ' + item;
    box.append(p);
  }
}

async function loadChatMessages() {
  if (!chatConversationId) return;
  const response = await fetch('/api/chat/messages/' + chatConversationId);
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Не удалось загрузить чат');
  renderChatMessages(data.messages || []);
}

async function refreshChats(preferId = null) {
  try {
    const response = await fetch('/api/chat/conversations');
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Не удалось загрузить чаты');
    const list = data.conversations || [];
    renderMemories(data.memories || []);
    const select = $('chat-conversation');
    select.replaceChildren();
    for (const item of list) {
      const option = document.createElement('option');
      option.value = item.id;
      option.textContent = item.title + (item.job_id ? ' · ролик' : '');
      select.append(option);
    }
    chatConversationId = preferId || chatConversationId || (list[0] && list[0].id) || null;
    if (chatConversationId && list.some(x => x.id === chatConversationId)) {
      select.value = chatConversationId;
      await loadChatMessages();
    } else {
      renderChatMessages([]);
    }
  } catch (error) {
    $('chat-status').textContent = error.message;
  }
}

async function createChat() {
  $('chat-status').textContent = '';
  const response = await fetch('/api/chat/conversations', {
    method:'POST',
    headers:chatHeaders(),
    body:JSON.stringify({job_id:selectedJob || null})
  });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || 'Не удалось создать чат');
  chatConversationId = data.id;
  await refreshChats(chatConversationId);
}

$('chat-new').addEventListener('click', async () => {
  try { await createChat(); } catch (error) { $('chat-status').textContent = error.message; }
});

$('chat-conversation').addEventListener('change', async e => {
  chatConversationId = e.target.value || null;
  try { await loadChatMessages(); } catch (error) { $('chat-status').textContent = error.message; }
});

$('chat-form').addEventListener('submit', async e => {
  e.preventDefault();
  if (chatSending) return;
  const message = $('chat-input').value.trim();
  if (!message) return;
  try {
    chatSending = true;
    $('chat-send').disabled = true;
    $('chat-status').textContent = 'Монтажёр думает…';
    if (!chatConversationId) await createChat();
    const response = await fetch('/api/chat/send', {
      method:'POST',
      headers:chatHeaders(),
      body:JSON.stringify({
        conversation_id:chatConversationId,
        message,
        job_id:selectedJob || null
      })
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Ошибка Gemini');
    $('chat-input').value = '';
    $('chat-model').textContent = data.model || 'Gemini';
    $('chat-status').textContent = '';
    renderMemories(data.memories || []);
    await refreshChats(chatConversationId);
  } catch (error) {
    $('chat-status').textContent = error.message;
  } finally {
    chatSending = false;
    $('chat-send').disabled = false;
  }
});

refreshChats();
