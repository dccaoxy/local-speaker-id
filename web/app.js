/* 一个页面，两个视图：列表（/）与确认（/r/<rid>）。
 * 所有阈值文案都从 /health 读，前端不写死任何数字——写死一个，
 * 改配置那天界面就会开始自打脸（「要 0.44 才敢认」而实际 0.40 就落名字）。 */
'use strict';

const app = document.getElementById('app');
const RID = location.pathname.startsWith('/r/') ? decodeURIComponent(location.pathname.slice(3)) : null;
let H = null;                 // /health 的门槛，供文案使用
let toastTimer = null;

const $ = (sel, el = document) => el.querySelector(sel);
function el(tag, attrs = {}, ...kids) {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) n.setAttribute(k, v);
  }
  for (const kid of kids.flat()) {
    if (kid === null || kid === undefined) continue;
    n.appendChild(typeof kid === 'string' ? document.createTextNode(kid) : kid);
  }
  return n;
}
const fmt = (s) => {
  s = Math.round(s || 0);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
  return h ? `${h} 小时 ${m} 分` : `${m} 分 ${s % 60} 秒`;
};

async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) {
    let msg = r.status;
    try { msg = (await r.json()).detail || msg; } catch (e) {}
    throw new Error(msg);
  }
  return r.json();
}

/* ── 文案：只说这一支真正意味着什么，一个数字都不给（few_windows 的秒数除外） ── */
function why(c) {
  const t = c.top1;
  switch (c.reason) {
    case 'mixed': return '这段像几个人在抢话，分段本身就不准';
    case 'mixed_weak': return '断断续续，可能还混了别人';
    case 'noisy': return '声音太小或太远，这段没敢认';
    case 'low_margin': return (t && c.second) ? `我也分不清是${t}还是${c.second}` : '像两个人，分不出';
    case 'few_windows': return `听起来像${t || '某个人'}，但这段总共 ${c.few_seconds} 秒，我不敢定`;
    case 'weak_proto': return `听着像${t || '某个人'}，但我对他的声音样本还不够`;
    case 'gray_zone': return `听着像${t || '某个人'}，但把握不够`;
    case 'low_rho': return '这段前后不一致，我拿不准是谁';
    case 'few_match': return (c.candidates || []).length ? '库里还没有这个人' : '这个声音我不认识';
    case 'too_short': return '这段太短，没认人';
    default: return '这段我拿不准';
  }
}

/* ── 列表 ─────────────────────────────────────────────────────────── */
async function renderList() {
  const data = await api('/api/recordings');
  const items = data.items || [];
  const s = data.stats || {};
  const head = el('header', {},
    el('div', {},
      el('h1', {}, '录音 · 说话人确认'),
      el('div', { class: 'sub' },
        s.awaiting ? `还有 ${s.awaiting} 段没定` : '都定完了'),
    ),
    el('div', { class: 'spacer' }),
    el('button', { class: 'ghost', onclick: async () => { await api('/api/scan', { method: 'POST' }); renderList(); } }, '扫一下')
  );

  const rows = items.map((r) => {
    const status = {
      queued: ['gray', '排队中'], waiting_stable: ['gray', '等文件传完'],
      processing: ['gray', `处理中 ${r.progress || 0}%`],
      awaiting_review: r.pending ? ['warn', `${r.pending} 个待确认`] : ['warn', '待定稿'],
      finalized: ['gray', '已定稿'], failed: ['warn', '处理失败'],
    }[r.status] || ['gray', r.status || '—'];
    return el('div', { class: 'row' },
      el('div', { class: 't' },
        el('b', { title: r.title }, r.title || r.rid),
        el('span', {}, `${fmt(r.duration)} · ${r.n_clusters || 0} 个说话人`),
      ),
      el('span', { class: `pill ${status[0]}` }, status[1]),
      r.status === 'failed'
        ? el('button', { onclick: () => retry(r.rid) }, '重试')
        : el('button', { onclick: () => location.href = `/r/${r.rid}` },
            r.status === 'awaiting_review' ? '去确认' : '查看'),
    );
  });
  app.replaceChildren(head, ...(rows.length ? rows : [el('div', { class: 'empty' }, '还没有录音。把音频放进录音目录，一分钟内会自动开始。')]));
}

async function retry(rid) {
  await api('/api/retry', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ rid }) });
  renderList();
}

/* ── 确认页 ───────────────────────────────────────────────────────── */
async function renderConfirm() {
  let data;
  try { data = await api(`/api/recordings/${RID}`); }
  catch (e) { app.replaceChildren(el('div', { class: 'empty' }, '找不到这段录音')); return; }
  H = H || data.thresholds;

  const pending = (data.clusters || []).filter((c) => !c.decided);
  const idx = Math.max(0, (data.clusters || []).length - pending.length);

  const head = el('header', {},
    el('a', { href: '/' }, '←'),
    el('div', {},
      el('h1', {}, data.title || RID),
      el('div', { class: 'sub' }, `${fmt(data.duration)} · ${(data.clusters || []).length} 个说话人 · ` +
        (pending.length ? `还剩 ${pending.length} 个没确认` : '都确认了')),
    ),
    el('div', { class: 'spacer' }),
    el('div', { class: 'dots' }, ...(data.clusters || []).map((c, i) =>
      el('button', {
        class: `dot ${c.decided ? 'done' : (i === idx ? 'on' : '')}`,
        onclick: () => focusCluster(c.cluster_id),
      }))),
  );

  const banners = (data.warnings || []).map((w) => el('div', { class: 'banner' }, w));
  // 「人数不对」出口：模型数不准说话人（实测双人电话录音会被切成 40 多个碎片簇），
  // 与其假装能自动，不如给一个两秒就能改的入口。
  const clusterPicker = el('div', { class: 'tiny', style: 'margin:-6px 0 14px' },
    '分离成了 ', el('b', {}, String((data.clusters || []).length)), ' 个人。',
    el('select', { id: 'kpick', style: 'margin:0 6px' },
      ...[1, 2, 3, 4, 5, 6].map((n) => el('option', { value: n, selected: n === data.clusters.length ? 'selected' : null }, String(n)))),
    el('button', { onclick: async () => {
      const k = Number(document.getElementById('kpick').value);
      await api('/api/retry', { method: 'POST', headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ rid: RID, clusters: k }) });
      location.reload();
    } }, '重新分离'));
  banners.push(clusterPicker);

  if (!pending.length) {
    app.replaceChildren(head, ...banners,
      el('div', { class: 'card' },
        el('h2', {}, '都确认完了'),
        el('div', { class: 'meta' }, '稿子已经重新生成，改人名不会重跑分离。'),
        el('div', { class: 'actions' },
          el('button', { class: 'primary', onclick: () => location.href = '/' }, '回列表'),
          el('button', { onclick: () => location.reload() }, '再看一遍'))));
    return;
  }

  const c = pending[0];
  app.replaceChildren(head, ...banners, cardFor(c, data));
  bindSpace(c);
}

let currentAudio = null;
function bindSpace(c) {
  document.onkeydown = (e) => {
    if (e.code === 'Space' && document.activeElement.tagName !== 'INPUT') {
      e.preventDefault();
      const b = $('.play');
      if (b) b.click();
    }
  };
}

function cardFor(c, data) {
  const audio = new Audio();
  let loaded = false;
  const bar = el('div', { class: 'bar' });
  const label = el('span', {}, c.clip_url ? '▶ 听 15 秒' : '这段没有试听音频');
  const play = el('button', { class: 'play', disabled: c.clip_url ? null : 'disabled' },
    label, bar);
  play.onclick = () => {
    if (!c.clip_url) return;
    if (!loaded) { audio.src = c.clip_url; loaded = true; }
    if (audio.paused) { audio.play(); label.textContent = '暂停'; }
    else { audio.pause(); label.textContent = '▶ 继续听'; }
  };
  audio.ontimeupdate = () => {
    if (audio.duration) bar.style.width = `${(audio.currentTime / audio.duration) * 100}%`;
  };
  audio.onended = () => { label.textContent = '▶ 再听一遍'; bar.style.width = '0'; };

  const samples = el('ul', { class: 'samples' },
    ...(c.samples || []).slice(0, 3).map((t) => el('li', {}, t)));

  const node = el('div', { class: 'card' },
    el('h2', {}, `说话人 ${String.fromCharCode(65 + (data.clusters || []).findIndex((x) => x.cluster_id === c.cluster_id))}`),
    el('div', { class: 'meta' }, `说了 ${fmt(c.duration_s)}（占 ${Math.round((c.share || 0) * 100)}%）`),
    play,
    el('div', { class: 'tiny' }, c.clip_url ? '跨 3 段拼接，按空格键也能播' : '这段太短，听不出是谁'),
    (c.samples || []).length ? samples : null,
    el('div', { class: 'why', html: why(c).replace(/&/g, '&amp;') }),
  );

  node.appendChild(actionsFor(c, data));
  return node;
}

function actionsFor(c, data) {
  const box = el('div', { class: 'actions' });
  const cands = el('div', { class: 'cands' });
  const decide = (verdict, name, persist) => submit(c, verdict, name, persist);

  const showCandidates = (extra) => {
    cands.replaceChildren(el('div', { class: 'hint' }, '最可能是这几个（我不确定）'));
    (extra || []).forEach((n) => cands.appendChild(
      el('button', { onclick: () => decide('name', n, false) }, n)));
    (c.candidates || []).filter((n) => n !== c.top1).forEach((n) => cands.appendChild(
      el('button', { onclick: () => decide('name', n, false) }, n)));
    cands.appendChild(el('button', { onclick: () => openNew(c) }, '新建一个名字…'));
    cands.appendChild(el('button', { onclick: () => decide('skip', null, false) }, '都不认识，先留空'));
    box.style.display = 'none';
  };

  switch (c.primary_action) {
    case 'yes_named':
      box.append(
        el('button', { class: 'primary', onclick: () => decide('name', c.top1, false) }, `是 ${c.top1}`),
        el('button', { onclick: () => showCandidates() }, '不是，选人'));
      break;
    case 'confirm':
      box.append(
        el('button', { class: 'primary', onclick: () => decide('name', c.top1, false) }, `就按 ${c.top1} 记（我确认）`),
        el('button', { onclick: () => showCandidates() }, '不是，选人'));
      break;
    case 'choose_one':
      box.append(
        el('button', { class: 'primary', onclick: () => decide('name', c.top1, false) }, `是 ${c.top1}`),
        el('button', { class: 'primary', onclick: () => decide('name', c.second, false) }, `是 ${c.second}`),
        el('button', { onclick: () => showCandidates() }, '都不是'));
      break;
    case 'new_person':
      box.append(
        el('button', { class: 'primary', onclick: () => openNew(c) }, '新建一个名字'),
        el('button', { onclick: () => openNew(c, true) }, '一次性的人，不入库'),
        el('button', { class: 'ghost', onclick: () => decide('skip', null, false) }, '先留空'));
      break;
    case 'skip':
    default:
      box.append(
        el('button', { class: 'primary', onclick: () => decide('skip', null, false) }, '先留空'),
        el('button', { onclick: () => showCandidates() }, '其实是其中某个人'));
      break;
  }
  return el('div', {}, box, cands);
}

function openNew(c, oneOff) {
  const input = el('input', { type: 'text', placeholder: '他的名字', autofocus: 'autofocus' });
  const rKeep = el('input', { type: 'radio', name: 'p', value: 'keep' });
  const rOne = el('input', { type: 'radio', name: 'p', value: 'one' });
  if (oneOff) rOne.checked = true;
  const modal = el('div', { class: 'modal', onclick: (e) => { if (e.target === modal) modal.remove(); } },
    el('div', { class: 'box' },
      el('h3', {}, oneOff ? '一次性的人' : '新建一个名字'),
      input,
      el('div', { class: 'choice' },
        el('label', {}, rKeep, '记住他，以后自动认'),
        el('label', {}, rOne, '只用这一次，下次还会问你')),
      el('div', { class: 'tiny' }, '「记住」会把这段的声音存进本地声纹库；「只用这一次」不存。'),
      el('div', { class: 'actions' },
        el('button', { class: 'primary', onclick: () => {
          const name = input.value.trim();
          if (!name) { input.focus(); return; }
          modal.remove();
          submit(c, 'new', name, rKeep.checked);
        } }, '确定'),
        el('button', { class: 'ghost', onclick: () => modal.remove() }, '取消'))));
  document.body.appendChild(modal);
  setTimeout(() => input.focus(), 30);
  input.onkeydown = (e) => { if (e.key === 'Enter') $('.actions .primary', modal).click(); };
}

async function submit(c, verdict, name, persist) {
  try {
    const r = await api('/api/decisions', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ rid: RID, cluster_id: c.cluster_id, verdict, name, persist: !!persist }),
    });
    let msg = name ? `已标记「${name}」` : '已留空';
    if (r.enroll && r.enroll.source_id) msg += '，已入库';
    else if (r.enroll && r.enroll.error) msg += `；未入库：${r.enroll.error}`;
    showToast(msg, () => api('/api/undo', {
      method: 'POST', headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ rid: RID, cluster_id: c.cluster_id }),
    }).then(() => { renderConfirm(); }));
    renderConfirm();
  } catch (e) {
    showToast(`没记上：${e.message}`, null, true);
  }
}

function showToast(text, onUndo, isError) {
  document.querySelectorAll('.toast').forEach((t) => t.remove());
  const t = el('div', { class: 'toast' }, text,
    onUndo && !isError ? el('button', { onclick: () => { t.remove(); onUndo(); } }, '撤销') : null);
  document.body.appendChild(t);
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.remove(), isError ? 6000 : 8000);
}

function focusCluster(cid) {
  // 已确认的簇也能回看：跳到第一张未确认的重画一次，简单但不会漏
  renderConfirm();
}

/* ── 启动 ─────────────────────────────────────────────────────────── */
async function boot() {
  try {
    if (RID) await renderConfirm(); else await renderList();
  } catch (e) {
    app.replaceChildren(el('div', { class: 'empty' }, `连不上本地服务：${e.message}`));
  }
  setInterval(() => {
    if (document.querySelector('.modal')) return;
    if (RID) { if (!$('.card')) renderConfirm(); }
    else renderList();
  }, 3000);
}
boot();
