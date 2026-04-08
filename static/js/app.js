/**
 * app.js  —  CricIQ dynamic API-driven dashboard
 *
 * ALL prediction data comes from Flask API calls — nothing is hardcoded.
 *
 * Flow:
 *   1. loadSchedule()         → GET /api/schedule
 *   2. renderMatchList()      → left panel — match cards + run total chips
 *   3. selectMatch(idx)       → picks match, resets state, fires renderMatchDetail
 *   4. renderMatchDetail()    → builds skeleton DOM (header, proj cards, player rows)
 *   5. loadAllPredictions()   → fires one fetch per player concurrently
 *   6. renderPrediction()     → fills each player row as its fetch resolves
 *   7. flushProjectedTotals() → writes team run totals once all fetches settle
 */

'use strict';

/**
 * safeJson(res, label)
 * --------------------
 * Checks res.ok before calling .json() so a Flask HTML error page never
 * triggers "Unexpected token '<'".  When the server returns a non-2xx
 * status the function throws a readable Error instead of a JSON parse crash.
 */
async function safeJson(res, label = 'API') {
  if (!res.ok) {
    // Try to extract a message from the body, but never call .json() blindly
    const text = await res.text().catch(() => '');
    // If the body looks like HTML (Flask error page), give a clean message
    const msg = text.startsWith('<')
      ? `${label} returned HTTP ${res.status}. Is 06_app.py up to date?`
      : (text.slice(0, 200) || `HTTP ${res.status}`);
    throw new Error(msg);
  }
  return res.json();
}

// ── STATE ─────────────────────────────────────────────────────────────────────
const state = {
  format:       'ipl',
  schedule:     {},       // { ipl: [], t20i: [] }  from /api/schedule
  activeMatch:  null,
  pendingCalls: new Set(),
  // Per-match prediction cache: { [matchKey]: { [playerName]: responseData } }
  // Used to backfill the match list with run totals after a match is loaded.
  cache:        {},
};

const TODAY = new Date().toISOString().slice(0, 10);

// ── SYNC STATE ────────────────────────────────────────────────────────────────
const syncState = { syncing: false };

// ── BOOT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSchedule();
  loadSyncStatus();
});

// ── SCHEDULE ──────────────────────────────────────────────────────────────────

async function loadSchedule() {
  showListLoading();
  try {
    const res  = await fetch('/api/schedule');
    const data = await safeJson(res, '/api/schedule');
    state.schedule = data;
    updateTabCounts();
    renderMatchList();
  } catch (err) {
    showListError('Could not reach /api/schedule — is Flask running?');
    console.error(err);
  }
}

function updateTabCounts() {
  ['ipl', 't20i'].forEach(fmt => {
    const tab = document.getElementById(`tab-${fmt}`);
    if (tab) tab.querySelector('.tab-count').textContent =
      (state.schedule[fmt] || []).length;
  });
}

// ── FORMAT SWITCHING ──────────────────────────────────────────────────────────

function switchFormat(fmt) {
  state.format      = fmt;
  state.activeMatch = null;
  document.querySelectorAll('.format-tab').forEach(t =>
    t.classList.toggle('active', t.dataset.format === fmt));
  document.getElementById('match-search').value = '';
  renderMatchList();
  showDetailPlaceholder();
}

// ── MATCH LIST ────────────────────────────────────────────────────────────────

function renderMatchList(query = '') {
  const matches = (state.schedule[state.format] || []).filter(m => {
    if (!query) return true;
    const q = query.toLowerCase();
    return m.home.toLowerCase().includes(q) ||
           m.away.toLowerCase().includes(q) ||
           m.venue.toLowerCase().includes(q);
  });

  const panel = document.getElementById('match-list');
  if (!matches.length) {
    panel.innerHTML = `<div class="state-placeholder" style="min-height:160px">
      <div class="state-sub">NO MATCHES</div></div>`;
    return;
  }

  panel.innerHTML = matches.map((m, idx) => {
    // Status comes from the sync module: 'completed' | 'live' | 'upcoming'
    // Fall back to date-based heuristic if status not set
    const status   = m.status || (m.date < TODAY ? 'completed' : m.date === TODAY ? 'live' : 'upcoming');
    const isLive   = status === 'live';
    const isDone   = status === 'completed';
    const badge    = isLive ? 'today' : isDone ? 'past' : 'upcoming';
    const label    = isLive ? 'LIVE'  : isDone ? 'FT'   : 'UPCOMING';
    const active   = state.activeMatch?.match === m.match &&
                     state.activeMatch?.format === m.format ? 'active' : '';
    const dateStr  = new Date(m.date).toLocaleDateString('en-IN',
                     { day: 'numeric', month: 'short' });

    // Result row for completed matches (from sync)
    let bottomHTML = '';
    if (isDone && m.result) {
      const r = m.result;
      const winnerShort = r.winner ? `<span class="mi-result-winner">${r.winner}</span> won` : 'Result';
      const margin = r.margin ? ` by ${r.margin}` : '';
      bottomHTML = `
        <div class="mi-result">
          <div class="mi-result-summary">${winnerShort}${margin}</div>
          ${r.home_score || r.away_score ? `
          <div class="mi-scores">
            ${r.home_score ? `<span class="mi-score">${m.home} ${r.home_score}</span>` : ''}
            ${r.away_score ? `<span class="mi-score">${m.away} ${r.away_score}</span>` : ''}
          </div>` : ''}
        </div>`;
    } else if (!isDone) {
      // Show cached prediction totals for upcoming/live matches
      const mKey   = matchKey(m);
      const totals = buildCachedTotals(mKey, m);
      if (totals) bottomHTML = `<div class="mi-totals">${totals}</div>`;
    }

    return `<div class="match-item ${active}" onclick="selectMatch(${idx})" data-idx="${idx}">
      <div class="mi-header">
        <span class="mi-num">M${m.match} · ${m.format.toUpperCase()}</span>
        <span class="mi-badge ${badge}">${label}</span>
      </div>
      <div class="mi-teams">${m.home}<span class="mi-vs">vs</span>${m.away}</div>
      <div class="mi-venue">${m.venue}</div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--ink3);margin-top:2px">${dateStr}</div>
      ${bottomHTML}
    </div>`;
  }).join('');
}

function buildCachedTotals(mKey, m) {
  const cache = state.cache[mKey];
  if (!cache) return null;
  const teams = Object.keys(m.teams);
  return teams.map(team => {
    const sq    = m.teams[team];
    let total   = 0;
    let n       = 0;
    for (const p of sq.bat.filter(x => x.in_profile)) {
      const d = cache[p.name];
      if (d && d.runs_mid != null) { total += d.runs_mid; n++; }
    }
    if (!n) return '';
    return `<span class="mi-total-chip loaded">${team} ~${Math.round(total)}</span>`;
  }).filter(Boolean).join('');
}

function filterMatchList(query) {
  renderMatchList(query);
}

// ── SELECT MATCH ──────────────────────────────────────────────────────────────

function selectMatch(idx) {
  const matches = state.schedule[state.format] || [];
  // idx here is the index in the currently filtered list — we need the real match
  // The filtered list is just a visual filter; onclick passes the original idx.
  const match = matches[idx];
  if (!match) return;

  state.pendingCalls.clear();
  state.activeMatch = match;

  document.querySelectorAll('.match-item').forEach((el, i) =>
    el.classList.toggle('active', i === idx));

  renderMatchDetail(match);
}

// ── MATCH DETAIL ──────────────────────────────────────────────────────────────

function renderMatchDetail(match) {
  const panel = document.getElementById('match-detail');
  ctxChipsSet = false;

  const status    = match.status || (match.date < TODAY ? 'completed' : 'upcoming');
  const isCompleted = status === 'completed';

  if (isCompleted) {
    // ── COMPLETED: show result + top performers + predictions as context ──
    panel.innerHTML = `
      ${matchHeaderHTML(match)}
      ${matchResultBannerHTML(match)}
      <div class="filter-row">
        <span class="filter-label">Pre-match predictions</span>
        <button class="filter-pill active" onclick="setRoleFilter('all', this)">All</button>
        <button class="filter-pill" onclick="setRoleFilter('bat', this)">Batters</button>
        <button class="filter-pill bowl-active" onclick="setRoleFilter('bowl', this)"
                style="--active-bg:var(--red-d);--active-border:var(--red);--active-color:var(--red)">
          Bowlers</button>
      </div>
      <div id="teams-container">
        ${teamSectionHTML(match, match.home, match.teams[match.home] || {bat:[],bowl:[]})}
        ${teamSectionHTML(match, match.away, match.teams[match.away] || {bat:[],bowl:[]})}
      </div>`;
  } else {
    // ── UPCOMING / LIVE: show projected totals + per-player predictions ──
    panel.innerHTML = `
      ${matchHeaderHTML(match)}
      <div class="filter-row">
        <span class="filter-label">Show</span>
        <button class="filter-pill active" onclick="setRoleFilter('all', this)">All</button>
        <button class="filter-pill" onclick="setRoleFilter('bat', this)">Batters</button>
        <button class="filter-pill bowl-active" onclick="setRoleFilter('bowl', this)"
                style="--active-bg:var(--red-d);--active-border:var(--red);--active-color:var(--red)">
          Bowlers</button>
      </div>
      <div class="projected-row" id="proj-row">
        ${projCardHTML(match.home)}
        ${projCardHTML(match.away)}
      </div>
      <div id="teams-container">
        ${teamSectionHTML(match, match.home, match.teams[match.home] || {bat:[],bowl:[]})}
        ${teamSectionHTML(match, match.away, match.teams[match.away] || {bat:[],bowl:[]})}
      </div>`;
  }

  requestAnimationFrame(() => {
    document.querySelectorAll('.player-row').forEach((el, i) => {
      setTimeout(() => el.classList.add('visible'), i * 30);
    });
  });

  loadAllPredictions(match);
}

// ── ROLE FILTER ───────────────────────────────────────────────────────────────

function setRoleFilter(role, btn) {
  btn.closest('.filter-row').querySelectorAll('.filter-pill')
     .forEach(p => p.classList.remove('active'));
  btn.classList.add('active');

  document.querySelectorAll('.player-row[data-role]').forEach(row => {
    const show = role === 'all' || row.dataset.role === role;
    row.style.display = show ? '' : 'none';
  });

  // Hide role-dividers that have no visible rows after them
  document.querySelectorAll('.role-divider').forEach(div => {
    let sibling = div.nextElementSibling;
    let hasVisible = false;
    while (sibling && !sibling.classList.contains('role-divider')) {
      if (sibling.style.display !== 'none') { hasVisible = true; break; }
      sibling = sibling.nextElementSibling;
    }
    div.style.display = hasVisible ? '' : 'none';
  });
}

// ── MATCH HEADER HTML ─────────────────────────────────────────────────────────

function matchHeaderHTML(match) {
  return `
  <div class="match-header-card">
    <div class="mh-top">
      <span class="mh-matchno">Match ${match.match} · ${match.format.toUpperCase()}</span>
      <span class="mh-date">${formatDate(match.date)}</span>
    </div>
    <div class="mh-teams">
      <span class="mh-team">${match.home}</span>
      <span class="mh-vs">vs</span>
      <span class="mh-team">${match.away}</span>
    </div>
    <div class="mh-venue">📍 ${match.venue}</div>
    <div class="context-chips" id="ctx-chips">
      <span class="ctx-chip loading pitch-balanced">⛏ loading…</span>
      <span class="ctx-chip loading dew-low">💧 loading…</span>
      <span class="ctx-chip loading chase-no">🏃 loading…</span>
    </div>
  </div>`;
}

// ── RESULT BANNER (completed matches) ─────────────────────────────────────────

function matchResultBannerHTML(match) {
  const r = match.result;
  if (!r) {
    return `<div class="result-banner no-result">
      <div class="rb-label">RESULT</div>
      <div class="rb-summary">Result data not yet synced — run <code>python 07_sync_schedule.py</code> or POST /api/sync</div>
    </div>`;
  }

  const winnerLine = r.winner
    ? `<span class="rb-winner">${r.winner}</span> won${r.margin ? ` by ${r.margin}` : ''}`
    : (r.summary || 'Match completed');

  const scoresHTML = (r.home_score || r.away_score) ? `
    <div class="rb-scores">
      ${r.home_score ? `<div class="rb-score-item"><span class="rb-team">${match.home}</span><span class="rb-score">${r.home_score}</span></div>` : ''}
      ${r.away_score ? `<div class="rb-score-item"><span class="rb-team">${match.away}</span><span class="rb-score">${r.away_score}</span></div>` : ''}
    </div>` : '';

  const topHTML = r.top_performers && r.top_performers.length ? `
    <div class="rb-top-performers">
      <div class="rb-tp-label">Top performers</div>
      <div class="rb-tp-list">
        ${r.top_performers.slice(0, 6).map(p => `
          <div class="rb-tp-row">
            <span class="rb-tp-name">${p.name}</span>
            ${p.role === 'BAT'
              ? `<span class="rb-tp-stat bat">${p.runs} (${p.balls})</span>`
              : `<span class="rb-tp-stat bowl">${p.wickets}/${p.runs}</span>`}
          </div>`).join('')}
      </div>
    </div>` : '';

  return `
  <div class="result-banner">
    <div class="rb-label">FULL TIME</div>
    <div class="rb-summary">${winnerLine}</div>
    ${scoresHTML}
    ${topHTML}
  </div>`;
}

// ── PROJECTED TOTAL CARD ──────────────────────────────────────────────────────

function projCardHTML(team) {
  return `
  <div class="proj-card" id="proj-${safeId(team)}">
    <div class="proj-label">Projected batting runs</div>
    <div class="proj-team-name">${team}</div>
    <div class="proj-loading" id="proj-loading-${safeId(team)}">
      <div class="spinner" style="width:14px;height:14px;border-width:1.5px"></div>
      fetching…
    </div>
    <div class="proj-bar-wrap" style="display:none" id="proj-bar-${safeId(team)}">
      <div class="proj-bar-track"><div class="proj-bar-fill" style="width:0%"></div></div>
    </div>
  </div>`;
}

// ── TEAM SECTION HTML ─────────────────────────────────────────────────────────

function teamSectionHTML(match, team, squad) {
  const bats    = squad.bat.filter(p => p.in_profile);
  const bowls   = squad.bowl.filter(p => p.in_profile);
  const skipped = [...squad.bat, ...squad.bowl].filter(p => !p.in_profile);

  const rows = [
    bats.length
      ? `<div class="role-divider">Batters</div>` +
        bats.map(p => playerRowHTML(p.name, 'bat', match)).join('')
      : '',
    bowls.length
      ? `<div class="role-divider">Bowlers</div>` +
        bowls.map(p => playerRowHTML(p.name, 'bowl', match)).join('')
      : '',
    skipped.length
      ? `<div class="role-divider" style="color:var(--ink3)">No profile data</div>` +
        skipped.map(p => `
          <div class="player-row visible" data-role="skip" style="opacity:.35">
            <div class="player-row-left">
              <div class="p-role-dot" style="background:var(--border)"></div>
              <div class="p-info">
                <div class="p-name">${p.name}</div>
                <div class="p-sub">Not in training data</div>
              </div>
            </div>
          </div>`).join('')
      : '',
  ].join('');

  return `
  <div class="team-section">
    <div class="team-section-header">
      <span class="team-section-name">${team}</span>
      <span class="team-section-sub">${bats.length} bat · ${bowls.length} bowl</span>
    </div>
    ${rows}
  </div>`;
}

function playerRowHTML(name, role, match) {
  const pid = playerDomId(name, match);
  return `
  <div class="player-row" id="row-${pid}" data-role="${role}" data-player="${name}">
    <div class="player-row-left">
      <div class="p-role-dot ${role}"></div>
      <div class="p-info">
        <div class="p-name">${name}</div>
        <div class="p-sub" id="sub-${pid}">
          <span style="opacity:.5">loading…</span>
        </div>
      </div>
    </div>
    <div class="pred-widget" id="pred-${pid}">
      <div class="pred-loading">
        <div class="spinner" style="width:14px;height:14px;border-width:1.5px"></div>
        predicting…
      </div>
    </div>
  </div>`;
}

// ── PREDICTION CALLS ──────────────────────────────────────────────────────────

/**
 * Fire one prediction fetch per profiled player, all concurrently.
 * callId guards against stale results when the user clicks a new match
 * before the previous batch has settled.
 */
async function loadAllPredictions(match) {
  const callId = Symbol(`${match.format}-${match.match}`);
  state.pendingCalls.add(callId);

  initProjTotals(match);

  const { teams, venue, date, format: fmt } = match;
  const tasks = [];
  for (const [team, squad] of Object.entries(teams)) {
    for (const p of squad.bat.filter(x => x.in_profile))
      tasks.push({ name: p.name, role: 'bat',  team, venue, date, fmt });
    for (const p of squad.bowl.filter(x => x.in_profile))
      tasks.push({ name: p.name, role: 'bowl', team, venue, date, fmt });
  }

  const promises = tasks.map(task =>
    getPrediction(task)
      .then(data => {
        if (!state.pendingCalls.has(callId)) return;
        cacheResult(match, task.name, data);
        renderPrediction(task, data, match);
      })
      .catch(err => {
        if (!state.pendingCalls.has(callId)) return;
        renderPredictionError(task, match);
        console.warn(`Prediction failed for ${task.name}:`, err.message);
      })
  );

  Promise.allSettled(promises).then(() => {
    if (!state.pendingCalls.has(callId)) return;
    flushProjectedTotals(match);
    // Refresh match list item to show cached totals
    renderMatchList(document.getElementById('match-search').value);
    // Re-highlight active item (renderMatchList rebuilds DOM)
    document.querySelectorAll('.match-item').forEach((el, i) => {
      const idx = (state.schedule[state.format] || []).findIndex(
        m => m.match === match.match && m.format === match.format);
      el.classList.toggle('active', i === idx);
    });
  });
}

/**
 * Single prediction fetch — routes to the correct endpoint.
 */
async function getPrediction({ name, role, team, venue, date, fmt }) {
  const endpoint = {
    'ipl-bat':    '/predict',
    'ipl-bowl':   '/predict_bowl',
    't20i-bat':   '/predict_t20_bat',
    't20i-bowl':  '/predict_t20_bowl',
  }[`${fmt}-${role}`];

  const res = await fetch(endpoint, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ player: name, venue, team, date }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }
  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ── RESULT CACHE ──────────────────────────────────────────────────────────────

function cacheResult(match, playerName, data) {
  const key = matchKey(match);
  if (!state.cache[key]) state.cache[key] = {};
  state.cache[key][playerName] = data;
}

// ── DOM UPDATES ───────────────────────────────────────────────────────────────

function renderPrediction(task, data, match) {
  const pid   = playerDomId(task.name, match);
  const isBat = task.role === 'bat';

  const lo   = isBat ? data.runs_low   : data.wkts_low;
  const md   = isBat ? data.runs_mid   : data.wkts_mid;
  const hi   = isBat ? data.runs_high  : data.wkts_high;
  const unit = isBat ? 'runs' : 'wkts';
  const maxV = isBat ? 80 : 4;

  const loFr = Math.min(lo / maxV * 100, 100).toFixed(1);
  const hiFr = Math.min(hi / maxV * 100, 100).toFixed(1);
  const wFr  = Math.max(0, hiFr - loFr).toFixed(1);

  // Confidence derived client-side from interval width vs career average
  const conf = computeConfidence(lo, hi, md, data.career_avg, task.role);

  // Sub-label: career stats + sparkline from last-N (synthesised from career/recent avg)
  const subEl = document.getElementById(`sub-${pid}`);
  if (subEl) {
    const avgLabel = isBat
      ? `avg ${data.career_avg}`
      : `avg ${data.career_avg} wkt`;
    subEl.innerHTML =
      `${avgLabel} · ${data.innings} inn &nbsp;` +
      `<span class="conf-badge ${conf}">${conf}</span>`;
  }

  // Accumulate bat totals for projected score card
  if (isBat) accumulateBatTotal(task.team, lo, md, hi);

  // Update header context chips on first result
  updateContextChips(data);

  // Pitch tag
  const pitchClass = { batting:'pitch-bat', balanced:'pitch-bal', bowling:'pitch-bowl' }
    [data.pitch] || 'pitch-bal';
  const chips = [
    `<span class="pred-chip ${pitchClass}">⛏ ${data.pitch}</span>`,
    `<span class="pred-chip dew">💧 ${(data.dew * 100).toFixed(0)}%</span>`,
    data.chasing ? `<span class="pred-chip chase">🌊 chasing</span>` : '',
  ].join('');

  const predEl = document.getElementById(`pred-${pid}`);
  if (!predEl) return;

  predEl.innerHTML = `
    <div class="pred-numbers">
      <span class="pred-mid">${md}</span>
      <span class="pred-unit">${unit}</span>
    </div>
    <div class="pred-range-bar">
      <div class="pred-range-fill ${task.role}"
           style="left:${loFr}%;width:${wFr}%"></div>
    </div>
    <div class="pred-range-text">${lo} – ${hi}</div>
    <div class="pred-chips">${chips}</div>`;
}

function renderPredictionError(task, match) {
  const pid    = playerDomId(task.name, match);
  const predEl = document.getElementById(`pred-${pid}`);
  if (predEl) predEl.innerHTML = `<span class="pred-error">not in model</span>`;
  const subEl = document.getElementById(`sub-${pid}`);
  if (subEl)  subEl.textContent = 'no profile data';
}

// ── CONFIDENCE ────────────────────────────────────────────────────────────────

function computeConfidence(lo, hi, mid, careerAvg, role) {
  const width = hi - lo;
  if (role === 'bat') {
    const ratio = width / Math.max(mid || careerAvg, 1);
    if (ratio < 1.0)  return 'HIGH';
    if (ratio < 2.0)  return 'MED';
    return 'LOW';
  } else {
    if (width <= 1)   return 'HIGH';
    if (width <= 2)   return 'MED';
    return 'LOW';
  }
}

// ── CONTEXT CHIPS ─────────────────────────────────────────────────────────────

let ctxChipsSet = false;

function updateContextChips(data) {
  if (ctxChipsSet) return;
  ctxChipsSet = true;

  const el = document.getElementById('ctx-chips');
  if (!el) return;

  const pitchClass = `pitch-${data.pitch}`;
  const dewClass   = data.dew > 0.55 ? 'dew-high' : 'dew-low';
  const chaseClass = data.chasing ? 'chase-yes' : 'chase-no';

  el.innerHTML = `
    <span class="ctx-chip ${pitchClass}">⛏ ${capitalise(data.pitch)} pitch</span>
    <span class="ctx-chip ${dewClass}">💧 Dew ${(data.dew * 100).toFixed(0)}%</span>
    <span class="ctx-chip ${chaseClass}">🏃 Chasing ${data.chasing ? 'advantage' : 'neutral'}</span>`;
}

// ── PROJECTED TOTALS ──────────────────────────────────────────────────────────
// Accumulated directly from API data as each fetch resolves.

const projTotals = {};

function initProjTotals(match) {
  for (const key of Object.keys(projTotals)) delete projTotals[key];
  for (const team of Object.keys(match.teams))
    projTotals[safeId(team)] = { lo: 0, mid: 0, hi: 0, n: 0 };
}

function accumulateBatTotal(team, lo, mid, hi) {
  const key = safeId(team);
  if (!projTotals[key]) return;
  projTotals[key].lo  += lo;
  projTotals[key].mid += mid;
  projTotals[key].hi  += hi;
  projTotals[key].n   += 1;
}

function flushProjectedTotals(match) {
  let maxMid = 0;
  for (const t of Object.values(projTotals))
    if (t.n > 0) maxMid = Math.max(maxMid, t.mid);

  for (const [key, totals] of Object.entries(projTotals)) {
    const card    = document.getElementById(`proj-${key}`);
    if (!card) continue;

    const loadEl  = document.getElementById(`proj-loading-${key}`);
    const barWrap = document.getElementById(`proj-bar-${key}`);

    if (totals.n === 0) {
      if (loadEl) loadEl.outerHTML = `<div class="proj-range" style="color:var(--ink3)">no profiled batters</div>`;
      continue;
    }

    card.classList.add('loaded');

    if (loadEl) {
      loadEl.insertAdjacentHTML('beforebegin', `
        <div class="proj-total">${Math.round(totals.mid)}</div>
        <div class="proj-range">${Math.round(totals.lo)} – ${Math.round(totals.hi)} est.</div>`);
      loadEl.remove();
    }

    // Fill the visual bar proportional to the max team total
    if (barWrap && maxMid > 0) {
      barWrap.style.display = '';
      const pct = Math.min((totals.mid / maxMid) * 100, 100).toFixed(1);
      barWrap.querySelector('.proj-bar-fill').style.width = pct + '%';
    }
  }
}

// ── EXPORT CSV ────────────────────────────────────────────────────────────────

function exportCSV() {
  const match = state.activeMatch;
  if (!match) { showToast('Select a match first'); return; }

  const mKey = matchKey(match);
  const cache = state.cache[mKey];
  if (!cache || !Object.keys(cache).length) {
    showToast('Predictions still loading…'); return;
  }

  const rows = [['Match','Date','Home','Away','Venue','Player','Team','Role',
    'Pred_Low','Pred_Mid','Pred_High','Career_Avg','Innings','Confidence',
    'Pitch','Dew','Chasing']];

  for (const [team, squad] of Object.entries(match.teams)) {
    for (const p of [...squad.bat, ...squad.bowl]) {
      if (!p.in_profile) continue;
      const d = cache[p.name];
      if (!d) continue;
      const role   = squad.bat.some(x => x.name === p.name) ? 'BAT' : 'BOWL';
      const isBat  = role === 'BAT';
      const lo     = isBat ? d.runs_low  : d.wkts_low;
      const md     = isBat ? d.runs_mid  : d.wkts_mid;
      const hi     = isBat ? d.runs_high : d.wkts_high;
      const conf   = computeConfidence(lo, hi, md, d.career_avg, isBat ? 'bat' : 'bowl');
      rows.push([match.match, match.date, match.home, match.away,
        `"${match.venue}"`, `"${p.name}"`, team, role,
        lo, md, hi, d.career_avg, d.innings, conf,
        d.pitch, d.dew, d.chasing]);
    }
  }

  const csv  = rows.map(r => r.join(',')).join('\n');
  const blob = new Blob([csv], { type: 'text/csv' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url;
  a.download = `criciq_m${match.match}_${match.home}_vs_${match.away}.csv`;
  a.click();
  URL.revokeObjectURL(url);
  showToast('CSV exported ✓');
}

// ── TOAST ─────────────────────────────────────────────────────────────────────

function showToast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  setTimeout(() => el.classList.remove('show'), 2400);
}

// ── PLACEHOLDER STATES ────────────────────────────────────────────────────────

function showDetailPlaceholder() {
  ctxChipsSet = false;
  document.getElementById('match-detail').innerHTML = `
    <div class="state-placeholder">
      <div class="state-icon">🏏</div>
      <div class="state-title">Select a match</div>
      <div class="state-sub">Predictions load live from the Flask API.<br>Each player is fetched independently as you click.</div>
    </div>`;
}

function showListLoading() {
  document.getElementById('match-list').innerHTML = `
    <div class="state-placeholder" style="min-height:200px">
      <div class="spinner"></div>
      <div class="state-sub">LOADING SCHEDULE</div>
    </div>`;
}

function showListError(msg) {
  document.getElementById('match-list').innerHTML = `
    <div class="state-placeholder" style="min-height:200px">
      <div class="state-icon">⚠</div>
      <div class="state-sub" style="color:var(--red)">${msg}</div>
    </div>`;
}

// ── UTILS ─────────────────────────────────────────────────────────────────────

function playerDomId(name, match) {
  return `${match.format}-${match.match}-${name}`.replace(/[^a-zA-Z0-9-]/g, '_');
}

function safeId(name) {
  return name.replace(/[^a-zA-Z0-9]/g, '_');
}

function matchKey(m) {
  return `${m.format}-${m.match}`;
}

function capitalise(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}

function formatDate(d) {
  return new Date(d).toLocaleDateString('en-IN',
    { day: 'numeric', month: 'short', year: 'numeric' });
}

/* ═══════════════════════════════════════════════════════════════════════════
   NEW TABS — Player Insights · Player Comparison · Model Summary
   All predictions come from the Flask /predict* API.  No hardcoded values.
   ═══════════════════════════════════════════════════════════════════════════ */

// ── TAB STATE ─────────────────────────────────────────────────────────────────

// Player lists loaded per format, keyed by fmt ('ipl' | 't20i')
const playerLists = {};

// Venue lists loaded per format
const venueLists = {};

// Track which tab is active so the match tabs keep their state
let activeTab = 'ipl';

// ── TAB SWITCHING ─────────────────────────────────────────────────────────────

/**
 * switchTab replaces the old switchFormat for the two match tabs and adds
 * three new tab IDs: 'insights', 'compare', 'model'.
 *
 * For the match tabs (ipl / t20i) we keep calling switchFormat() internally
 * so all existing match-tab logic is completely unchanged.
 */
function switchTab(tab) {
  activeTab = tab;

  // Update tab button styles
  document.querySelectorAll('.format-tab').forEach(btn =>
    btn.classList.toggle('active', btn.dataset.format === tab));

  // Show/hide the match layout vs the new tab panels
  const matchLayout = document.querySelector('.main-layout');
  const insPanel    = document.getElementById('panel-insights');
  const cmpPanel    = document.getElementById('panel-compare');
  const modPanel    = document.getElementById('panel-model');

  const isMatchTab  = tab === 'ipl' || tab === 't20i';

  matchLayout.style.display = isMatchTab ? '' : 'none';
  insPanel.style.display    = tab === 'insights' ? '' : 'none';
  cmpPanel.style.display    = tab === 'compare'  ? '' : 'none';
  modPanel.style.display    = tab === 'model'    ? '' : 'none';

  if (isMatchTab) {
    // Delegate to the existing match format handler
    switchFormat(tab);
    return;
  }

  if (tab === 'insights') {
    insInit();
    return;
  }

  if (tab === 'compare') {
    cmpInit();
    return;
  }

  if (tab === 'model') {
    modelInit();
    return;
  }
}

// ── SHARED HELPERS ────────────────────────────────────────────────────────────

/**
 * Load /api/players for a given format, with caching.
 * Returns the array; subsequent calls return from cache immediately.
 */
async function loadPlayerList(fmt) {
  if (playerLists[fmt]) return playerLists[fmt];
  const res = await fetch(`/api/players?fmt=${fmt}`);
  playerLists[fmt] = await safeJson(res, '/api/players');
  return playerLists[fmt];
}

/**
 * Load /api/venues for a given format, with caching.
 */
async function loadVenueList(fmt) {
  if (venueLists[fmt]) return venueLists[fmt];
  const res  = await fetch('/api/venues');
  const data = await safeJson(res, '/api/venues');
  venueLists['ipl']  = data.ipl  || [];
  venueLists['t20i'] = data.t20i || [];
  return venueLists[fmt] || [];
}

/**
 * Populate a <select> from an array of strings.
 */
function fillSelect(id, items, firstOption = null) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = '';
  if (firstOption) {
    const o = document.createElement('option');
    o.value = ''; o.textContent = firstOption;
    el.appendChild(o);
  }
  items.forEach(item => {
    const o = document.createElement('option');
    o.value = o.textContent = item;
    el.appendChild(o);
  });
}

/**
 * Get IPL team names from the schedule (already loaded).
 */
function getTeamList() {
  const teams = new Set();
  for (const fmt of ['ipl', 't20i']) {
    for (const m of (state.schedule[fmt] || [])) {
      teams.add(m.home); teams.add(m.away);
    }
  }
  return [...teams].sort();
}

/**
 * Chip HTML for pitch/dew/chasing — shared across both new tabs.
 */
function contextChipHTML(data) {
  const pitchClass = { batting:'pitch-bat', balanced:'pitch-bal', bowling:'pitch-bowl' }
    [data.pitch] || 'pitch-bal';
  return [
    `<span class="pred-chip ${pitchClass}">⛏ ${data.pitch}</span>`,
    `<span class="pred-chip dew">💧 ${(data.dew * 100).toFixed(0)}%</span>`,
    data.chasing ? `<span class="pred-chip chase">🌊 chasing</span>` : '',
  ].join('');
}

// ── PLAYER INSIGHTS ──────────────────────────────────────────────────────────
// Loads once from /api/player_overviews, renders a grid of player cards.
// Filters (role, archetype) operate on the cached data — no re-fetching.

let insOverviewData  = null;   // cached API response
let insRoleFilter    = 'ALL';
let insArchFilter    = 'ALL';

async function insInit() {
  if (insOverviewData) { insRenderGrid(); return; }

  document.getElementById('ins-grid').innerHTML = `
    <div class="state-placeholder" style="grid-column:1/-1">
      <div class="spinner"></div>
      <div class="state-sub">Loading player profiles…</div>
    </div>`;

  try {
    const res = await fetch('/api/player_overviews');
    insOverviewData = await safeJson(res, '/api/player_overviews');
    insRenderGrid();
  } catch (err) {
    document.getElementById('ins-grid').innerHTML = `
      <div class="state-placeholder" style="grid-column:1/-1">
        <div class="state-icon">⚠</div>
        <div class="state-sub" style="color:var(--red)">${err.message}</div>
      </div>`;
  }
}

function insSetRole(role, btn) {
  insRoleFilter = role;
  btn.closest('.overview-pills').querySelectorAll('.ov-pill')
     .forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  insRenderGrid();
}

function insSetArch(arch, btn) {
  insArchFilter = arch;
  btn.closest('.overview-pills').querySelectorAll('.ov-pill')
     .forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  insRenderGrid();
}

function insRenderGrid() {
  if (!insOverviewData) return;

  // Merge batters + bowlers into one list
  let all = [...insOverviewData.batters, ...insOverviewData.bowlers];

  // Role filter
  if (insRoleFilter !== 'ALL') all = all.filter(p => p.role === insRoleFilter);

  // Archetype filter — partial match so "Economy" matches "Economy + Wickets" etc.
  if (insArchFilter !== 'ALL') {
    all = all.filter(p => p.archetype.toLowerCase().includes(insArchFilter.toLowerCase()));
  }

  const grid = document.getElementById('ins-grid');
  if (!all.length) {
    grid.innerHTML = `<div class="state-placeholder" style="grid-column:1/-1">
      <div class="state-sub">No players match this filter.</div></div>`;
    return;
  }

  grid.innerHTML = all.map(p => insPlayerCardHTML(p)).join('');

  // Animate form bars after paint
  requestAnimationFrame(() => {
    grid.querySelectorAll('.ov-form-fill[data-w]').forEach(el => {
      el.style.width = el.dataset.w + '%';
    });
  });
}

function insPlayerCardHTML(p) {
  const isBat     = p.role === 'BAT';
  const archClass = `arch-${p.archetype}`.replace(/ /g, '\ ');
  const formClass = p.form; // 'hot' | 'good' | 'steady' | 'cold'

  // Form bar width: compare form5 to career, capped at 120%
  const formRatio  = p.form5_avg / Math.max(isBat ? p.career_avg : p.career_avg, 0.01);
  const formWidth  = Math.min(formRatio * 70, 100).toFixed(0);  // 70% = career average baseline
  const fillClass  = p.form === 'hot' ? 'hot' : (isBat ? 'bat' : 'bowl');

  const mainStat   = isBat ? p.career_avg : p.career_avg;
  const secondStat = isBat ? p.career_sr  : p.career_econ;
  const mainLabel  = isBat ? 'AVG'  : 'AVG WKT';
  const secLabel   = isBat ? 'SR'   : 'ECON';
  const form5Label = isBat ? `${p.form5_avg} (last 5)` : `${p.form5_avg} (last 5)`;

  const venueHTML = p.best_venue
    ? `<div class="ov-venue">
        <div style="color:var(--ink3);font-size:8px;letter-spacing:1px;text-transform:uppercase;margin-bottom:2px">
          Best venue
        </div>
        <div class="ov-venue-name">${p.best_venue.venue}</div>
        <div style="color:var(--ink3)">
          ${isBat
            ? `${p.best_venue.avg_runs} avg · ${p.best_venue.avg_sr} SR`
            : `${p.best_venue.avg_wkts} avg · ${p.best_venue.avg_econ} econ`}
          · ${p.best_venue.matches}m
        </div>
      </div>`
    : '';

  const specs = (p.specialities || []).map(s =>
    `<span class="ov-spec-tag">${s}</span>`).join('');

  return `
  <div class="ov-card" data-role="${p.role}" data-arch="${p.archetype}">
    <div class="ov-card-top">
      <div class="ov-name">${p.name}</div>
      <span class="ov-arch-badge ${archClass}">${p.archetype}</span>
    </div>

    <div class="ov-stats">
      <div class="ov-stat">
        <div class="ov-stat-label">${mainLabel}</div>
        <div class="ov-stat-value accent">${mainStat}</div>
      </div>
      <div class="ov-stat">
        <div class="ov-stat-label">${secLabel}</div>
        <div class="ov-stat-value">${secondStat}</div>
      </div>
      <div class="ov-stat">
        <div class="ov-stat-label">INN</div>
        <div class="ov-stat-value">${p.innings}</div>
      </div>
    </div>

    <div class="ov-form-row">
      <span class="ov-form-label">FORM</span>
      <div class="ov-form-track">
        <div class="ov-form-fill ${fillClass}" style="width:0%" data-w="${formWidth}"></div>
      </div>
      <span class="ov-form-badge ${formClass}">${form5Label}</span>
    </div>

    ${specs ? `<div class="ov-specialities">${specs}</div>` : ''}
    ${venueHTML}
  </div>`;
}

// ── PLAYER COMPARISON / RIVALRIES ─────────────────────────────────────────────
// Displays pre-defined bat-vs-bowl rivalries derived from shared IPL history.
// No user inputs — the data speaks for itself.

let cmpOverviewData = null;
let cmpEdgeFilter   = 'ALL';

async function cmpInit() {
  if (cmpOverviewData) { cmpRenderList(); return; }

  document.getElementById('cmp-list').innerHTML = `
    <div class="state-placeholder">
      <div class="spinner"></div>
      <div class="state-sub">Loading rivalries…</div>
    </div>`;

  try {
    // Re-use the same endpoint as Insights — both tabs share the same data
    const res = await fetch('/api/player_overviews');
    const data = await safeJson(res, '/api/player_overviews');
    // Share cache with insights tab too
    insOverviewData = insOverviewData || data;
    cmpOverviewData = data;
    cmpRenderList();
  } catch (err) {
    document.getElementById('cmp-list').innerHTML = `
      <div class="state-placeholder">
        <div class="state-icon">⚠</div>
        <div class="state-sub" style="color:var(--red)">${err.message}</div>
      </div>`;
  }
}

function cmpSetEdge(edge, btn) {
  cmpEdgeFilter = edge;
  btn.closest('.overview-pills').querySelectorAll('.ov-pill')
     .forEach(p => p.classList.remove('active'));
  btn.classList.add('active');
  cmpRenderList();
}

function cmpRenderList() {
  if (!cmpOverviewData) return;

  let rivalries = cmpOverviewData.rivalries || [];
  if (cmpEdgeFilter !== 'ALL') rivalries = rivalries.filter(r => r.edge === cmpEdgeFilter);

  const list = document.getElementById('cmp-list');
  if (!rivalries.length) {
    list.innerHTML = `<div class="state-placeholder">
      <div class="state-sub">No rivalries match this filter.</div></div>`;
    return;
  }

  list.innerHTML = rivalries.map(r => cmpRivalryCardHTML(r)).join('');

  requestAnimationFrame(() => {
    list.querySelectorAll('[data-bar-w]').forEach(el => {
      el.style.width = el.dataset.barW + '%';
    });
  });
}

function cmpRivalryCardHTML(r) {
  const edgeLabel = r.edge === 'bat'  ? 'Bat Edge'  :
                    r.edge === 'bowl' ? 'Bowl Edge' : 'Even';

  // Normalise bars: career avg vs a 100-run baseline for bat, 2 wickets for bowl
  const batBar    = Math.min(r.bat_avg  / 50  * 100, 100).toFixed(0);
  const batSrBar  = Math.min(r.bat_sr   / 180 * 100, 100).toFixed(0);
  const batFBar   = Math.min(r.bat_form / 50  * 100, 100).toFixed(0);
  const bwlBar    = Math.min(r.bowl_avg  / 2   * 100, 100).toFixed(0);
  const econBar   = Math.min((12 - r.bowl_econ) / 6 * 100, 100).toFixed(0); // lower econ = taller bar
  const bwlFBar   = Math.min(r.bowl_form / 2   * 100, 100).toFixed(0);

  const sharedHTML = r.shared_venues && r.shared_venues.length
    ? `<div style="margin-top:2px">
        <div style="font-family:var(--mono);font-size:8px;letter-spacing:1px;
             text-transform:uppercase;color:var(--ink3);margin-bottom:5px">
          Shared IPL venues
        </div>
        <div class="rv-venues">
          ${r.shared_venues.map(v => `
            <div class="rv-venue-row">
              <div class="rv-venue-name">${v.venue}</div>
              <div class="rv-venue-bat">${v.bat_avg} avg</div>
              <div class="rv-venue-bowl">${v.bowl_avg} wkt</div>
            </div>`).join('')}
        </div>
      </div>`
    : '';

  return `
  <div class="rivalry-card" data-edge="${r.edge}">
    <div class="rv-header">
      <div class="rv-player">
        <div class="rv-name">${r.bat}</div>
        <div class="rv-arch">${r.bat_arch} · ${r.bat_avg} avg · SR ${r.bat_sr}</div>
      </div>
      <div class="rv-vs">
        <div class="rv-vs-text">vs</div>
        <span class="rv-edge-badge ${r.edge}">${edgeLabel}</span>
      </div>
      <div class="rv-player right">
        <div class="rv-name">${r.bowl}</div>
        <div class="rv-arch">${r.bowl_arch} · ${r.bowl_avg} wkt · ${r.bowl_econ} econ</div>
      </div>
    </div>

    <div class="rv-body">
      <div class="rv-stats-row">

        <!-- Batter stats -->
        <div class="rv-stat-group">
          <div class="rv-stat-title">🏏 ${r.bat} — Batting</div>
          <div class="rv-bars">
            <div class="rv-bar-row">
              <div class="rv-bar-key">Career</div>
              <div class="rv-bar-track"><div class="rv-bar-fill bat" style="width:0%" data-bar-w="${batBar}"></div></div>
              <div class="rv-bar-val">${r.bat_avg}</div>
            </div>
            <div class="rv-bar-row">
              <div class="rv-bar-key">Strike R</div>
              <div class="rv-bar-track"><div class="rv-bar-fill bat" style="width:0%;opacity:.6" data-bar-w="${batSrBar}"></div></div>
              <div class="rv-bar-val">${r.bat_sr}</div>
            </div>
            <div class="rv-bar-row">
              <div class="rv-bar-key">Last 5</div>
              <div class="rv-bar-track"><div class="rv-bar-fill ${r.bat_form > r.bat_avg * 1.1 ? 'hot ov-form-fill' : 'bat'}" style="width:0%" data-bar-w="${batFBar}"></div></div>
              <div class="rv-bar-val">${r.bat_form}</div>
            </div>
          </div>
        </div>

        <!-- Bowler stats -->
        <div class="rv-stat-group">
          <div class="rv-stat-title">🎯 ${r.bowl} — Bowling</div>
          <div class="rv-bars">
            <div class="rv-bar-row">
              <div class="rv-bar-key">Wkt Avg</div>
              <div class="rv-bar-track"><div class="rv-bar-fill bowl" style="width:0%" data-bar-w="${bwlBar}"></div></div>
              <div class="rv-bar-val">${r.bowl_avg}</div>
            </div>
            <div class="rv-bar-row">
              <div class="rv-bar-key">Economy</div>
              <div class="rv-bar-track"><div class="rv-bar-fill bowl" style="width:0%;opacity:.6" data-bar-w="${econBar}"></div></div>
              <div class="rv-bar-val">${r.bowl_econ}</div>
            </div>
            <div class="rv-bar-row">
              <div class="rv-bar-key">Last 5</div>
              <div class="rv-bar-track"><div class="rv-bar-fill bowl" style="width:0%" data-bar-w="${bwlFBar}"></div></div>
              <div class="rv-bar-val">${r.bowl_form}</div>
            </div>
          </div>
        </div>

      </div>

      <div class="rv-context">${r.context}</div>
      ${sharedHTML}
    </div>
  </div>`;
}

// ── MODEL SUMMARY ─────────────────────────────────────────────────────────────

let modelSummaryLoaded = false;

async function modelInit() {
  if (modelSummaryLoaded) return;  // already rendered, no need to refetch

  const out = document.getElementById('model-output');
  out.innerHTML = `<div class="state-placeholder" style="min-height:160px">
    <div class="spinner"></div><div class="state-sub">Loading model summary…</div></div>`;

  try {
    const res  = await fetch('/api/model_summary');
    const ms   = await safeJson(res, '/api/model_summary');
    renderModelSummary(ms);
    modelSummaryLoaded = true;
  } catch (err) {
    out.innerHTML = `<div style="color:var(--red);font-family:var(--mono);font-size:12px;padding:20px">
      Error: ${err.message}</div>`;
  }
}

function renderModelSummary(ms) {
  const out = document.getElementById('model-output');

  // Feature importance rows HTML
  function fiRows(items, cls) {
    const maxPct = items[0]?.importance || 1;
    return items.map(f => `
      <div class="fi-row">
        <div class="fi-name">${f.feature}</div>
        <div class="fi-track">
          <div class="fi-fill ${cls}" style="width:0%" data-target="${(f.importance/maxPct*100).toFixed(1)}"></div>
        </div>
        <div class="fi-pct">${f.importance}%</div>
      </div>`).join('');
  }

  // Feature explanation cards
  const expCards = Object.entries(ms.feature_explanations).map(([key, text]) => `
    <div class="feat-exp-card">
      <div class="feat-exp-name">${key}</div>
      <div class="feat-exp-text">${text}</div>
    </div>`).join('');

  out.innerHTML = `
    <!-- Overview -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot"></div>Algorithm Overview</div>
      <div class="model-section-body">
        <div class="model-metrics-row">
          <div class="model-metric">
            <div class="mm-label">Algorithm</div>
            <div class="mm-value" style="font-size:13px;letter-spacing:0">${ms.algorithm}</div>
          </div>
          <div class="model-metric green">
            <div class="mm-label">Quantiles</div>
            <div class="mm-value" style="font-size:13px">${ms.quantiles.join(' · ')}</div>
          </div>
          <div class="model-metric amber">
            <div class="mm-label">Train / Test Split</div>
            <div class="mm-value" style="font-size:13px;letter-spacing:0">${ms.train_split}</div>
          </div>
        </div>
        <div style="font-family:var(--mono);font-size:10px;color:var(--ink3);margin-top:12px;line-height:1.7">
          Three quantile models (Q25, Q50, Q75) are trained independently per format.
          The Q50 model provides the median prediction; Q25 and Q75 form the 50% confidence interval.
          Interval coverage target is 50% — a well-calibrated model has half of actuals fall within the range.
        </div>
      </div>
    </div>

    <!-- IPL Batting -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot" style="background:var(--blue)"></div>IPL Batting Model</div>
      <div class="model-section-body">
        <div class="model-metrics-row" style="margin-bottom:18px">
          <div class="model-metric"><div class="mm-label">MAE</div><div class="mm-value">${ms.ipl.bat.mae}</div><div class="mm-sub">runs (median)</div></div>
          <div class="model-metric green"><div class="mm-label">50% Coverage</div><div class="mm-value">${ms.ipl.bat.coverage_50}%</div><div class="mm-sub">target ~50%</div></div>
          <div class="model-metric amber"><div class="mm-label">Features</div><div class="mm-value">${ms.ipl.bat.n_features}</div><div class="mm-sub">input columns</div></div>
        </div>
        <div class="fi-list" id="fi-ipl-bat">${fiRows(ms.ipl.bat.importances, '')}</div>
      </div>
    </div>

    <!-- IPL Bowling -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot" style="background:var(--red)"></div>IPL Bowling Model</div>
      <div class="model-section-body">
        <div class="model-metrics-row" style="margin-bottom:18px">
          <div class="model-metric"><div class="mm-label">MAE</div><div class="mm-value">${ms.ipl.bowl.mae}</div><div class="mm-sub">wickets (median)</div></div>
          <div class="model-metric green"><div class="mm-label">50% Coverage</div><div class="mm-value">${ms.ipl.bowl.coverage_50}%</div><div class="mm-sub">over-covers (discrete target)</div></div>
          <div class="model-metric amber"><div class="mm-label">Features</div><div class="mm-value">${ms.ipl.bowl.n_features}</div><div class="mm-sub">input columns</div></div>
        </div>
        <div class="fi-list">${fiRows(ms.ipl.bowl.importances, 'bowl')}</div>
      </div>
    </div>

    <!-- T20I Batting -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot" style="background:var(--purple)"></div>T20I Batting Model</div>
      <div class="model-section-body">
        <div class="model-metrics-row" style="margin-bottom:18px">
          <div class="model-metric"><div class="mm-label">MAE</div><div class="mm-value">${ms.t20i.bat.mae}</div><div class="mm-sub">runs (median)</div></div>
          <div class="model-metric green"><div class="mm-label">50% Coverage</div><div class="mm-value">${ms.t20i.bat.coverage_50}%</div><div class="mm-sub">target ~50%</div></div>
          <div class="model-metric amber"><div class="mm-label">Features</div><div class="mm-value">${ms.t20i.bat.n_features}</div><div class="mm-sub">input columns</div></div>
        </div>
        <div class="fi-list">${fiRows(ms.t20i.bat.importances, '')}</div>
      </div>
    </div>

    <!-- T20I Bowling -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot" style="background:var(--amber)"></div>T20I Bowling Model</div>
      <div class="model-section-body">
        <div class="model-metrics-row" style="margin-bottom:18px">
          <div class="model-metric"><div class="mm-label">MAE</div><div class="mm-value">${ms.t20i.bowl.mae}</div><div class="mm-sub">wickets (median)</div></div>
          <div class="model-metric green"><div class="mm-label">50% Coverage</div><div class="mm-value">${ms.t20i.bowl.coverage_50}%</div><div class="mm-sub">over-covers (discrete target)</div></div>
          <div class="model-metric amber"><div class="mm-label">Features</div><div class="mm-value">${ms.t20i.bowl.n_features}</div><div class="mm-sub">input columns</div></div>
        </div>
        <div class="fi-list">${fiRows(ms.t20i.bowl.importances, 'bowl')}</div>
      </div>
    </div>

    <!-- Feature explanations -->
    <div class="model-section">
      <div class="model-section-title"><div class="dot" style="background:var(--green)"></div>Contextual Feature Explanations</div>
      <div class="model-section-body">
        <div class="feat-exp-grid">${expCards}</div>
      </div>
    </div>`;

  // Animate feature importance bars after paint
  requestAnimationFrame(() => {
    document.querySelectorAll('.fi-fill[data-target]').forEach(bar => {
      bar.style.width = bar.dataset.target + '%';
    });
  });
}

// ── SYNC ──────────────────────────────────────────────────────────────────────

async function loadSyncStatus() {
  try {
    const res  = await fetch('/api/sync/status');
    if (!res.ok) return;
    const data = await res.json();
    renderSyncBadge(data);
  } catch { /* sync endpoint optional */ }
}

function renderSyncBadge(data) {
  const el = document.getElementById('sync-badge');
  if (!el) return;
  if (!data.cached) {
    el.textContent = 'Not synced';
    el.title = 'Run python 07_sync_schedule.py to sync';
    return;
  }
  const ipl  = data.ipl  || {};
  const t20i = data.t20i || {};
  const done = (ipl.completed || 0) + (t20i.completed || 0);
  const up   = (ipl.upcoming  || 0) + (t20i.upcoming  || 0);
  el.textContent = `${data.age_human} · ${done} done · ${up} upcoming`;
  el.title = `Last synced ${data.age_human}`;
}

async function triggerSync() {
  if (syncState.syncing) return;
  syncState.syncing = true;

  const btn = document.getElementById('sync-btn');
  if (btn) { btn.disabled = true; btn.textContent = '⟳ Syncing…'; }
  showToast('Sync started — schedule will refresh shortly…');

  try {
    const res = await fetch('/api/sync', { method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ enrich: false }) });
    const data = await res.json();
    if (data.status === 'started') {
      // Poll for completion: check sync/status every 3s for up to 30s
      let attempts = 0;
      const poll = setInterval(async () => {
        attempts++;
        const s = await fetch('/api/sync/status').then(r => r.json()).catch(() => null);
        if (s && s.age_seconds < 15) {
          clearInterval(poll);
          syncState.syncing = false;
          if (btn) { btn.disabled = false; btn.textContent = '⟳ Sync'; }
          renderSyncBadge(s);
          // Reload schedule to pick up new data
          await loadSchedule();
          showToast('Schedule updated ✓');
        }
        if (attempts >= 10) {
          clearInterval(poll);
          syncState.syncing = false;
          if (btn) { btn.disabled = false; btn.textContent = '⟳ Sync'; }
          showToast('Sync running in background — reload to see updates');
        }
      }, 3000);
    }
  } catch (err) {
    syncState.syncing = false;
    if (btn) { btn.disabled = false; btn.textContent = '⟳ Sync'; }
    showToast('Sync unavailable — is Flask running?');
  }
}
