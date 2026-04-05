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

const TODAY = '2026-04-05';

// ── BOOT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', loadSchedule);

// ── SCHEDULE ──────────────────────────────────────────────────────────────────

async function loadSchedule() {
  showListLoading();
  try {
    const res  = await fetch('/api/schedule');
    const data = await res.json();
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
    const isToday = m.date === TODAY;
    const isPast  = m.date <  TODAY;
    const badge   = isToday ? 'today' : isPast ? 'past' : 'upcoming';
    const label   = isToday ? 'TODAY' : isPast ? 'DONE' : 'UPCOMING';
    const active  = state.activeMatch?.match === m.match &&
                    state.activeMatch?.format === m.format ? 'active' : '';
    const dateStr = new Date(m.date).toLocaleDateString('en-IN',
                    { day: 'numeric', month: 'short' });

    // Show cached run totals if this match was already loaded
    const mKey   = matchKey(m);
    const totals = buildCachedTotals(mKey, m);
    const totalsHTML = totals
      ? `<div class="mi-totals">${totals}</div>`
      : '';

    return `<div class="match-item ${active}" onclick="selectMatch(${idx})" data-idx="${idx}">
      <div class="mi-header">
        <span class="mi-num">M${m.match} · ${m.format.toUpperCase()}</span>
        <span class="mi-badge ${badge}">${label}</span>
      </div>
      <div class="mi-teams">${m.home}<span class="mi-vs">vs</span>${m.away}</div>
      <div class="mi-venue">${m.venue}</div>
      <div style="font-family:var(--mono);font-size:9px;color:var(--ink3);margin-top:2px">${dateStr}</div>
      ${totalsHTML}
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

  // Reset per-match mutable flags
  ctxChipsSet = false;

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
      ${teamSectionHTML(match, match.home, match.teams[match.home])}
      ${teamSectionHTML(match, match.away, match.teams[match.away])}
    </div>`;

  // Stagger-reveal all player rows
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
