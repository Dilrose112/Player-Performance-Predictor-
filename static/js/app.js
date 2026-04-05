/**
 * app.js  —  CricIQ dynamic API-driven dashboard
 *
 * ALL prediction data comes from Flask API calls.
 * No hardcoded predictions anywhere in this file.
 *
 * Flow:
 *   1. loadSchedule()        → GET /api/schedule
 *   2. renderMatchList()     → builds left-panel cards from schedule metadata
 *   3. selectMatch(match)    → renders right panel + triggers all predictions
 *   4. renderMatchDetail()   → builds the match header card (venue, context chips)
 *   5. renderTeamSection()   → renders each team's player rows (skeleton state)
 *   6. getPrediction()       → POST /predict* per player, fills in results live
 */

'use strict';

// ── STATE ─────────────────────────────────────────────────────────────────────
const state = {
  format:        'ipl',       // 'ipl' | 't20i'
  schedule:      {},          // { ipl: [], t20i: [] }  from /api/schedule
  activeMatch:   null,        // currently selected match object
  pendingCalls:  new Set(),   // track in-flight fetch IDs for cancellation
};

const TODAY = '2026-04-05';

// ── BOOT ──────────────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  loadSchedule();
});

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
    showListError('Could not load schedule — is the Flask server running?');
    console.error(err);
  }
}

function updateTabCounts() {
  const ipl  = state.schedule.ipl  || [];
  const t20i = state.schedule.t20i || [];
  const iplTab  = document.getElementById('tab-ipl');
  const t20Tab  = document.getElementById('tab-t20i');
  if (iplTab)  iplTab.querySelector('.tab-count').textContent  = ipl.length;
  if (t20Tab)  t20Tab.querySelector('.tab-count').textContent  = t20i.length;
}

// ── FORMAT SWITCHING ──────────────────────────────────────────────────────────

function switchFormat(fmt) {
  state.format      = fmt;
  state.activeMatch = null;

  // Update tab styles
  document.querySelectorAll('.format-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.format === fmt);
  });

  renderMatchList();
  showDetailPlaceholder();
}

// ── MATCH LIST ────────────────────────────────────────────────────────────────

function renderMatchList() {
  const matches = state.schedule[state.format] || [];
  const panel   = document.getElementById('match-list');
  if (!matches.length) {
    panel.innerHTML = `<div class="state-placeholder" style="min-height:200px">
      <div class="state-sub">NO MATCHES</div></div>`;
    return;
  }

  panel.innerHTML = matches.map((m, idx) => {
    const isToday = m.date === TODAY;
    const isPast  = m.date <  TODAY;
    const badge   = isToday ? 'today' : isPast ? 'past' : 'upcoming';
    const label   = isToday ? 'TODAY' : isPast ? 'COMPLETED' : 'UPCOMING';
    const active  = state.activeMatch && state.activeMatch.match === m.match && state.activeMatch.format === m.format
                    ? 'active' : '';
    const dateStr = new Date(m.date).toLocaleDateString('en-IN',
                    { day: 'numeric', month: 'short' });

    return `<div class="match-item ${active}" onclick="selectMatch(${idx})" data-idx="${idx}">
      <div class="mi-header">
        <span class="mi-num">M${m.match} · ${m.format.toUpperCase()}</span>
        <span class="mi-badge ${badge}">${label}</span>
      </div>
      <div class="mi-teams">
        ${m.home} <span class="mi-vs">vs</span> ${m.away}
      </div>
      <div class="mi-venue">${m.venue}</div>
      <div class="mi-date">${dateStr}</div>
    </div>`;
  }).join('');
}

// ── SELECT MATCH ──────────────────────────────────────────────────────────────

function selectMatch(idx) {
  const matches = state.schedule[state.format] || [];
  const match   = matches[idx];
  if (!match) return;

  // Cancel any still-running prediction calls from prior selection
  state.pendingCalls.clear();

  state.activeMatch = match;

  // Highlight selected item
  document.querySelectorAll('.match-item').forEach((el, i) => {
    el.classList.toggle('active', i === idx);
  });

  renderMatchDetail(match);
}

// ── MATCH DETAIL ──────────────────────────────────────────────────────────────

function renderMatchDetail(match) {
  const panel = document.getElementById('match-detail');

  // Build skeleton HTML synchronously
  panel.innerHTML = `
    ${matchHeaderHTML(match)}
    <div class="projected-row" id="proj-row">
      ${projCardHTML(match.home, match.date)}
      ${projCardHTML(match.away, match.date)}
    </div>
    <div id="teams-container">
      ${teamSectionHTML(match, match.home, match.teams[match.home])}
      ${teamSectionHTML(match, match.away, match.teams[match.away])}
    </div>`;

  // Fire all prediction calls (they update DOM cells when they resolve)
  loadAllPredictions(match);
}

// ── MATCH HEADER HTML ─────────────────────────────────────────────────────────

function matchHeaderHTML(match) {
  return `
  <div class="match-header-card">
    <div class="mh-top">
      <span class="mh-matchno">MATCH ${match.match} · ${match.format.toUpperCase()}</span>
      <span class="mh-date">${formatDate(match.date)}</span>
    </div>
    <div class="mh-teams">
      <span class="mh-team">${match.home}</span>
      <span class="mh-vs">vs</span>
      <span class="mh-team">${match.away}</span>
    </div>
    <div class="mh-venue">📍 ${match.venue}</div>
    <div class="context-chips" id="ctx-chips">
      <span class="ctx-chip pitch-balanced">⛏ Loading…</span>
    </div>
  </div>`;
}

// ── PROJECTED TOTAL CARD ──────────────────────────────────────────────────────

function projCardHTML(team, date) {
  return `
  <div class="proj-card" id="proj-${safeId(team)}">
    <div class="proj-label">Projected Batting Runs</div>
    <div class="proj-team-name">${team}</div>
    <div class="proj-loading"><div class="spinner" style="width:16px;height:16px"></div> fetching…</div>
  </div>`;
}

// ── TEAM SECTION HTML ─────────────────────────────────────────────────────────

function teamSectionHTML(match, team, squad) {
  const bats  = squad.bat.filter(p => p.in_profile);
  const bowls = squad.bowl.filter(p => p.in_profile);
  const skipped = [...squad.bat, ...squad.bowl].filter(p => !p.in_profile);

  const rows = [
    bats.length  ? `<div class="role-divider">Batters</div>`  + bats.map(p  => playerRowHTML(p.name,  'bat', match)).join('') : '',
    bowls.length ? `<div class="role-divider">Bowlers</div>`  + bowls.map(p => playerRowHTML(p.name, 'bowl', match)).join('') : '',
    skipped.length ? `<div class="role-divider" style="color:var(--ink3)">No Profile Data</div>` +
      skipped.map(p => `<div class="player-row" style="opacity:.4">
        <div class="player-row-left">
          <div class="p-role-dot bat" style="background:var(--border)"></div>
          <div class="p-info"><div class="p-name">${p.name}</div>
          <div class="p-sub">Not in training data</div></div></div></div>`).join('') : '',
  ].join('');

  return `
  <div class="team-section">
    <div class="team-section-header">
      <span class="team-section-name">${team}</span>
      <span class="team-section-sub">${bats.length} BAT · ${bowls.length} BOWL</span>
    </div>
    ${rows}
  </div>`;
}

function playerRowHTML(name, role, match) {
  const pid = playerDomId(name, match);
  return `
  <div class="player-row" id="row-${pid}">
    <div class="player-row-left">
      <div class="p-role-dot ${role}"></div>
      <div class="p-info">
        <div class="p-name">${name}</div>
        <div class="p-sub" id="sub-${pid}">—</div>
      </div>
    </div>
    <div class="pred-widget" id="pred-${pid}">
      <div class="pred-loading"><div class="spinner"></div>predicting…</div>
    </div>
  </div>`;
}

// ── PREDICTION CALLS ──────────────────────────────────────────────────────────

/**
 * Fire one prediction API call per player in the match.
 * Each call resolves independently and updates its own DOM cell.
 * A callId is stored in state.pendingCalls; if the match changes
 * while calls are in flight, we discard stale results.
 */
async function loadAllPredictions(match) {
  const callId = Symbol(match.match + match.format);
  state.pendingCalls.add(callId);

  const teams  = match.teams;
  const fmt    = match.format;
  const venue  = match.venue;
  const date   = match.date;

  // Collect all player calls
  const tasks = [];
  for (const [team, squad] of Object.entries(teams)) {
    for (const p of squad.bat.filter(x => x.in_profile)) {
      tasks.push({ name: p.name, role: 'bat', team, venue, date, fmt });
    }
    for (const p of squad.bowl.filter(x => x.in_profile)) {
      tasks.push({ name: p.name, role: 'bowl', team, venue, date, fmt });
    }
  }

  // Resolve all; update DOM as each settles
  const promises = tasks.map(task =>
    getPrediction(task).then(result => {
      if (!state.pendingCalls.has(callId)) return; // match changed, discard
      renderPrediction(task, result, match);
    }).catch(err => {
      if (!state.pendingCalls.has(callId)) return;
      renderPredictionError(task, match);
      console.warn(`Prediction failed for ${task.name}:`, err);
    })
  );

  // After all settle, compute projected totals and update context chips
  Promise.allSettled(promises).then(() => {
    if (!state.pendingCalls.has(callId)) return;
    computeProjectedTotals(match);
    // Context chips are set from the first successful prediction result
  });
}

/**
 * Single prediction fetch.
 * Routes to the correct endpoint based on format and role.
 */
async function getPrediction({ name, role, team, venue, date, fmt }) {
  const endpointMap = {
    'ipl-bat':   '/predict',
    'ipl-bowl':  '/predict_bowl',
    't20i-bat':  '/predict_t20_bat',
    't20i-bowl': '/predict_t20_bowl',
  };
  const endpoint = endpointMap[`${fmt}-${role}`];
  const body = { player: name, venue, team, date };

  const res  = await fetch(endpoint, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify(body),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.error || `HTTP ${res.status}`);
  }

  const data = await res.json();
  if (data.error) throw new Error(data.error);
  return data;
}

// ── DOM UPDATES ───────────────────────────────────────────────────────────────

function renderPrediction(task, data, match) {
  const pid  = playerDomId(task.name, match);
  const role = task.role;
  const isBat = role === 'bat';

  const lo  = isBat ? data.runs_low  : data.wkts_low;
  const md  = isBat ? data.runs_mid  : data.wkts_mid;
  const hi  = isBat ? data.runs_high : data.wkts_high;
  const unit = isBat ? 'runs' : 'wkts';
  const maxVal = isBat ? 80 : 4;

  const loFrac  = Math.min(lo  / maxVal * 100, 100).toFixed(1);
  const hiFrac  = Math.min(hi  / maxVal * 100, 100).toFixed(1);
  const widthFr = Math.max(0, hiFrac - loFrac).toFixed(1);

  // Update sub-label with career stats
  const subEl = document.getElementById(`sub-${pid}`);
  if (subEl) {
    const avg = isBat ? `avg ${data.career_avg}` : `avg ${data.career_avg} wkt`;
    subEl.textContent = `${avg} · ${data.innings} inn`;
  }

  // Build context chips (from first bat result in the match)
  updateContextChips(data);

  // Pitch chip class
  const pitchClass = {
    batting:  'pitch-bat',
    balanced: 'pitch-bal',
    bowling:  'pitch-bowl',
  }[data.pitch] || 'pitch-bal';

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
      <div class="pred-range-fill ${role}"
           style="left:${loFrac}%;width:${widthFr}%"></div>
    </div>
    <div class="pred-range-text">${lo} – ${hi}</div>
    <div class="pred-chips">${chips}</div>`;
}

function renderPredictionError(task, match) {
  const pid   = playerDomId(task.name, match);
  const predEl = document.getElementById(`pred-${pid}`);
  if (predEl) predEl.innerHTML = `<span class="pred-error">not in model</span>`;
}

// ── CONTEXT CHIPS UPDATE ──────────────────────────────────────────────────────

let ctxChipsSet = false;

function updateContextChips(data) {
  if (ctxChipsSet) return;   // only need to do this once per match
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

function computeProjectedTotals(match) {
  for (const [team, squad] of Object.entries(match.teams)) {
    let totalLo = 0, totalMid = 0, totalHi = 0, n = 0;

    for (const p of squad.bat.filter(x => x.in_profile)) {
      const pid   = playerDomId(p.name, match);
      const predEl = document.getElementById(`pred-${pid}`);
      if (!predEl) continue;

      // Parse median value from the rendered DOM
      const midEl = predEl.querySelector('.pred-mid');
      if (!midEl) continue;
      const midVal = parseFloat(midEl.textContent) || 0;

      // Parse range text  "lo – hi"
      const rangeEl = predEl.querySelector('.pred-range-text');
      if (!rangeEl) continue;
      const parts = rangeEl.textContent.split('–').map(s => parseFloat(s.trim()) || 0);
      if (parts.length < 2) continue;

      totalLo  += parts[0];
      totalMid += midVal;
      totalHi  += parts[1];
      n++;
    }

    const projEl = document.getElementById(`proj-${safeId(team)}`);
    if (!projEl) continue;

    if (n === 0) {
      projEl.querySelector('.proj-loading').outerHTML =
        `<div class="proj-total">—</div><div class="proj-range">no profiled batters</div>`;
      continue;
    }

    // Replace loading spinner with numbers
    const loadEl = projEl.querySelector('.proj-loading');
    if (loadEl) {
      loadEl.outerHTML = `
        <div class="proj-total">${Math.round(totalMid)}</div>
        <div class="proj-range">${Math.round(totalLo)} – ${Math.round(totalHi)} est.</div>`;
    }
  }
}

// ── PLACEHOLDER STATES ────────────────────────────────────────────────────────

function showDetailPlaceholder() {
  ctxChipsSet = false;
  document.getElementById('match-detail').innerHTML = `
    <div class="state-placeholder">
      <div class="state-icon">🏏</div>
      <div class="state-title">Select a match</div>
      <div class="state-sub">Click any fixture on the left to load live predictions</div>
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

/** Stable DOM id from player name + match (avoids spaces / special chars) */
function playerDomId(name, match) {
  return `${match.format}-${match.match}-${name}`.replace(/[^a-zA-Z0-9-]/g, '_');
}

/** Stable DOM id from team name */
function safeId(name) {
  return name.replace(/[^a-zA-Z0-9]/g, '_');
}

function capitalise(s) {
  return s ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}

function formatDate(d) {
  return new Date(d).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' });
}
