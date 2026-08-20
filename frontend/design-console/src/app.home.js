/* ═══════════════════════════════════════════════════════════
   HOME + CATALOGUE
   ═══════════════════════════════════════════════════════════ */

/* A function, not a const array: hydrate() replaces SERVED and GATING after
   this file is parsed, and an array literal here would freeze the served run's
   numbers at whatever they were when the bundle was built. The six tiles,
   their labels, their order and their counters are exactly as designed — only
   the values behind them now come from /model, so a promotion moves them. */
const NOT_MEASURED = '—';
function missionStats() {
  const g = GATING || {};
  const kepler = (SERVED.missions || []).find(m => m.mission === 'Kepler');
  return [
    { label:'Candidates Scored', value:(SERVED.nScored || 5388).toLocaleString(),
      accent:false, count:SERVED.nScored || 5388, dur:2000 },
    { label:'High Confidence',   value:(SERVED.nHighConfidence || 146).toLocaleString(),
      accent:true,  count:SERVED.nHighConfidence || 146,  dur:1500 },
    // the pooled 0.955 is gone: the gating mission is the number that decides promotion
    { label:`${g.mission || 'TESS'} ROC-AUC`,
      value: g.auc != null ? g.auc.toFixed(4) : NOT_MEASURED, accent:false,
      sub: g.aucErr != null ? `±${g.aucErr.toFixed(4)} · gating mission` : 'not measured for this run' },
    { label:`${g.mission || 'TESS'} Recall @1% FPR`,
      value: g.recall != null ? g.recall.toFixed(4) : NOT_MEASURED, accent:false,
      sub: g.recallErr != null ? `±${g.recallErr.toFixed(4)} · shortlist criterion` : 'not measured for this run' },
    { label:'TESS Sectors',      value:'63',    accent:false, count:63,   dur:1200 },
    // 9,564 is the full public KOI catalogue; this is what was actually trained on
    { label:'Kepler KOIs Trained', value:(kepler ? kepler.n : 2500).toLocaleString(),
      accent:false, count: kepler ? kepler.n : 2500, dur:2200 },
  ];
}

const FLOATING_CANDIDATES = [
  { id:'TOI-4328.01',  period:'P=703.8d', prob:'0.989', x:'62%', y:'22%' },
  { id:'TOI-4565.01',  period:'P=412.1d', prob:'0.983', x:'52%', y:'52%' },
  { id:'TIC 77175217', period:'P=88.4d',  prob:'0.912', x:'78%', y:'42%' },
];

const PIPELINE = [
  { step:'01', title:'Catalog Refresh',  desc:'Automated ingestion from NASA ExoFOP, MAST, and Kepler archive. Nightly delta sync.' },
  { step:'02', title:'Validation Gates', desc:'Multi-stage quality filters: centroid shift, secondary eclipse, odd-even depth, ghost diagnostic.' },
  { step:'03', title:'GPU Training',     desc:'On-demand 11-branch CNN training on phase-folded light curves. Platt scaling for calibration.' },
  { step:'04', title:'Live Scoring',     desc:'Calibrated probability with per-fold agreement and MC-dropout spread. Every candidate scored in <2s.' },
];

const H1 = 'font-family:\'Space Grotesk\', sans-serif;font-size:clamp(3.5rem, 7vw, 6rem);line-height:1.0;letter-spacing:-0.03em;margin-bottom:0';

function Home() {
  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;position:relative">
    <section style="min-height:100vh;display:flex;flex-direction:column;justify-content:center;position:relative;overflow:hidden;padding-bottom:40px">
      <canvas id="hero-bg" style="position:absolute;inset:0;width:100%;height:100%;opacity:0.35"></canvas>
      <div style="position:absolute;inset:0;background:linear-gradient(to right, rgba(5,6,8,0.95) 40%, rgba(5,6,8,0.4) 70%, rgba(5,6,8,0.6) 100%)"></div>
      <div style="position:absolute;bottom:0;left:0;right:0;height:200px;background:linear-gradient(to bottom, transparent, #050608)"></div>

      ${FLOATING_CANDIDATES.map((c, i) => `
        <div class="rv hero-note" style="position:absolute;left:${c.x};top:${c.y};pointer-events:none;transition:opacity 1s ease ${0.8 + i * 0.2}s">
          <div style="display:flex;align-items:center;gap:0.4rem">
            <div style="width:6px;height:6px;border-radius:50%;border:1px solid #4DFFD2"></div>
            <div>
              <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:rgba(240,238,232,0.8)">${c.id}</div>
              <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:rgba(138,143,168,0.7)">${c.period} · <span style="color:#4DFFD2">${c.prob}</span></div>
            </div>
          </div>
        </div>`).join('')}

      <div class="page-pad" style="position:relative;z-index:2;padding:0 3rem;max-width:1440px;margin:0 auto;width:100%">
        <div class="hero-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:4rem;align-items:center">
          <div>
            <div class="section-label rv" style="margin-bottom:2rem;transform:translateY(16px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.1s">Exoplanet Hunter V2 — Transit Detection Console</div>
            <h1 class="rv" style="${H1};font-weight:700;color:#F0EEE8;transform:translateY(24px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.2s">HUNTING</h1>
            <h1 class="rv" style="${H1};font-weight:700;color:#F0EEE8;transform:translateY(24px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.3s">WORLDS</h1>
            <h1 class="rv" style="${H1};font-weight:300;color:rgba(240,238,232,0.25);margin-bottom:2.5rem;transform:translateY(24px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.4s">BEYOND</h1>
            <p class="rv" style="font-family:'Inter', sans-serif;font-size:1rem;line-height:1.7;color:rgba(240,238,232,0.6);max-width:480px;margin-bottom:2.5rem;transform:translateY(16px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.5s">
              A calibrated deep-learning pipeline vetting unconfirmed NASA transit candidates — with uncertainty you can trust.
            </p>
            <div class="rv" style="display:flex;gap:1rem;flex-wrap:wrap;transform:translateY(16px);transition:all 0.8s cubic-bezier(0.23,1,0.32,1) 0.6s">
              <button class="btn-teal" data-nav="#/catalogue">Explore Candidates →</button>
              <button class="btn-ghost" data-nav="#/upload">Submit Data</button>
            </div>
            <div class="rv" style="margin-top:4rem;display:flex;align-items:center;gap:0.75rem;transition:opacity 1s ease 1.2s" data-visible-opacity="0.5">
              <div style="width:1px;height:40px;background:linear-gradient(to bottom, transparent, rgba(77,255,210,0.6))"></div>
              <span style="font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.18em;text-transform:uppercase;color:#8A8FA8">Scroll to explore</span>
            </div>
          </div>
          <div class="hero-orbit rv" style="display:flex;justify-content:center;align-items:center;transition:opacity 1.2s ease 0.7s">
            ${orbitalDiagramHTML(460)}
          </div>
        </div>
      </div>

      ${healthPanelHTML()}
    </section>

    <div class="hairline"></div>

    <section class="page-pad" style="padding:4rem 3rem;background:#050608;position:relative;z-index:1">
      <div style="max-width:1440px;margin:0 auto">
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:2.5rem">
          <div class="section-label">Mission Status</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
            Serving <span style="color:#4DFFD2">${SERVED.runId}</span> · promoted ${SERVED.promotedAt}
          </div>
        </div>
        <div class="six-up" style="display:grid;grid-template-columns:repeat(6, 1fr);gap:2rem">
          ${missionStats().map(s => `
            <div style="border-top:1px solid ${s.accent ? 'rgba(77,255,210,0.3)' : 'rgba(255,255,255,0.1)'};padding-top:1.25rem">
              <div class="stat-label" style="margin-bottom:0.5rem">${s.label}</div>
              <div class="stat-value" style="font-size:1.75rem;font-weight:500;color:${s.accent ? '#4DFFD2' : '#F0EEE8'}"
                   ${s.count ? `data-count="${s.count}" data-dur="${s.dur}"` : ''}>${s.count ? '0' : s.value}</div>
              ${s.sub ? `<div style="font-family:'JetBrains Mono';font-size:0.6rem;color:rgba(138,143,168,0.8);margin-top:0.4rem">${s.sub}</div>` : ''}
            </div>`).join('')}
        </div>
      </div>
    </section>

    <div class="hairline"></div>

    <section class="page-pad" style="padding:8rem 3rem;background:#050608;position:relative;z-index:1">
      <div class="about-grid" style="max-width:1440px;margin:0 auto;display:grid;grid-template-columns:1fr 1fr;gap:8rem;align-items:start">
        <div>
          <div class="section-label" style="margin-bottom:1.5rem">What Is This</div>
          <h2 style="font-family:'Space Grotesk', sans-serif;font-size:clamp(2rem, 4vw, 3rem);font-weight:600;line-height:1.1;letter-spacing:-0.02em;color:#F0EEE8;margin-bottom:1.5rem">
            Every component must outperform the simplest baseline.
          </h2>
          <p style="font-family:'Inter';font-size:0.95rem;line-height:1.8;color:rgba(240,238,232,0.55);margin-bottom:1rem">
            Exoplanet Hunter V2 is a fully automated, cloud-first machine learning platform for detecting exoplanet transits in NASA TESS and Kepler light curves.
          </p>
          <p style="font-family:'Inter';font-size:0.95rem;line-height:1.8;color:rgba(240,238,232,0.55)">
            It chains together catalog refresh → validation gates → on-demand GPU training → live calibrated probability scoring → interactive web-based vetting console.
          </p>
        </div>
        <div>
          <div class="section-label" style="margin-bottom:1.5rem">Pipeline Architecture</div>
          ${PIPELINE.map((item, i) => `
            <div style="display:flex;gap:1.5rem;margin-bottom:2rem;padding-bottom:2rem;border-bottom:${i < 3 ? '1px solid rgba(255,255,255,0.06)' : 'none'}">
              <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#4DFFD2;padding-top:2px;min-width:20px">${item.step}</div>
              <div>
                <div style="font-family:'Space Grotesk';font-size:0.85rem;font-weight:600;color:#F0EEE8;margin-bottom:0.4rem;letter-spacing:0.02em">${item.title}</div>
                <div style="font-family:'Inter';font-size:0.8rem;line-height:1.7;color:rgba(240,238,232,0.5)">${esc(item.desc)}</div>
              </div>
            </div>`).join('')}
        </div>
      </div>
    </section>

    <div class="hairline"></div>

    <section class="page-pad" style="padding:8rem 3rem 10rem;background:#050608;position:relative;z-index:1;text-align:center">
      <div style="max-width:600px;margin:0 auto">
        <div class="section-label" style="margin-bottom:1.5rem">Begin Analysis</div>
        <h2 style="font-family:'Space Grotesk', sans-serif;font-size:clamp(2rem, 4vw, 2.5rem);font-weight:600;line-height:1.2;letter-spacing:-0.02em;color:#F0EEE8;margin-bottom:1rem">
          Submit a light curve.<br>Get a calibrated answer.
        </h2>
        <p style="font-family:'Inter';font-size:0.9rem;line-height:1.7;color:rgba(240,238,232,0.5);margin-bottom:2.5rem">
          Name a TIC or KIC target. The pipeline pulls its photometry from MAST and scores it against the promoted model.
        </p>
        <button class="btn-teal" data-nav="#/upload" style="font-size:0.8rem;padding:1rem 2.5rem">Score a Target →</button>
      </div>
    </section>
  </div>`;

  heroBackdrop();
  mountOrbitalDiagram(460);
  mountHealth();
  bindNavButtons();

  const reveal = () => {
    app.querySelectorAll('.rv').forEach(el => {
      el.style.opacity = el.dataset.visibleOpacity || '1';
      el.style.transform = 'translateY(0)';
    });
    app.querySelectorAll('[data-count]').forEach(el => countUp(el));
  };
  bootReady.then(() => setTimeout(reveal, 100));
}

function countUp(el) {
  const target = +el.dataset.count;
  // rAF does not run in a hidden document, so a backgrounded tab would sit on "0"
  if (document.hidden) { el.textContent = target.toLocaleString(); return; }
  const obj = { v: 0 };
  animate(obj, {
    v: target, duration: +el.dataset.dur || 2000, delay: 600, ease: 'out(3)',
    onUpdate: () => { el.textContent = Math.floor(obj.v).toLocaleString(); },
    onComplete: () => { el.textContent = target.toLocaleString(); },
  });
}

/* Stand-in for the hero photograph (hero-space-bg_8e38c7b4.jpg, which was not
   in the design hand-off). A plain star field on the void, rendered at the same
   0.35 opacity and sitting under the same two gradient overlays as the
   original. No nebula. */
function heroBackdrop() {
  const c = document.getElementById('hero-bg');
  if (!c) return;
  const paint = () => {
    const w = c.width = c.offsetWidth, h = c.height = c.offsetHeight;
    if (!w || !h) return;
    const ctx = c.getContext('2d');
    let seed = 20260719;
    const rnd = () => (seed = (seed * 1664525 + 1013904223) % 4294967296) / 4294967296;

    ctx.fillStyle = '#050608';
    ctx.fillRect(0, 0, w, h);

    for (let i = 0; i < 900; i++) {
      const x = rnd() * w, y = rnd() * h, s = rnd() * 1.3 + 0.2, o = rnd() * 0.55 + 0.05;
      ctx.fillStyle = `rgba(240,238,232,${o})`;
      ctx.beginPath(); ctx.arc(x, y, s, 0, Math.PI * 2); ctx.fill();
    }
  };
  paint();
  const ro = new ResizeObserver(paint); ro.observe(c); chartObservers.push(ro);
}

/* ── CATALOGUE ───────────────────────────────────────────── */
const DISPOSITIONS = ['All', 'PC', 'CP', 'KP', 'FP'];
const SOURCES = ['All', 'TESS', 'Kepler'];

function Catalogue() {
  const state = { search:'', disposition:'All', source:'All', sortKey:'prob', sortDir:'desc', minProb:0 };

  const filtered = () => {
    let data = [...CANDIDATES];
    if (state.search) {
      const q = state.search.toLowerCase();
      data = data.filter(c => c.id.toLowerCase().includes(q) || c.ticId.toLowerCase().includes(q));
    }
    if (state.disposition !== 'All') data = data.filter(c => c.disposition === state.disposition);
    if (state.source !== 'All') data = data.filter(c => c.source === state.source);
    if (state.minProb > 0) data = data.filter(c => c.prob != null && c.prob >= state.minProb / 100);
    data.sort((a, b) => {
      const av = a[state.sortKey], bv = b[state.sortKey];
      // Nulls sink to the bottom in either direction: an unscored row is not
      // the lowest-scoring row, and sorting it there would read as one.
      const an = av == null || (typeof av === 'number' && !Number.isFinite(av));
      const bn = bv == null || (typeof bv === 'number' && !Number.isFinite(bv));
      if (an && bn) return 0;
      if (an) return 1;
      if (bn) return -1;
      if (typeof av === 'number' && typeof bv === 'number') return state.sortDir === 'asc' ? av - bv : bv - av;
      return state.sortDir === 'asc' ? String(av).localeCompare(String(bv)) : String(bv).localeCompare(String(av));
    });
    return data;
  };

  const COLS = [
    ['Candidate ID','id'], ['TIC / KIC','ticId'], ['Period (d)','period'], ['Depth','depth'],
    ['P(planet)','prob'], ['σ','score_std'], ['Disp.','disposition'], ['Source','source'],
    ['T-mag','tmag'], ['SNR','snr'], ['TSM','tsm'], ['ESM','esm'],
    ['Baseline (d)','baselineDays'], ['Last Scored','lastScored'],
  ];

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0;position:relative">
      <div class="orbital-motif" style="top:100px;right:-100px;opacity:0.6"></div>

      <div style="display:flex;justify-content:space-between;align-items:flex-end;gap:1rem;flex-wrap:wrap;margin-bottom:2.5rem">
        <div>
          <div class="section-label" style="margin-bottom:0.75rem">Candidate Catalogue</div>
          <h1 id="cat-count" style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem"></h1>
          <h2 style="font-family:'Space Grotesk';font-size:clamp(1.2rem, 2.5vw, 2rem);font-weight:300;letter-spacing:-0.02em;color:rgba(240,238,232,0.25);line-height:1.0;margin-bottom:0.5rem">UNDER INVESTIGATION</h2>
          <p id="cat-sub" style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8"></p>
        </div>
        <button class="btn-ghost" id="cat-export" style="display:flex;align-items:center;gap:0.5rem">
          <svg width="12" height="12" viewBox="0 0 12 12" fill="none"><path d="M6 1v7M3 5l3 3 3-3M1 10h10" stroke="currentColor" stroke-width="1.2" stroke-linecap="round"/></svg>
          Export CSV
        </button>
      </div>

      <div style="display:flex;gap:1rem;align-items:center;margin-bottom:1.25rem;flex-wrap:wrap">
        <div style="position:relative;flex:1;min-width:200px;max-width:320px">
          <svg style="position:absolute;left:0.75rem;top:50%;transform:translateY(-50%)" width="12" height="12" viewBox="0 0 12 12" fill="none">
            <circle cx="5" cy="5" r="4" stroke="#8A8FA8" stroke-width="1.2"/><path d="M8 8l2.5 2.5" stroke="#8A8FA8" stroke-width="1.2" stroke-linecap="round"/>
          </svg>
          <input type="text" id="cat-search" placeholder="Search by ID or TIC..." aria-label="Search candidates"
            style="width:100%;background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.1);color:#F0EEE8;font-family:'JetBrains Mono';font-size:0.75rem;padding:0.6rem 0.75rem 0.6rem 2rem">
        </div>
        <div style="display:flex;gap:0.4rem" id="cat-disp">
          ${DISPOSITIONS.map(d => `<button data-d="${d}" style="font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:0.4rem 0.75rem;transition:all 150ms ease">${d}</button>`).join('')}
        </div>
        <div style="display:flex;gap:0.4rem" id="cat-src">
          ${SOURCES.map(s => `<button data-s="${s}" style="font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;padding:0.4rem 0.75rem;transition:all 150ms ease">${s}</button>`).join('')}
        </div>
        <div style="display:flex;align-items:center;gap:0.75rem">
          <span class="stat-label">Min P(planet)</span>
          <input type="range" id="cat-range" min="0" max="90" step="5" value="0" aria-label="Minimum probability" style="width:80px">
          <span id="cat-range-v" style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2;min-width:3rem">0.00</span>
        </div>
      </div>

      <div class="note" style="margin-bottom:1.5rem">
        <span class="ico">▸</span>
        <span class="txt">
          <b style="color:rgba(240,238,232,0.8);font-weight:500">TSM / ESM</b> are Kempton (2018) follow-up metrics — sort on them to rank targets for telescope time rather than by score alone.
          <b style="color:rgba(240,238,232,0.8);font-weight:500">Baseline</b> is the observed span for the host; long baselines inflate detectability, so scores are not comparable across very different baselines.
          Per-row score σ is pending — <span style="font-family:'JetBrains Mono';font-size:0.68rem">score_std</span> is not yet persisted to the catalogue table.
        </span>
      </div>

      <div style="overflow-x:auto;border:1px solid rgba(255,255,255,0.08)">
        <table style="width:100%;border-collapse:collapse;min-width:1180px">
          <thead><tr id="cat-head" style="background:rgba(255,255,255,0.02)"></tr></thead>
          <tbody id="cat-body"></tbody>
        </table>
      </div>
      <div id="cat-empty"></div>
    </div>
  </div>`;

  const headEl = document.getElementById('cat-head');
  const bodyEl = document.getElementById('cat-body');

  const paint = () => {
    const data = filtered();
    document.getElementById('cat-count').textContent = `${data.length} WORLDS`;
    document.getElementById('cat-sub').textContent = `${CANDIDATES.length} total · sorted by ${state.sortKey} ${state.sortDir}`;

    headEl.innerHTML = COLS.map(([label, col]) => {
      const sortable = col !== 'score_std';
      const on = state.sortKey === col;
      const icon = !sortable ? '' : on ? (state.sortDir === 'asc' ? '↑' : '↓') : '↕';
      return `<th ${sortable ? `data-col="${col}"` : ''} style="padding:0.75rem 1rem;text-align:left;font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.14em;text-transform:uppercase;color:${on ? '#4DFFD2' : '#8A8FA8'};cursor:${sortable ? 'pointer' : 'default'};white-space:nowrap;user-select:none;border-bottom:1px solid rgba(255,255,255,0.08)">${label}<span style="margin-left:4px;color:${on ? '#4DFFD2' : 'rgba(138,143,168,0.4)'};font-size:0.6rem">${icon}</span></th>`;
    }).join('') + `<th style="padding:0.75rem 1rem;border-bottom:1px solid rgba(255,255,255,0.08)"></th>`;

    bodyEl.innerHTML = data.map(c => {
      const dc = getDispositionColor(c.disposition);
      const fu = followUp(c);
      const td = 'padding:0.85rem 1rem';
      const mono = "font-family:'JetBrains Mono';font-size:0.75rem;color:#F0EEE8;font-variant-numeric:tabular-nums";
      const dim = "font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8;font-variant-numeric:tabular-nums";
      const n = (v, d) => (v == null || !Number.isFinite(v) || v === 0 ? '—' : v.toFixed(d));
      return `<tr class="data-row" data-id="${esc(c.id)}" style="cursor:pointer">
        <td style="${td}"><span style="${mono};font-weight:500">${esc(c.id)}</span></td>
        <td style="${td}"><span style="${dim}">${esc(c.ticId)}</span></td>
        <td style="${td}"><span style="${mono}">${n(c.period, 1)}</span></td>
        <td style="${td}"><span style="${mono}">${n(c.depth, 4)}</span></td>
        <td style="${td}">${c.prob == null
            ? `<span style="${dim};color:rgba(138,143,168,0.5)" title="/candidates carries no score — open to vet">not scored</span>`
            : `<span class="${getProbClass(c.prob)}">${c.prob.toFixed(3)}</span>`}</td>
        <td style="${td}"><span style="${dim}${c.probStd == null ? ';color:rgba(138,143,168,0.5)' : ''}" title="${c.probStd == null ? 'not scored' : 'ensemble spread over the five folds'}">${c.probStd == null ? '—' : c.probStd.toFixed(3)}</span></td>
        <td style="${td}"><span style="font-family:'JetBrains Mono';font-size:0.65rem;font-weight:600;color:${dc};background:${dc}18;border:1px solid ${dc}44;padding:0.15rem 0.5rem;border-radius:2px">${c.disposition}</span></td>
        <td style="${td}"><span style="font-family:'Space Grotesk';font-size:0.65rem;font-weight:600;letter-spacing:0.08em;color:#8A8FA8">${c.source}</span></td>
        <td style="${td}"><span style="${mono}">${n(c.tmag, 1)}</span></td>
        <td style="${td}"><span style="${mono}">${n(c.snr, 1)}</span></td>
        <td style="${td}"><span style="${mono};color:${fu.tsmPass ? '#4DFFD2' : '#F0EEE8'}">${n(fu.tsm, 1)}</span></td>
        <td style="${td}"><span style="${mono};color:${fu.esmPass ? '#4DFFD2' : '#F0EEE8'}">${n(fu.esm, 2)}</span></td>
        <td style="${td}"><span style="${mono};color:${c.baselineDays >= 1000 ? '#F5A623' : '#F0EEE8'}">${c.baselineDays == null ? '—' : c.baselineDays.toLocaleString()}</span></td>
        <td style="${td}"><span style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">${c.lastScored}</span></td>
        <td style="${td}"><span style="font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.1em;color:#4DFFD2;text-transform:uppercase">Vet →</span></td>
      </tr>`;
    }).join('');

    document.getElementById('cat-empty').innerHTML = data.length === 0
      ? `<div style="text-align:center;padding:4rem;color:#8A8FA8;font-family:'Inter';font-size:0.85rem">No candidates match the current filters.</div>` : '';

    headEl.querySelectorAll('th[data-col]').forEach(th => th.addEventListener('click', () => {
      const key = th.dataset.col;
      if (state.sortKey === key) state.sortDir = state.sortDir === 'asc' ? 'desc' : 'asc';
      else { state.sortKey = key; state.sortDir = 'desc'; }
      paint();
    }));
    bodyEl.querySelectorAll('tr').forEach(tr => tr.addEventListener('click', () => {
      location.hash = '#/vetting/' + encodeURIComponent(tr.dataset.id);
    }));
  };

  const paintChips = () => {
    document.querySelectorAll('#cat-disp button').forEach(b => {
      const on = b.dataset.d === state.disposition;
      b.style.border = `1px solid ${on ? 'rgba(77,255,210,0.5)' : 'rgba(255,255,255,0.1)'}`;
      b.style.background = on ? 'rgba(77,255,210,0.08)' : 'transparent';
      b.style.color = on ? '#4DFFD2' : '#8A8FA8';
    });
    document.querySelectorAll('#cat-src button').forEach(b => {
      const on = b.dataset.s === state.source;
      b.style.border = `1px solid ${on ? 'rgba(245,166,35,0.5)' : 'rgba(255,255,255,0.1)'}`;
      b.style.background = on ? 'rgba(245,166,35,0.06)' : 'transparent';
      b.style.color = on ? '#F5A623' : '#8A8FA8';
    });
  };

  document.getElementById('cat-search').addEventListener('input', e => { state.search = e.target.value; paint(); });
  document.querySelectorAll('#cat-disp button').forEach(b => b.addEventListener('click', () => { state.disposition = b.dataset.d; paintChips(); paint(); }));
  document.querySelectorAll('#cat-src button').forEach(b => b.addEventListener('click', () => { state.source = b.dataset.s; paintChips(); paint(); }));
  document.getElementById('cat-range').addEventListener('input', e => {
    state.minProb = +e.target.value;
    document.getElementById('cat-range-v').textContent = (state.minProb / 100).toFixed(2);
    paint();
  });
  document.getElementById('cat-export').addEventListener('click', () => {
    const headers = ['ID','TIC ID','Period (d)','Duration (h)','Depth','Probability','Score std','Disposition','Source','T-mag','SNR','TSM','ESM','Baseline (d)','Last Scored'];
    const rows = filtered().map(c => {
      const fu = followUp(c);
      return [c.id, c.ticId, c.period, c.duration, c.depth, c.prob, '', c.disposition, c.source, c.tmag, c.snr,
              fu.tsm.toFixed(1), fu.esm.toFixed(2), c.baselineDays, c.lastScored];
    });
    const csv = [headers, ...rows].map(r => r.join(',')).join('\n');
    const url = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    const a = document.createElement('a');
    a.href = url; a.download = 'exoplanet-hunter-candidates.csv'; a.click();
    URL.revokeObjectURL(url);
  });

  paintChips();
  paint();
}
