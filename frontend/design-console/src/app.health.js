/* ═══════════════════════════════════════════════════════════
   SERVICE STATUS — the /healthz state machine
   ═══════════════════════════════════════════════════════════

   Measured against the live service:
     resume from suspend  2.0 s   RAM snapshot restored, ensemble still resident
     warm                 0.14 s  steady state
     /candidates          0.78 s  parquet read
     /reliability         0.87 s  parquet read

   fly.toml sets auto_stop_machines = "suspend", so the common idle path is a
   ~2 s resume with the model already in memory — not a cold start. A true cold
   start (stopped machine, or post-deploy) runs `dvc pull` from R2 then imports
   TensorFlow and five folds: 60–180 s, and the Fly health check allows 180 s.

   Three rules this panel follows:
     1. Browsing is never gated. /candidates and /reliability are parquet reads
        that answer in ~0.8 s whatever the ensemble is doing. Only Score waits.
     2. The served model version is always named once known, so the console
        cannot quietly drift from what is actually deployed.
     3. Determinate bar only for the ~90 s ensemble load. A bar that completes
        in 2 s reads as broken; a spinner that runs for 90 s reads as hung.

   NOTE ON THE CONTRACT: the deployed GET /healthz currently returns only
   { status, model_loaded, model_version }. The "warming" state needs two more
   fields — ensemble_ready (bool) and uptime_s (number). Until they exist this
   client degrades to connecting → waking → ready, which is correct behaviour
   for the suspend-resume path but skips the cold-start progress entirely.
*/

const HEALTH = {
  live: false,                 // flip to true to poll the real endpoint
  endpoint: '/healthz',
  pollMs: 1000,
  wakingAfterMs: 2000,         // an in-flight request older than this is a resume
  ensembleLoadS: 90,           // documented cold-start ensemble load
};

const HEALTH_STATES = {
  connecting: {
    label: 'CONNECTING',
    colour: '#8A8FA8',
    detail: () => `GET ${HEALTH.endpoint}`,
    bar: 'indeterminate',
    foot: 'Catalogue and reliability are parquet reads — they answer in ~0.8 s whatever the model is doing.',
  },
  waking: {
    label: 'WAKING THE OBSERVATORY',
    colour: '#F5A623',
    detail: () => 'restoring RAM snapshot · ensemble still resident',
    bar: 'indeterminate',
    foot: 'The machine suspends rather than stops, so this is a ~2 s resume, not a cold start.',
  },
  warming: {
    label: 'WARMING MODEL',
    colour: '#F5A623',
    detail: () => 'loading 5-fold ensemble · TensorFlow import',
    bar: 'determinate',
    foot: 'Browse the catalogue meanwhile — only scoring waits on the ensemble.',
  },
  ready: {
    label: 'MODEL WARM',
    colour: '#4DFFD2',
    detail: s => s.modelVersion || SERVED.modelVersion,
    bar: 'done',
    foot: 'Scoring ready · 0.14 s steady state.',
  },
  degraded: {
    label: 'NO PROMOTED MODEL',
    colour: '#FF4D4D',
    detail: () => 'registry.json has no promoted run',
    bar: 'none',
    foot: 'Scoring is unavailable. The catalogue and reliability views still work.',
  },
};

let healthTimer = null;
let healthAnims = [];
let healthPhase = null;

function stopHealth() {
  clearInterval(healthTimer);
  healthTimer = null;
  healthAnims.forEach(a => { try { a.revert(); } catch (e) { /* already gone */ } });
  healthAnims = [];
  healthPhase = null;
}

/* The real call, kept next to the simulation so swapping is a one-line change. */
async function fetchHealth() {
  const started = performance.now();
  const res = await fetch(HEALTH.endpoint, { cache: 'no-store' });
  const body = await res.json();
  return {
    inflightMs: performance.now() - started,
    status: body.status,
    modelVersion: body.model_version,
    // absent on the current contract — see the note at the top of this file
    ensembleReady: body.ensemble_ready,
    uptimeS: body.uptime_s,
  };
}

/* Prototype clock: plays one honest cold start, then behaves like a warm
   service on subsequent loads, which is what the real thing does. */
function simulatedHealth(t0, forceCold) {
  const elapsed = (performance.now() - t0) / 1000;
  if (!forceCold) return { status:'ok', ensembleReady:true, modelVersion:SERVED.modelVersion, inflightMs:140 };
  if (elapsed < 0.9)  return { status:null, inflightMs: elapsed * 1000 };
  if (elapsed < 3.0)  return { status:null, inflightMs: elapsed * 1000 };
  if (elapsed < 15.0) {
    // join the cold start late: uptime runs 78 → 90 s in real time, so the
    // countdown shown to the user is truthful rather than compressed
    return { status:'ok', ensembleReady:false, uptimeS: 78 + (elapsed - 3.0), modelVersion:SERVED.modelVersion, inflightMs:120 };
  }
  return { status:'ok', ensembleReady:true, modelVersion:SERVED.modelVersion, inflightMs:140 };
}

function phaseFor(snap) {
  if (snap.status === null || snap.status === undefined) {
    return snap.inflightMs > HEALTH.wakingAfterMs ? 'waking' : 'connecting';
  }
  if (snap.status === 'degraded') return 'degraded';
  // ensemble_ready absent → the contract predates it; treat a 200 as ready
  if (snap.ensembleReady === false) return 'warming';
  return 'ready';
}

function healthPanelHTML() {
  return `
    <aside id="hstat" aria-live="polite">
      <div class="row">
        <svg class="ring" id="hstat-ring" viewBox="0 0 24 24" aria-hidden="true">
          <circle class="track" cx="12" cy="12" r="9" stroke="rgba(255,255,255,0.14)"/>
          <circle class="arc" id="hstat-arc" cx="12" cy="12" r="9" stroke="#8A8FA8" stroke-linecap="round" stroke-dasharray="17 60"/>
        </svg>
        <div class="live-dot" id="hstat-dot" style="display:none"></div>
        <span class="label" id="hstat-label">CONNECTING</span>
        <button class="replay" id="hstat-replay" title="Replay the cold-start states" aria-label="Replay the cold-start states">↻</button>
      </div>
      <div class="detail" id="hstat-detail">GET /healthz</div>
      <div class="rail" id="hstat-rail"><div class="sliver" id="hstat-sliver"></div><div class="fill" id="hstat-fill"></div></div>
      <div class="eta" id="hstat-eta" style="display:none"><span id="hstat-eta-left"></span><span id="hstat-eta-right"></span></div>
      <div class="foot" id="hstat-foot"></div>
    </aside>`;
}

function mountHealth() {
  const root = document.getElementById('hstat');
  if (!root) return;

  const ring   = document.getElementById('hstat-ring');
  const arc    = document.getElementById('hstat-arc');
  const dot    = document.getElementById('hstat-dot');
  const label  = document.getElementById('hstat-label');
  const detail = document.getElementById('hstat-detail');
  const rail   = document.getElementById('hstat-rail');
  const sliver = document.getElementById('hstat-sliver');
  const fill   = document.getElementById('hstat-fill');
  const eta    = document.getElementById('hstat-eta');
  const etaL   = document.getElementById('hstat-eta-left');
  const etaR   = document.getElementById('hstat-eta-right');
  const foot   = document.getElementById('hstat-foot');

  const SEEN_KEY = 'eh.coldstart.seen';
  let seen = false;
  try { seen = sessionStorage.getItem(SEEN_KEY) === '1'; } catch (e) { /* private mode */ }

  let t0 = performance.now();
  let forceCold = !seen;

  const spinner = () => {
    healthAnims.push(animate(arc, { rotate: 360, duration: 900, ease: 'linear', loop: true }));
  };
  const travel = () => {
    healthAnims.push(animate(sliver, { x: ['-110%', '310%'], duration: 1250, ease: 'inOut(2)', loop: true }));
  };

  const applyPhase = (phase, snap) => {
    const spec = HEALTH_STATES[phase];
    healthAnims.forEach(a => { try { a.revert(); } catch (e) {} });
    healthAnims = [];

    label.style.color = spec.colour;
    arc.setAttribute('stroke', spec.colour);
    detail.textContent = spec.detail(snap);
    foot.textContent = spec.foot;

    if (REDUCED) label.textContent = spec.label;
    else animate(label, {
      innerHTML: text.scrambleText({
        text: spec.label, chars: 'A-Z', from: 'left',
        revealRate: 110, settleDuration: 130, duration: 420,
      }),
    });

    const busy = spec.bar === 'indeterminate';
    ring.style.display = phase === 'ready' || phase === 'degraded' ? 'none' : 'block';
    dot.style.display  = phase === 'ready' ? 'block' : 'none';
    if (phase === 'degraded') { dot.style.display = 'block'; dot.style.background = '#FF4D4D'; dot.style.boxShadow = '0 0 8px rgba(255,77,77,0.8)'; }

    rail.style.display   = spec.bar === 'none' ? 'none' : 'block';
    sliver.style.display = busy ? 'block' : 'none';
    fill.style.display   = busy ? 'none' : 'block';
    rail.classList.toggle('ready', phase === 'ready');
    eta.style.display    = spec.bar === 'determinate' ? 'flex' : 'none';

    if (!REDUCED && ring.style.display !== 'none') spinner();
    if (!REDUCED && busy) travel();
    if (spec.bar === 'done') healthAnims.push(animate(fill, { width: '100%', duration: 520, ease: 'out(3)' }));
  };

  const tick = () => {
    const snap = HEALTH.live ? null : simulatedHealth(t0, forceCold);
    if (!snap) return;                                  // live path resolves async
    render(snap);
  };

  const render = snap => {
    const phase = phaseFor(snap);
    if (phase !== healthPhase) { healthPhase = phase; applyPhase(phase, snap); }

    if (phase === 'ready' || phase === 'degraded') {
      try { sessionStorage.setItem(SEEN_KEY, '1'); } catch (e) {}
      clearInterval(healthTimer);
      healthTimer = null;
    }

    if (phase === 'warming') {
      const p = Math.max(0, Math.min(1, (snap.uptimeS || 0) / HEALTH.ensembleLoadS));
      const remaining = Math.max(0, Math.ceil(HEALTH.ensembleLoadS - (snap.uptimeS || 0)));
      fill.style.width = (p * 100).toFixed(1) + '%';
      etaL.textContent = `${remaining} s remaining`;
      etaR.textContent = `${Math.round((snap.uptimeS || 0))} / ${HEALTH.ensembleLoadS} s`;
    }
  };

  const start = () => {
    clearInterval(healthTimer);
    t0 = performance.now();
    healthPhase = null;
    fill.style.width = '0%';
    tick();
    healthTimer = setInterval(() => {
      if (!document.getElementById('hstat')) { stopHealth(); return; }
      if (HEALTH.live) fetchHealth().then(render).catch(() => render({ status:null, inflightMs: performance.now() - t0 }));
      else tick();
    }, HEALTH.pollMs);
  };

  document.getElementById('hstat-replay').addEventListener('click', () => {
    forceCold = true;
    try { sessionStorage.removeItem(SEEN_KEY); } catch (e) {}
    start();
  });

  start();
}
