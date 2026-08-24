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

   CONTRACT: GET /healthz returns { status, model_loaded, model_version,
   ensemble_ready, uptime_s }. The last two arrived after this file was first
   written, so the "warming" state is now driven by the service rather than
   skipped.

   THIS PANEL IS A LIVE READER. Every figure on it comes from the last ping:
   the model version is what the service reported, the round trip is measured
   here, and the age counts up until the next one lands. It used to stop
   polling the moment it first reached "ready" and then sit on a hardcoded
   "0.14 s steady state" — a one-shot reading, presented as a live one, that
   would still have said READY with the service long gone. The only number not
   measured per ping is the prototype mode's, which says so.
*/

const HEALTH = {
  // Live whenever probeApi() reached the service. Opened from disk with no
  // API the probe fails, and the scripted clock still demonstrates the states.
  get live() { return API.mode === 'live'; },
  get endpoint() { return `${API.base}/healthz`; },
  pollMs: 1000,                // while connecting, waking or warming
  // Once ready there is nothing to watch closely, and fly.toml suspends the
  // machine when it is idle — a 1 s poll would hold it awake for as long as
  // anyone left the page open. Slow enough to be cheap, often enough to notice.
  readyPollMs: 15000,
  wakingAfterMs: 2000,         // an in-flight request older than this is a resume
  ensembleLoadS: 90,           // documented cold-start ensemble load
};

const HEALTH_STATES = {
  connecting: {
    label: 'CONNECTING',
    colour: '#8A8FA8',
    detail: () => `GET ${HEALTH.endpoint}`,
    bar: 'indeterminate',
    foot: () => 'Catalogue and reliability are parquet reads, so they answer in ~0.8 s whatever the model is doing.',
  },
  waking: {
    label: 'WAKING THE OBSERVATORY',
    colour: '#F5A623',
    detail: () => 'restoring RAM snapshot · ensemble still resident',
    bar: 'indeterminate',
    foot: () => 'The machine suspends rather than stops, so this is a ~2 s resume, not a cold start.',
  },
  warming: {
    label: 'WARMING MODEL',
    colour: '#F5A623',
    detail: () => 'loading 5-fold ensemble · TensorFlow import',
    bar: 'determinate',
    foot: () => 'Browse the catalogue meanwhile; only scoring waits on the ensemble.',
  },
  ready: {
    label: 'MODEL WARM',
    colour: '#4DFFD2',
    detail: s => s.modelVersion || SERVED.modelVersion,
    bar: 'done',
    // Measured per ping when there is a service; the prototype's steady state
    // is the documented figure and is labelled as the prototype's.
    foot: s => HEALTH.live
      ? `Scoring ready · ${Math.round(s.inflightMs)} ms round trip · ${pingFreshness()}`
      : 'Scoring ready · 0.14 s steady state (prototype).',
  },
  degraded: {
    label: 'NO PROMOTED MODEL',
    colour: '#FF4D4D',
    detail: () => 'registry.json has no promoted run',
    bar: 'none',
    foot: () => 'Scoring is unavailable. The catalogue and reliability views still work.',
  },
  // A ping that does not come back. The panel used to render this as
  // CONNECTING, which reads as "starting up" rather than "gone".
  unreachable: {
    label: 'NO ANSWER',
    colour: '#FF4D4D',
    detail: s => s.error ? String(s.error).slice(0, 64) : `no response from ${HEALTH.endpoint}`,
    bar: 'none',
    foot: () => `Last ping failed ${pingAge()} s ago. Anything already on screen was loaded before it stopped answering.`,

  },
};

/* Seconds since the last completed ping, for the readings that count up. */
let lastPingAt = 0;
const pingAge = () => (lastPingAt ? Math.max(0, Math.round((performance.now() - lastPingAt) / 1000)) : 0);

/* Pings stop while the tab is in the background, so a panel left open there
   would go on asserting READY off a reading minutes old. Past three missed
   cadences the age stops being a detail and becomes the point. */
const STALE_AFTER_S = (HEALTH.readyPollMs / 1000) * 3;
const pingFreshness = () => {
  const age = pingAge();
  return age > STALE_AFTER_S
    ? `last answer ${age} s ago · not rechecked while this tab is in the background`
    : `pinged ${age} s ago`;
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
  if (snap.failed) return 'unreachable';
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

  let lastSnap = { status: null, inflightMs: 0 };
  let inflight = false;

  const render = snap => {
    lastSnap = snap;
    const phase = phaseFor(snap);
    if (phase !== healthPhase) { healthPhase = phase; applyPhase(phase, snap); }
    // Written on every ping, not only when the state changes, because both
    // lines carry values that move while the state stays put.
    detail.textContent = HEALTH_STATES[phase].detail(snap);
    foot.textContent = HEALTH_STATES[phase].foot(snap);

    if (phase === 'ready' || phase === 'degraded') {
      try { sessionStorage.setItem(SEEN_KEY, '1'); } catch (e) {}
    }

    if (phase === 'warming') {
      const p = Math.max(0, Math.min(1, (snap.uptimeS || 0) / HEALTH.ensembleLoadS));
      const remaining = Math.max(0, Math.ceil(HEALTH.ensembleLoadS - (snap.uptimeS || 0)));
      fill.style.width = (p * 100).toFixed(1) + '%';
      etaL.textContent = `${remaining} s remaining`;
      etaR.textContent = `${Math.round((snap.uptimeS || 0))} / ${HEALTH.ensembleLoadS} s`;
    }
  };

  /* One ping, whichever source is answering. `inflight` keeps a slow reply from
     stacking requests on a suspended machine. */
  const ping = () => {
    if (!HEALTH.live) { lastPingAt = performance.now(); render(simulatedHealth(t0, forceCold)); return; }
    inflight = true;
    const sent = performance.now();
    fetchHealth()
      .then(snap => { lastPingAt = performance.now(); render(snap); })
      .catch(e => {
        lastPingAt = performance.now();
        render({ status: null, failed: true, error: e.message, inflightMs: performance.now() - sent });
      })
      .finally(() => { inflight = false; });
  };

  /* A single 1 s tick drives both jobs: it fires a ping when the current
     phase's cadence is up, and it rewrites the foot every second regardless so
     the age reading counts rather than freezing between pings. Nothing is sent
     while the tab is hidden — nobody is reading it, and the machine would be
     held awake for a page nobody is looking at. */
  const start = () => {
    clearInterval(healthTimer);
    t0 = performance.now();
    lastPingAt = 0;
    healthPhase = null;
    fill.style.width = '0%';
    ping();
    healthTimer = setInterval(() => {
      if (!document.getElementById('hstat')) { stopHealth(); return; }
      const cadence = healthPhase === 'ready' || healthPhase === 'degraded'
        ? HEALTH.readyPollMs : HEALTH.pollMs;
      if (!document.hidden && !inflight && performance.now() - lastPingAt >= cadence) ping();
      else if (healthPhase) foot.textContent = HEALTH_STATES[healthPhase].foot(lastSnap);
    }, 1000);
  };

  document.getElementById('hstat-replay').addEventListener('click', () => {
    forceCold = true;
    try { sessionStorage.removeItem(SEEN_KEY); } catch (e) {}
    start();
  });

  start();
}
