/* ═══════════════════════════════════════════════════════════
   BOOT PRELOADER — instrument dial, anime.js v4

   Structure follows the anime.js hero dial: a glowing segmented
   outer ring, a dense tick ring that keeps turning, a glass disc
   with a specular highlight, thin concentric arcs drawn on with
   a stagger, and a particle field animated with blend
   composition. Palette and motif stay ours — the segments are
   the five folds plus the calibration and gate stages, and the
   particles are candidates being scored.
   ═══════════════════════════════════════════════════════════ */

let resolveBoot;
const bootReady = new Promise(r => { resolveBoot = r; });

(function boot() {
  const root = document.getElementById('boot');
  const statusEl = document.getElementById('boot-status');
  const stageEl = document.getElementById('boot-stage-label');
  const pctEl = document.getElementById('boot-pct');
  const fillEl = document.getElementById('boot-fill');

  let done = false;
  const finish = () => { if (done) return; done = true; root.remove(); resolveBoot(); };

  setTimeout(finish, 9000);              // the console must never stay behind the overlay
  if (REDUCED) { finish(); return; }

  const C = 180;                          // centre of the 360×360 viewBox

  /* ── outer ring: eight glowing segments ─────────────────── */
  const SEG_COLOURS = ['#4DFFD2', '#4DFFD2', '#F5A623', '#4DFFD2', '#FF4D4D', '#8A8FA8', '#4DFFD2', '#F5A623'];
  const SEG_SPAN = 37, SEG_GAP = 8;
  document.getElementById('boot-segs').innerHTML = SEG_COLOURS.map((col, i) => {
    const a0 = i * (SEG_SPAN + SEG_GAP);
    return `<path class="boot-seg" d="${arcPath(C, C, 166, a0, a0 + SEG_SPAN)}" stroke="${col}"/>`;
  }).join('');

  /* ── tick ring ──────────────────────────────────────────── */
  const TICKS = 96;
  document.getElementById('boot-ticks').innerHTML = Array.from({ length: TICKS }, (_, i) => {
    const deg = (i / TICKS) * 360;
    const major = i % 8 === 0;
    const [x1, y1] = polar(C, C, 152, deg);
    const [x2, y2] = polar(C, C, major ? 138 : 144, deg);
    return `<line class="boot-tick ${major ? 'tick-major' : 'tick-minor'}"
      x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"
      opacity="0" stroke="${major ? '#4DFFD2' : '#8A8FA8'}"/>`;
  }).join('');

  /* ── thin concentric arcs, staggered draw ───────────────── */
  const ARC_SPECS = [
    { r: 136, a0: 200, a1: 320, col: 'rgba(240,238,232,0.55)', w: 1.5 },
    { r: 128, a0: 214, a1: 306, col: 'rgba(240,238,232,0.38)', w: 1.5 },
    { r: 120, a0: 228, a1: 292, col: 'rgba(240,238,232,0.26)', w: 1.5 },
    { r: 112, a0: 242, a1: 278, col: 'rgba(77,255,210,0.40)',  w: 1.5 },
    { r: 104, a0: 40,  a1: 96,  col: 'rgba(245,166,35,0.35)',  w: 1.5 },
  ];
  document.getElementById('boot-arcs').innerHTML = ARC_SPECS.map(s =>
    `<path class="boot-arc" d="${arcPath(C, C, s.r, s.a0, s.a1)}" stroke="${s.col}" stroke-width="${s.w}"/>`
  ).join('');

  /* ── candidate field ────────────────────────────────────── */
  const GRID = 7, SPACING = 26;
  const fieldRng = rngFor('boot|field');
  document.getElementById('boot-field').innerHTML = Array.from({ length: GRID * GRID }, (_, i) => {
    const gx = i % GRID, gy = Math.floor(i / GRID);
    const x = C + (gx - (GRID - 1) / 2) * SPACING;
    const y = C + (gy - (GRID - 1) / 2) * SPACING;
    const roll = fieldRng();
    const col = roll > 0.86 ? '#F5A623' : roll > 0.94 ? '#FF4D4D' : '#4DFFD2';
    const r = 1.4 + fieldRng() * 1.8;
    return `<circle class="boot-particle" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(2)}"
      fill="${col}" opacity="0"/>`;
  }).join('');

  /* ── starting state ─────────────────────────────────────── */
  utils.set('.boot-tick', { opacity: 0 });
  utils.set('.boot-particle', { opacity: 0 });
  utils.set('#boot-glass', { scale: 0.7, opacity: 0 });
  utils.set('#boot-gloss', { opacity: 0, rotate: -35 });
  utils.set('#boot-star', { scale: 0, opacity: 0 });
  utils.set('#boot-halo', { scale: 0, opacity: 0 });
  utils.set('#boot-planet', { opacity: 0 });
  utils.set(['#boot-status', '#boot-sub', '.boot-meter', '.boot-corner'], { opacity: 0 });

  const segs = svg.createDrawable('.boot-seg');
  const arcs = svg.createDrawable('.boot-arc');
  const trace = svg.createDrawable('#boot-transit');
  const orbit = svg.createMotionPath('#boot-transit');

  const STAGES = [
    ['INITIALISING',       'Cold start'],
    ['CALIBRATING OPTICS', 'Detrending photometry'],
    ['ACQUIRING TARGETS',  'MAST · ExoFOP sync'],
    ['SCORING CANDIDATES', `${SERVED.runId} · 11-branch CNN`],
    ['CONSOLE READY',      'Systems nominal'],
  ];
  let stageIndex = -1;
  const setStage = i => {
    if (i === stageIndex) return;
    stageIndex = i;
    stageEl.textContent = STAGES[i][1];
    animate(statusEl, {
      innerHTML: text.scrambleText({
        text: STAGES[i][0], chars: 'A-Z', from: 'left',
        revealRate: 90, settleDuration: 160, duration: 520,
      }),
    });
  };

  /* the tick ring keeps turning for as long as the overlay is up */
  animate('#boot-ticks', { rotate: 360, duration: 48000, ease: 'linear', loop: true });

  createTimeline({
    defaults: { ease: 'inOut(3)' },
    onUpdate: self => {
      const p = self.progress;
      pctEl.textContent = String(Math.round(p * 100)).padStart(3, '0');
      fillEl.style.width = (p * 100).toFixed(2) + '%';
      setStage(Math.min(STAGES.length - 1, Math.floor(p * STAGES.length)));
    },
    onComplete: finish,
  })
    // chrome
    .add('.boot-corner', { opacity: [0, 1], duration: 500, delay: stagger(70) }, 0)
    .add('#boot-status', { opacity: [0, 1], duration: 400 }, 120)
    .add('#boot-sub',    { opacity: [0, 1], y: [6, 0], duration: 500 }, 200)
    .add('.boot-meter',  { opacity: [0, 1], y: [8, 0], duration: 500 }, 260)

    // outer ring draws itself on, segment by segment
    .add(segs, { draw: ['0 0', '0 1'], duration: 900, delay: stagger(55), ease: 'out(3)' }, 120)

    // tick ring lights up
    .add('.tick-minor', { opacity: [0, 0.30], duration: 380, delay: stagger(7), ease: 'outQuad' }, 420)
    .add('.tick-major', { opacity: [0, 0.85], duration: 420, delay: stagger(44), ease: 'outQuad' }, 420)

    // glass disc and its specular highlight
    .add('#boot-glass', { scale: [0.7, 1], opacity: [0, 1], duration: 900, ease: 'out(4)' }, 500)
    .add('#boot-gloss', { opacity: [0, 1], rotate: [-35, 8], duration: 1400, ease: 'out(3)' }, 620)

    // thin arcs sweep in behind the instrument
    .add(arcs, { draw: ['0 0', '0 1', '1 1'], duration: 1600, delay: stagger(90), ease: 'inOut(3)' }, 760)

    // the candidate field arrives from the centre outwards
    .add('.boot-particle', {
      opacity: [0, t => 0.35 + Number(t.getAttribute('r')) * 0.22],
      scale: [0, 1], duration: 620,
      delay: stagger(14, { grid: [GRID, GRID], from: 'center' }), ease: 'out(3)',
    }, 900)

    // star ignites
    .add('#boot-halo', { scale: [0, 1], opacity: [0, 1], duration: 900, ease: 'out(3)' }, 1200)
    .add('#boot-star', { scale: [0, 1], opacity: [0, 1], duration: 700, ease: 'outBack(2.2)' }, 1250)

    // a planet transits, tracing the light-curve dip beneath it
    .add('#boot-planet', { opacity: [0, 1], duration: 300 }, 1500)
    .add('#boot-planet', { x: orbit.translateX, y: orbit.translateY, duration: 1700, ease: 'inOutSine' }, 1500)
    .add(trace,          { draw: ['0 0', '0 1'], duration: 1700, ease: 'inOutSine' }, 1500)

    // lock
    .add('#boot-star', { scale: [1, 1.35, 1], duration: 620, ease: 'inOut(2)' }, 3150)
    .add('.boot-seg',  { opacity: [1, 0.45], duration: 500 }, 3250)

    // hand over
    .add('.boot-stage',  { scale: 1.07, opacity: 0, duration: 640, ease: 'in(3)' }, 3480)
    .add('.boot-readout, .boot-meter, .boot-corner', { opacity: 0, duration: 420 }, 3560)
    .add(root,           { opacity: 0, duration: 520, ease: 'outQuad' }, 3680);

  /* Blend composition: every particle keeps drifting on its own clock, and the
     drifts add rather than replace, so the field never snaps between states. */
  setTimeout(() => {
    if (done) return;
    animate('.boot-particle', {
      x: () => utils.random(-70, 70),
      y: () => utils.random(-70, 70),
      rotate: () => utils.random(-180, 180),
      scale: () => utils.random(0.5, 1.7),
      duration: () => utils.random(600, 1300),
      composition: 'blend',
      ease: 'inOut(2)',
      loop: true,
      alternate: true,
      delay: stagger(14, { grid: [GRID, GRID], from: 'center' }),
    });
  }, 1500);
})();

/* ── go ──────────────────────────────────────────────────── */
// replaceState, not `location.hash = …`, so the initial route renders once
if (!location.hash) history.replaceState(null, '', '#/');

/* hydrate() fills SERVED and CANDIDATES from the API before the first render,
   which keeps every page function synchronous. It never rejects — an
   unreachable service resolves to mock mode — so route() is not guarded. The
   boot overlay covers this; on a warm service it is ~0.3 s, and the overlay
   runs 4 s regardless. */
hydrate().then(({ mode, notes }) => {
  API.notes = notes;
  if (mode !== 'live') console.info('[eh] no API reachable — prototype data');
  else console.info(`[eh] live against ${API.base}`, notes);
  mountTicker();
  route();
});
