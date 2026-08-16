/* ═══════════════════════════════════════════════════════════
   Exoplanet Hunter — Transit Detection Console (design prototype)
   Deep Space Cinematic · Electric Teal #4DFFD2

   Part 1/2 — data model, shared components, Home, Catalogue.
   ═══════════════════════════════════════════════════════════ */

const { animate, createTimeline, stagger, svg, text, utils, engine } = ANIME;

// anime.js pauses its engine while the document is hidden. Right default for
// decorative motion, but it would park the boot overlay in front of the console
// for anyone opening this in a background tab.
engine.pauseOnDocumentHidden = false;

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const signed = (v, d = 2) => {
  const r = +v.toFixed(d);                         // avoid rendering "−0.00"
  return (r >= 0 ? '+' : '−') + Math.abs(r).toFixed(d);
};

/* deterministic per-candidate randomness — evidence must not reshuffle on every render */
function rngFor(seed) {
  let h = 2166136261;
  for (let i = 0; i < seed.length; i++) { h ^= seed.charCodeAt(i); h = Math.imul(h, 16777619); }
  return () => { h ^= h << 13; h ^= h >>> 17; h ^= h << 5; h |= 0; return ((h >>> 0) % 1000000) / 1000000; };
}

/* polar helpers, shared by the boot dial */
const polar = (cx, cy, r, deg) => [
  cx + r * Math.cos((deg - 90) * Math.PI / 180),
  cy + r * Math.sin((deg - 90) * Math.PI / 180),
];
function arcPath(cx, cy, r, a0, a1) {
  const [x0, y0] = polar(cx, cy, r, a0);
  const [x1, y1] = polar(cx, cy, r, a1);
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${a1 - a0 > 180 ? 1 : 0} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

/* ── served model — registry.json ────────────────────────── */
const SERVED = {
  runId: 'ca906040',
  modelVersion: 'cnn_dualview-cv-ca906040',
  promotedAt: '2026-07-19',
  arch: '5-fold dual-view CNN ensemble · MC-dropout · Platt calibration',
  noiseFloor: { auc: 0.0070, recall: 0.0337 },
  // The panel is built from this list, not from three hardcoded columns.
  missions: [
    { mission:'TESS',   role:'gating',     evaluation:'out-of-fold', n:5156,
      auc:0.9100, aucErr:0.0070, recall:0.6120, recallErr:0.0337, brier:0.0871, brierErr:0.0042, ece:0.0130, eceErr:0.0031 },
    { mission:'Kepler', role:'diagnostic', evaluation:'out-of-fold', n:2500,
      auc:0.9914, aucErr:0.0038, recall:0.9410, recallErr:0.0180, brier:0.0212, brierErr:0.0019, ece:0.0084, eceErr:0.0022 },
    { mission:'K2',     role:'diagnostic', evaluation:'zero-shot',   n:412,
      auc:0.8480, aucErr:0.0210, recall:0.4030, recallErr:0.0610, brier:0.1420, brierErr:0.0090, ece:0.0410, eceErr:0.0070 },
  ],
};
const GATING = SERVED.missions.find(m => m.role === 'gating');

const RUNS = [
  { runId:'ca906040', date:'2026-07-19', auc:0.9100, aucErr:0.0070, recall:0.6120, brier:0.0871, status:'active', verdict:'PROMOTE',
    reason:'TESS AUC +0.0180 over champion — 2.6× the ±0.0070 noise floor. Brier and ECE not degraded.' },
  { runId:'7b1e4c23', date:'2026-07-02', auc:0.8960, aucErr:0.0071, recall:0.5980, brier:0.0903, status:'archived', verdict:'REJECT',
    reason:'ΔAUC +0.0040 falls inside the ±0.0070 noise floor — not distinguishable from the champion.' },
  { runId:'a3f2d891', date:'2026-06-18', auc:0.8930, aucErr:0.0068, recall:0.5510, brier:0.0918, status:'archived', verdict:'REJECT',
    reason:'Recall @1% FPR −0.0410 against champion, beyond the ±0.0337 shortlist floor. AUC gain does not compensate.' },
  { runId:'2d9f7a55', date:'2026-06-04', auc:0.8920, aucErr:0.0074, recall:0.5920, brier:0.0921, status:'archived', verdict:'PROMOTE',
    reason:'First run with an out-of-fold TESS evaluation; champion had none. Promoted as the new baseline.' },
];

/* ── data/candidates.ts ──────────────────────────────────── */
const CANDIDATES = [
  { id:'TOI-4328.01', ticId:'TIC 43288669',  period:703.8, duration:8.2, depth:0.0142, prob:0.989, disposition:'PC', source:'TESS',   tmag:9.4,  sectors:'14,15,21', lastScored:'2026-08-12', snr:42.1, baselineDays:1495 },
  { id:'TOI-4565.01', ticId:'TIC 94986319',  period:412.1, duration:6.8, depth:0.0089, prob:0.983, disposition:'PC', source:'TESS',   tmag:10.1, sectors:'18,25',    lastScored:'2026-08-11', snr:38.7, baselineDays:1113 },
  { id:'TOI-1843.01', ticId:'TIC 77175217',  period:0.177, duration:0.9, depth:0.0031, prob:0.976, disposition:'CP', source:'TESS',   tmag:11.8, sectors:'9,36',     lastScored:'2026-08-10', snr:29.4, baselineDays:82 },
  { id:'KOI-7016.01', ticId:'KIC 8120608',   period:267.3, duration:7.1, depth:0.0201, prob:0.971, disposition:'PC', source:'Kepler', tmag:12.2, sectors:'Q1-Q17',   lastScored:'2026-08-09', snr:55.8, baselineDays:1459 },
  { id:'TOI-2180.01', ticId:'TIC 292108806', period:260.8, duration:7.4, depth:0.0178, prob:0.964, disposition:'PC', source:'TESS',   tmag:8.7,  sectors:'32,33',    lastScored:'2026-08-08', snr:61.2, baselineDays:767 },
  { id:'TOI-3362.01', ticId:'TIC 466206508', period:18.1,  duration:3.2, depth:0.0055, prob:0.897, disposition:'PC', source:'TESS',   tmag:9.9,  sectors:'7,34',     lastScored:'2026-08-07', snr:22.3, baselineDays:82 },
  { id:'KOI-3284.01', ticId:'KIC 4138008',   period:35.2,  duration:4.1, depth:0.0067, prob:0.881, disposition:'PC', source:'Kepler', tmag:13.1, sectors:'Q1-Q17',   lastScored:'2026-08-06', snr:18.9, baselineDays:1459 },
  { id:'TOI-5205.01', ticId:'TIC 419411415', period:1.6,   duration:1.1, depth:0.0289, prob:0.863, disposition:'CP', source:'TESS',   tmag:12.4, sectors:'40,41',    lastScored:'2026-08-05', snr:34.6, baselineDays:55 },
  { id:'TOI-700.01',  ticId:'TIC 150428135', period:37.4,  duration:2.8, depth:0.0041, prob:0.841, disposition:'KP', source:'TESS',   tmag:13.1, sectors:'1,2,3',    lastScored:'2026-08-04', snr:15.2, baselineDays:137 },
  { id:'KOI-1686.01', ticId:'KIC 9941662',   period:52.5,  duration:5.2, depth:0.0093, prob:0.792, disposition:'PC', source:'Kepler', tmag:14.2, sectors:'Q1-Q17',   lastScored:'2026-08-03', snr:12.8, baselineDays:1459 },
  { id:'TOI-2285.01', ticId:'TIC 259377017', period:27.3,  duration:3.6, depth:0.0072, prob:0.744, disposition:'PC', source:'TESS',   tmag:11.5, sectors:'22,23',    lastScored:'2026-08-02', snr:16.4, baselineDays:82 },
  { id:'TOI-1231.01', ticId:'TIC 229742722', period:24.2,  duration:2.9, depth:0.0048, prob:0.721, disposition:'CP', source:'TESS',   tmag:9.1,  sectors:'8,35',     lastScored:'2026-08-01', snr:19.7, baselineDays:110 },
  { id:'KOI-2626.01', ticId:'KIC 7340288',   period:38.1,  duration:4.8, depth:0.0134, prob:0.698, disposition:'PC', source:'Kepler', tmag:13.8, sectors:'Q1-Q17',   lastScored:'2026-07-31', snr:11.3, baselineDays:1459 },
  { id:'TOI-4633.01', ticId:'TIC 280655495', period:34.0,  duration:3.4, depth:0.0061, prob:0.672, disposition:'PC', source:'TESS',   tmag:10.8, sectors:'19,26',    lastScored:'2026-07-30', snr:14.1, baselineDays:1330 },
  { id:'TOI-3235.01', ticId:'TIC 167600516', period:2.6,   duration:1.4, depth:0.0198, prob:0.641, disposition:'CP', source:'TESS',   tmag:12.9, sectors:'37,38',    lastScored:'2026-07-29', snr:28.9, baselineDays:55 },
  { id:'KOI-4878.01', ticId:'KIC 12644769',  period:449.7, duration:9.1, depth:0.0231, prob:0.589, disposition:'PC', source:'Kepler', tmag:11.7, sectors:'Q1-Q17',   lastScored:'2026-07-28', snr:9.4,  baselineDays:1459 },
  { id:'TOI-2095.01', ticId:'TIC 235678745', period:17.7,  duration:2.1, depth:0.0038, prob:0.521, disposition:'PC', source:'TESS',   tmag:10.3, sectors:'29,30',    lastScored:'2026-07-27', snr:8.7,  baselineDays:356 },
  { id:'TOI-1899.01', ticId:'TIC 172370679', period:29.1,  duration:3.1, depth:0.0082, prob:0.487, disposition:'FP', source:'TESS',   tmag:11.2, sectors:'16,23',    lastScored:'2026-07-26', snr:7.2,  baselineDays:82 },
  { id:'KOI-5236.01', ticId:'KIC 6922244',   period:21.3,  duration:2.7, depth:0.0059, prob:0.412, disposition:'FP', source:'Kepler', tmag:14.5, sectors:'Q1-Q17',   lastScored:'2026-07-25', snr:6.1,  baselineDays:1459 },
  { id:'TOI-3984.01', ticId:'TIC 301825483', period:4.5,   duration:1.8, depth:0.0144, prob:0.334, disposition:'FP', source:'TESS',   tmag:13.6, sectors:'43,44',    lastScored:'2026-07-24', snr:5.8,  baselineDays:55 },
];

const getProbClass = p => p >= 0.85 ? 'prob-high' : p >= 0.5 ? 'prob-med' : 'prob-low';
const getDispositionColor = d => d === 'CP' || d === 'KP' ? '#4DFFD2' : d === 'PC' ? '#F5A623' : d === 'FP' ? '#FF4D4D' : '#8A8FA8';
const probColor = p => p >= 0.85 ? '#4DFFD2' : p >= 0.5 ? '#F5A623' : '#FF4D4D';

/* ═══════════════════════════════════════════════════════════
   BRANCH EVIDENCE — the eleven input views
   Contributions are attributions on the probability scale: they
   sum from the model's mean output to the calibrated score.
   ═══════════════════════════════════════════════════════════ */
const BASE_RATE = Math.round((CANDIDATES.reduce((s, c) => s + c.prob, 0) / CANDIDATES.length) * 1000) / 1000;

const BRANCHES = [
  { key:'global_view', sees:'The whole phase-folded light curve', w:1.00,
    pos:'Folded depth and shape hold across the full period',
    neg:'Folded signal is not coherent across the period',
    neu:'Fold is consistent but adds little beyond the other views' },
  { key:'local_view', sees:'Zoomed on the transit itself', w:0.95,
    pos:'Ingress and egress are sharp and the floor is flat',
    neg:'Profile is V-shaped rather than flat-bottomed — grazing or binary',
    neu:'Transit shape is unremarkable at this SNR' },
  { key:'odd_view', sees:'Transits at odd epochs', w:0.72,
    pos:'Odd-epoch depth matches the global fold',
    neg:'Odd-epoch depth runs deeper than the fold',
    neu:'Too few odd-epoch transits to constrain the depth' },
  { key:'even_view', sees:'Transits at even epochs — a depth mismatch means an eclipsing binary', w:0.72,
    pos:'Odd and even depths agree — consistent with a planet',
    neg:'Even depth disagrees with odd — eclipsing-binary signature',
    neu:'Odd/even comparison is inconclusive at this depth' },
  { key:'secondary_view', sees:'Phase 0.5, where a secondary eclipse would sit', w:0.80,
    pos:'No secondary eclipse at phase 0.5',
    neg:'Secondary eclipse detected — a self-luminous companion',
    neu:'Phase 0.5 coverage is too sparse to rule an eclipse in or out' },
  { key:'centroid_view', sees:'Whether the light centroid moves during transit', w:0.78,
    pos:'Centroid holds still through transit — the flux is on target',
    neg:'Centroid shifts in transit — flux from a background binary',
    neu:'Centroid constraint is weak for a star this faint' },
  { key:'trend_view', sees:'Long-term stellar variability', w:0.48,
    pos:'Host is photometrically quiet; the dip is not stellar',
    neg:'Stellar variability competes with the transit signal',
    neu:'Mild variability, comfortably separated from the transit timescale' },
  { key:'unfolded_view', sees:'Twenty individual transits, unstacked', w:0.62,
    pos:'Individual events repeat at a consistent depth',
    neg:'Individual events vary too much to share one body',
    neu:'Individual events are consistent but low signal-to-noise' },
  { key:'gap_view', sees:'Where data is missing', w:0.34,
    pos:'Transits do not coincide with coverage gaps',
    neg:'Signal leans on epochs adjacent to data gaps',
    neu:'Coverage is patchy but not aligned with the transits' },
  { key:'periodogram_view', sees:'Periodicity across the light curve', w:0.55,
    pos:'Power concentrates at the candidate period',
    neg:'Dominant power sits away from the candidate period',
    neu:'Periodogram peak is present but not dominant' },
  { key:'periodogram_masked_view', sees:'Periodicity with the candidate signal removed', w:0.50,
    pos:'No residual periodicity once the candidate is masked',
    neg:'Residual periodicity survives masking — a second source',
    neu:'Residual power is marginal after masking' },
];

const CULPRITS = ['even_view', 'secondary_view', 'centroid_view', 'periodogram_masked_view', 'local_view'];

function branchEvidence(c) {
  const r = rngFor(c.id + '|branch');
  const target = c.prob - BASE_RATE;
  const drive = target >= 0 ? 1 : -1;

  // magnitudes first, signs second: branches that dissent from the verdict do so
  // weakly, so the headline row is never a contradiction of the score
  let raw = BRANCHES.map(b => {
    const dissents = r() < 0.20;
    const mag = b.w * (0.55 + r() * 0.7) * (dissents ? 0.22 : 1);
    return { b, v: (dissents ? -drive : drive) * mag };
  });

  // a false positive is caught by a specific view, not by a diffuse vote
  if (c.prob < 0.5) {
    const hit = raw.find(x => x.b.key === CULPRITS[Math.floor(r() * CULPRITS.length)]);
    if (hit) hit.v = -Math.abs(hit.v) * 2.6;
  }

  // scale (never shift) so the attributions sum to prob − base value
  const k = target / raw.reduce((s, x) => s + x.v, 0);
  const rows = raw.map(x => {
    const v = Math.round(x.v * k * 1000) / 1000 || 0;
    const weak = Math.abs(v) < 0.02;
    return {
      key: x.b.key, sees: x.b.sees, value: v,
      reading: weak ? x.b.neu : v >= 0 ? x.b.pos : x.b.neg,
    };
  }).sort((a, b) => Math.abs(b.value) - Math.abs(a.value));

  const residual = Math.round((target - rows.reduce((s, x) => s + x.value, 0)) * 1000) / 1000;
  rows[0].value = Math.round((rows[0].value + residual) * 1000) / 1000;
  return rows;
}

/* ═══════════════════════════════════════════════════════════
   MODEL AGREEMENT — per_fold + prob_std from ScoreResponse
   ═══════════════════════════════════════════════════════════ */
function foldAgreement(c) {
  const r = rngFor(c.id + '|folds');
  const ambiguity = 1 - Math.abs(c.prob - 0.5) * 2;
  const spread = 0.006 + ambiguity * 0.075;

  // draw offsets, then recentre so the fold mean *is* the ensemble score —
  // otherwise the mean marker floats outside its own dots
  const offsets = Array.from({ length: 5 }, () => (r() - 0.5) * 2 * spread);
  const centre = offsets.reduce((s, v) => s + v, 0) / offsets.length;
  const folds = offsets.map((o, i) => ({
    fold: i,
    score: Math.min(0.999, Math.max(0.001, Math.round((c.prob + o - centre) * 1000) / 1000)),
  }));
  const scores = folds.map(f => f.score);
  const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
  const foldStd = Math.sqrt(scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length);
  return {
    folds,
    foldStd: Math.round(foldStd * 10000) / 10000,
    probStd: Math.round((0.008 + ambiguity * 0.055) * 10000) / 10000,
    range: [Math.min(...scores), Math.max(...scores)],
  };
}

/* ═══════════════════════════════════════════════════════════
   DIAGNOSTIC FLAGS — pass / fail / NOT MEASURED
   The presence mask exists so "never measured" is never read as
   "measured zero"; the UI carries the same distinction.
   ═══════════════════════════════════════════════════════════ */
const NO_DV_REPORT = new Set(['TOI-4633.01', 'TOI-2095.01']);

const DIAGNOSTICS = [
  { key:'centroid', name:'Centroid Shift', field:'mean_sky_offset', unit:'σ', dp:2, threshold:'< 3.0 σ',
    pass:'In-transit centroid stays on the target star',
    fail:'Centroid offset in transit — flux may come from a neighbour' },
  { key:'oddeven', name:'Odd–Even Depth', field:'odd_even_depth_ratio', unit:'', dp:3, threshold:'< 0.050',
    pass:'Odd and even transit depths agree',
    fail:'Odd and even depths differ — eclipsing binary at twice the period' },
  { key:'secondary', name:'Secondary Eclipse', field:'weak_secondary_max_mes', unit:'', dp:2, threshold:'< 7.10',
    pass:'No significant secondary eclipse at phase 0.5',
    fail:'Secondary eclipse above threshold — self-luminous companion' },
  { key:'ghost', name:'Ghost Diagnostic', field:'ghost_core_statistic / ghost_halo_statistic', unit:'', dp:2, threshold:'core > halo',
    pass:'Signal is stronger in the core aperture than the halo',
    fail:'Halo statistic exceeds core — a contaminating source outside the aperture' },
  { key:'mes', name:'Detection Strength', field:'max_multiple_event_sigma', unit:'', dp:1, threshold:'≥ 7.1',
    pass:'Multiple-event statistic clears the detection threshold',
    fail:'Detection strength below threshold — marginal signal' },
  { key:'bootstrap', name:'False-Alarm Probability', field:'bootstrap_significance', unit:'', dp:0, threshold:'< 1e-6',
    pass:'Bootstrap false-alarm probability is negligible',
    fail:'False-alarm probability too high to rule out noise' },
  { key:'skyoffset', name:'Control Sky Offset', field:'control_sky_offset', unit:'σ', dp:2, threshold:'< 3.0 σ',
    pass:'Difference-image offset consistent with the target',
    fail:'Difference-image centroid falls off the target star' },
  { key:'completeness', name:'Transit Completeness', field:'transit_completeness', unit:'', dp:2, threshold:'> 0.70',
    pass:'Most predicted transits fall inside observed coverage',
    fail:'Many predicted transits fall in data gaps' },
];

function diagnosticsFor(c) {
  const r = rngFor(c.id + '|diag');
  const noReport = NO_DV_REPORT.has(c.id);
  const healthy = c.prob >= 0.5;

  return DIAGNOSTICS.map(d => {
    if (noReport) return { ...d, state:'unmeasured' };
    if ((d.key === 'ghost' || d.key === 'bootstrap') && r() < 0.18) return { ...d, state:'unmeasured' };

    const ok = healthy ? r() > 0.12 : r() > 0.55;
    let value;
    switch (d.key) {
      case 'centroid':     value = ok ? 0.2 + r() * 2.2 : 3.4 + r() * 4;                     break;
      case 'oddeven':      value = ok ? r() * 0.04 : 0.06 + r() * 0.22;                      break;
      case 'secondary':    value = ok ? 0.5 + r() * 5.5 : 7.6 + r() * 12;                    break;
      case 'ghost':        value = ok ? 1.2 + r() * 3 : 0.2 + r() * 0.7;                     break;
      case 'mes':          value = ok ? 8 + r() * 40 : 3 + r() * 3.8;                        break;
      case 'bootstrap':    value = ok ? 1e-9 * (1 + r() * 400) : 1e-5 * (1 + r() * 90);      break;
      case 'skyoffset':    value = ok ? 0.2 + r() * 2.3 : 3.3 + r() * 3.5;                   break;
      default:             value = ok ? 0.75 + r() * 0.24 : 0.25 + r() * 0.4;
    }
    return { ...d, state: ok ? 'pass' : 'fail', value };
  });
}

function diagValue(d) {
  if (d.state === 'unmeasured') return 'not measured';
  if (d.key === 'bootstrap') return d.value.toExponential(1);
  if (d.key === 'ghost') return `${d.value.toFixed(2)} / ${(d.value * 0.75).toFixed(2)}`;
  return d.value.toFixed(d.dp) + (d.unit ? ` ${d.unit}` : '');
}

/* ═══════════════════════════════════════════════════════════
   FOLLOW-UP PRIORITY — Kempton et al. (2018)
   ═══════════════════════════════════════════════════════════ */
const R_EARTH_PER_R_SUN = 109.2;
const planck = (umLambda, T) => 1 / (Math.exp(14387.77 / (umLambda * T)) - 1);

function followUp(c) {
  const rp = Math.sqrt(c.depth) * R_EARTH_PER_R_SUN;          // R⊕, assuming R* = 1 R☉
  const a = Math.pow(c.period / 365.25, 2 / 3);               // AU, assuming M* = 1 M☉
  const teq = 278 * Math.pow(a, -0.5);                        // K, Bond albedo 0.3
  const insol = 1 / (a * a);                                  // S⊕
  const mp = rp < 1.5 ? Math.pow(rp, 3.7) : 1.436 * Math.pow(rp, 1.70);   // M⊕, Chen & Kipping
  const mj = c.tmag - 0.6, mk = c.tmag - 0.9;                 // rough NIR proxies from T-mag
  const scale = rp < 1.5 ? 0.190 : rp < 2.75 ? 1.26 : rp < 4 ? 1.28 : 1.15;
  const tsm = scale * Math.pow(rp, 3) * teq / mp * Math.pow(10, -mj / 5);
  const esm = 4.29e6 * (planck(7.5, 1.10 * teq) / planck(7.5, 5772))
              * Math.pow(rp / R_EARTH_PER_R_SUN, 2) * Math.pow(10, -mk / 5);

  const tsmCut = rp < 1.5 ? 12 : rp < 2.75 ? 92 : rp < 4 ? 84 : 96;
  const hz = { inner: 0.95, outer: 1.67 };

  return { rp, a, teq, insol, tsm, esm, tsmCut, hz,
           inHz: a >= hz.inner && a <= hz.outer, tsmPass: tsm >= tsmCut, esmPass: esm >= 7.5 };
}

CANDIDATES.forEach(c => {
  const f = followUp(c);
  c.tsm = f.tsm; c.esm = f.esm; c.teqK = f.teq; c.rp = f.rp;
});

/* ── components/StarField.tsx ────────────────────────────── */
(function starField() {
  const canvas = document.getElementById('star-canvas');
  const ctx = canvas.getContext('2d');
  let stars = [];

  const generateStars = () => {
    const count = Math.floor((canvas.width * canvas.height) / 3000);
    stars = Array.from({ length: count }, () => {
      const layer = Math.random() < 0.6 ? 0 : Math.random() < 0.7 ? 1 : 2;
      return {
        x: Math.random() * canvas.width,
        y: Math.random() * canvas.height,
        size: layer === 0 ? Math.random() * 0.8 + 0.2 : layer === 1 ? Math.random() * 1.2 + 0.5 : Math.random() * 1.8 + 0.8,
        opacity: layer === 0 ? Math.random() * 0.4 + 0.1 : layer === 1 ? Math.random() * 0.5 + 0.2 : Math.random() * 0.6 + 0.3,
        twinkleSpeed: Math.random() * 0.015 + 0.005,
        twinkleOffset: Math.random() * Math.PI * 2,
        layer,
      };
    });
  };

  const resize = () => { canvas.width = window.innerWidth; canvas.height = window.innerHeight; generateStars(); };

  const draw = timestamp => {
    const t = timestamp * 0.001;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    for (const star of stars) {
      const twinkle = Math.sin(t * star.twinkleSpeed * 60 + star.twinkleOffset);
      const opacity = star.opacity * (0.7 + 0.3 * twinkle);
      const size = star.size * (0.9 + 0.1 * twinkle);
      ctx.beginPath();
      ctx.arc(star.x, star.y, size, 0, Math.PI * 2);
      if (star.layer === 2 && star.size > 1.4) {
        const g = ctx.createRadialGradient(star.x, star.y, 0, star.x, star.y, size * 3);
        g.addColorStop(0, `rgba(240, 238, 232, ${opacity})`);
        g.addColorStop(0.4, `rgba(200, 220, 255, ${opacity * 0.3})`);
        g.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = g;
        ctx.arc(star.x, star.y, size * 3, 0, Math.PI * 2);
      } else {
        ctx.fillStyle = `rgba(240, 238, 232, ${opacity})`;
      }
      ctx.fill();
    }
    requestAnimationFrame(draw);
  };

  resize();
  window.addEventListener('resize', resize);
  requestAnimationFrame(draw);
})();

/* ── components/OrbitalDiagram.tsx ───────────────────────── */
const ORBIT_PLANETS = [
  { radius: 60,  size: 3,   color:'#4DFFD2', speed:180, startAngle:45,  label:'TOI-1843.01', prob:'0.976' },
  { radius: 100, size: 4,   color:'#F5A623', speed:72,  startAngle:120, label:'TOI-4328.01', prob:'0.989' },
  { radius: 148, size: 3.5, color:'#4DFFD2', speed:36,  startAngle:200, label:'TOI-4565.01', prob:'0.983' },
  { radius: 200, size: 2.5, color:'#8A8FA8', speed:18,  startAngle:310, label:'KOI-7016.01', prob:'0.971' },
];

function orbitalDiagramHTML(size = 460) {
  const labels = ORBIT_PLANETS.map(p => {
    const a = (p.startAngle + 30) * Math.PI / 180;
    const tx = Math.cos(a) * (p.radius + 20), ty = Math.sin(a) * (p.radius + 20);
    return `<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) translate(${tx}px,${ty}px);pointer-events:none;white-space:nowrap">
      <div style="display:flex;align-items:center;gap:0.3rem">
        <div style="width:5px;height:5px;border-radius:50%;border:1px solid ${p.color};background:transparent"></div>
        <span style="font-family:'JetBrains Mono';font-size:0.6rem;color:rgba(240,238,232,0.6)">${p.label}</span>
        <span style="font-family:'JetBrains Mono';font-size:0.6rem;color:${p.color}">${p.prob}</span>
      </div></div>`;
  }).join('');
  return `<div style="position:relative;width:${size}px;height:${size}px;max-width:100%">
    <canvas id="orbit-canvas" width="${size}" height="${size}" style="display:block;max-width:100%"></canvas>${labels}</div>`;
}

function mountOrbitalDiagram(size = 460) {
  const canvas = document.getElementById('orbit-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const cx = size / 2, cy = size / 2;
  let start = null;

  const draw = timestamp => {
    if (!document.getElementById('orbit-canvas')) return;
    if (!start) start = timestamp;
    const elapsed = (timestamp - start) / 1000;
    ctx.clearRect(0, 0, size, size);

    ORBIT_PLANETS.forEach(p => {
      ctx.beginPath(); ctx.arc(cx, cy, p.radius, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.08)'; ctx.lineWidth = 0.5; ctx.stroke();
    });

    const starGrad = ctx.createRadialGradient(cx, cy, 0, cx, cy, 18);
    starGrad.addColorStop(0, 'rgba(255, 240, 200, 1)');
    starGrad.addColorStop(0.4, 'rgba(255, 200, 100, 0.6)');
    starGrad.addColorStop(1, 'rgba(255, 180, 50, 0)');
    ctx.beginPath(); ctx.arc(cx, cy, 18, 0, Math.PI * 2); ctx.fillStyle = starGrad; ctx.fill();
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2); ctx.fillStyle = 'rgba(255, 250, 230, 1)'; ctx.fill();

    ORBIT_PLANETS.forEach(p => {
      const angle = ((p.startAngle + elapsed * p.speed) % 360) * (Math.PI / 180);
      const px = cx + Math.cos(angle) * p.radius, py = cy + Math.sin(angle) * p.radius;
      const glow = ctx.createRadialGradient(px, py, 0, px, py, p.size * 4);
      glow.addColorStop(0, p.color + 'CC'); glow.addColorStop(0.5, p.color + '44'); glow.addColorStop(1, p.color + '00');
      ctx.beginPath(); ctx.arc(px, py, p.size * 4, 0, Math.PI * 2); ctx.fillStyle = glow; ctx.fill();
      ctx.beginPath(); ctx.arc(px, py, p.size, 0, Math.PI * 2); ctx.fillStyle = p.color; ctx.fill();
    });

    requestAnimationFrame(draw);
  };
  requestAnimationFrame(draw);
}

/* ── ScoringTicker ───────────────────────────────────────── */
(function scoringTicker() {
  const feed = CANDIDATES.map(c =>
    `<div class="item"><b>${c.id}</b><em>·</em><i style="color:${probColor(c.prob)}">P=${c.prob.toFixed(3)}</i><em>·</em><span>${c.period.toFixed(1)}d</span><em>·</em><span>${c.source}</span><em>·</em><span>SNR ${c.snr.toFixed(1)}</span></div>`
  ).join('');
  document.getElementById('ticker-track').innerHTML = feed + feed;
})();

/* ── charts ──────────────────────────────────────────────── */
const AXIS = '#8A8FA8';
const GRID = 'rgba(255,255,255,0.04)';
const AXIS_LINE = 'rgba(255,255,255,0.1)';

let chartObservers = [];
function clearCharts() { chartObservers.forEach(o => o.disconnect()); chartObservers = []; }

function niceTicks(min, max, count) {
  if (min === max) return [min];
  const span = max - min;
  const step = Math.pow(10, Math.floor(Math.log10(span / count)));
  const err = (span / count) / step;
  const mult = err >= 7.5 ? 10 : err >= 3.5 ? 5 : err >= 1.5 ? 2 : 1;
  const s = step * mult;
  const ticks = [];
  for (let v = Math.ceil(min / s) * s; v <= max + s * 1e-6; v += s) ticks.push(+v.toFixed(10));
  return ticks;
}

function renderChart(container, cfg) {
  let lastW = -1;
  const draw = () => {
    const w = container.clientWidth;
    if (!w || w === lastW) return;
    lastW = w;
    const h = cfg.height;
    const m = Object.assign({ top: 5, right: 20, bottom: 30, left: 48 }, cfg.margin);
    const iw = Math.max(10, w - m.left - m.right);
    const ih = Math.max(10, h - m.top - m.bottom);
    const data = cfg.data;

    const xs = data.map(d => d[cfg.xKey]);
    const xMin = cfg.xDomain ? cfg.xDomain[0] : Math.min(...xs);
    const xMax = cfg.xDomain ? cfg.xDomain[1] : Math.max(...xs);

    let yMin = Infinity, yMax = -Infinity;
    cfg.series.forEach(s => data.forEach(d => {
      const v = d[s.key];
      if (typeof v === 'number') { if (v < yMin) yMin = v; if (v > yMax) yMax = v; }
    }));
    if (cfg.yDomain) { [yMin, yMax] = cfg.yDomain(yMin, yMax); }

    const X = v => m.left + ((v - xMin) / (xMax - xMin || 1)) * iw;
    const Y = v => m.top + ih - ((v - yMin) / (yMax - yMin || 1)) * ih;

    const xTicks = cfg.xTicks || niceTicks(xMin, xMax, 6);
    const yTicks = cfg.yTicks || niceTicks(yMin, yMax, 5);
    const fx = cfg.xFormat || (v => String(+v.toFixed(4)));
    const fy = cfg.yFormat || (v => String(+v.toFixed(4)));

    let s = `<svg width="${w}" height="${h}" role="img">`;
    xTicks.forEach(t => { s += `<line x1="${X(t)}" y1="${m.top}" x2="${X(t)}" y2="${m.top + ih}" stroke="${GRID}" stroke-dasharray="2 4"/>`; });
    yTicks.forEach(t => { s += `<line x1="${m.left}" y1="${Y(t)}" x2="${m.left + iw}" y2="${Y(t)}" stroke="${GRID}" stroke-dasharray="2 4"/>`; });

    (cfg.refLines || []).forEach(r => {
      if (r.y !== undefined) {
        s += `<line x1="${m.left}" y1="${Y(r.y)}" x2="${m.left + iw}" y2="${Y(r.y)}" stroke="${r.stroke}" ${r.dash ? `stroke-dasharray="${r.dash}"` : ''}/>`;
        if (r.label) s += `<text class="chart-axis-text" x="${m.left + 6}" y="${Y(r.y) - 5}" font-size="9" fill="${r.labelFill || r.stroke}">${r.label}</text>`;
      }
      if (r.segment) {
        s += `<line x1="${X(r.segment[0].x)}" y1="${Y(r.segment[0].y)}" x2="${X(r.segment[1].x)}" y2="${Y(r.segment[1].y)}" stroke="${r.stroke}" ${r.dash ? `stroke-dasharray="${r.dash}"` : ''}/>`;
      }
    });

    cfg.series.forEach(sr => {
      const pts = data.map(d => [X(d[cfg.xKey]), Y(d[sr.key])]);
      const path = pts.map((p, i) => `${i ? 'L' : 'M'}${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(' ');
      if (sr.type === 'area') {
        s += `<path d="${path} L${pts[pts.length - 1][0].toFixed(2)},${(m.top + ih).toFixed(2)} L${pts[0][0].toFixed(2)},${(m.top + ih).toFixed(2)} Z" fill="${sr.fill}" stroke="none"/>`;
      } else {
        s += `<path d="${path}" fill="none" stroke="${sr.stroke}" stroke-width="${sr.width || 1}" ${sr.dash ? `stroke-dasharray="${sr.dash}"` : ''} stroke-linejoin="round" stroke-linecap="round"/>`;
        if (sr.dots) pts.forEach(p => { s += `<circle cx="${p[0].toFixed(2)}" cy="${p[1].toFixed(2)}" r="3" fill="${sr.stroke}"/>`; });
      }
    });

    s += `<line x1="${m.left}" y1="${m.top + ih}" x2="${m.left + iw}" y2="${m.top + ih}" stroke="${AXIS_LINE}"/>`;
    s += `<line x1="${m.left}" y1="${m.top}" x2="${m.left}" y2="${m.top + ih}" stroke="${AXIS_LINE}"/>`;
    xTicks.forEach(t => { s += `<text class="chart-axis-text" x="${X(t)}" y="${m.top + ih + 14}" font-size="${cfg.fontSize || 10}" text-anchor="middle">${fx(t)}</text>`; });
    yTicks.forEach(t => { s += `<text class="chart-axis-text" x="${m.left - 6}" y="${Y(t) + 3}" font-size="${cfg.fontSize || 10}" text-anchor="end">${fy(t)}</text>`; });
    if (cfg.xLabel) s += `<text class="chart-axis-text" x="${m.left + iw / 2}" y="${h - 2}" font-size="${cfg.fontSize || 10}" text-anchor="middle">${cfg.xLabel}</text>`;
    if (cfg.yLabel) s += `<text class="chart-axis-text" x="12" y="${m.top + ih / 2}" font-size="${cfg.fontSize || 10}" text-anchor="middle" transform="rotate(-90 12 ${m.top + ih / 2})">${cfg.yLabel}</text>`;

    s += `<rect class="chart-hit" x="${m.left}" y="${m.top}" width="${iw}" height="${ih}" fill="transparent"/>`;
    s += `</svg><div class="chart-tooltip"></div>`;
    container.innerHTML = s;

    const tip = container.querySelector('.chart-tooltip');
    const hit = container.querySelector('.chart-hit');
    hit.addEventListener('pointermove', e => {
      const rect = container.getBoundingClientRect();
      const xv = xMin + ((e.clientX - rect.left - m.left) / iw) * (xMax - xMin);
      let best = 0, bd = Infinity;
      data.forEach((d, i) => { const dd = Math.abs(d[cfg.xKey] - xv); if (dd < bd) { bd = dd; best = i; } });
      const d = data[best];
      tip.innerHTML = (cfg.tooltipLabel ? `<div style="color:${AXIS};margin-bottom:0.25rem;font-size:0.6rem">${cfg.tooltipLabel(d[cfg.xKey])}</div>` : '')
        + cfg.series.filter(sr => !sr.hideTooltip).map(sr =>
          `<div style="display:flex;gap:0.5rem;justify-content:space-between"><span style="color:${AXIS}">${sr.name}:</span><span style="color:${cfg.tooltipValueColor || '#4DFFD2'}">${(cfg.tooltipFormat || (v => v.toFixed(5)))(d[sr.key])}</span></div>`
        ).join('');
      tip.style.opacity = '1';
      tip.style.left = Math.min(Math.max(X(d[cfg.xKey]) + 12, 0), w - tip.offsetWidth - 2) + 'px';
      tip.style.top = (m.top + 8) + 'px';
    });
    hit.addEventListener('pointerleave', () => { tip.style.opacity = '0'; });
  };

  draw();
  const ro = new ResizeObserver(() => draw());
  ro.observe(container);
  chartObservers.push(ro);
}

/* ── router ──────────────────────────────────────────────── */
const NAV_LINKS = [
  { label:'Mission',   href:'#/' },
  { label:'Catalogue', href:'#/catalogue' },
  { label:'Vetting',   href:'#/vetting/TOI-4328.01' },
  { label:'Model',     href:'#/model' },
  { label:'Upload',    href:'#/upload' },
];

function renderNav(path) {
  document.getElementById('nav-links').innerHTML = NAV_LINKS.map(l => {
    const active = l.href === '#' + path || (l.href.startsWith('#/vetting') && path.startsWith('/vetting'));
    return `<a href="${l.href}" class="nav-link${active ? ' active' : ''}" style="color:${active ? '#F0EEE8' : 'rgba(240,238,232,0.5)'}">${l.label}</a>`;
  }).join('');
}

const app = document.getElementById('app');

function bindNavButtons() {
  app.querySelectorAll('[data-nav]:not([data-bound])').forEach(b => {
    b.setAttribute('data-bound', '');
    b.addEventListener('click', () => { location.hash = b.dataset.nav; });
  });
}

function route() {
  clearCharts();
  stopHealth();
  const path = location.hash.replace(/^#/, '') || '/';
  renderNav(path);
  window.scrollTo(0, 0);
  if (path === '/catalogue') return Catalogue();
  if (path.startsWith('/vetting/')) return Vetting(decodeURIComponent(path.slice('/vetting/'.length)));
  if (path === '/vetting') return Vetting('TOI-4328.01');
  if (path === '/model') return ModelPerformance();
  if (path === '/upload') return Upload();
  return Home();
}

window.addEventListener('hashchange', route);
window.addEventListener('scroll', () => {
  document.getElementById('nav').classList.toggle('scrolled', window.scrollY > 40);
}, { passive: true });
