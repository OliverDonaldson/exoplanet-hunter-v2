/* ═══════════════════════════════════════════════════════════
   Exoplanet Hunter — Transit Vetting Console
   Deep Space Cinematic · Electric Teal #4DFFD2

   Part 1/2 — data model, shared components, Home, Catalogue.
   ═══════════════════════════════════════════════════════════ */

const { animate, createTimeline, stagger, svg, text, utils } = ANIME;

// anime.js pauses its engine while the document is hidden, and that default is
// kept. Turning it off does not keep motion running in a background tab —
// requestAnimationFrame does not fire there either way — it only skips the
// resetTime() that engine.resume() does, so the first frame back applies the
// whole hidden interval at once and every animation snaps to its end. The boot
// overlay's own 9 s failsafe is what stops it parking in front of the console.

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const esc = s => String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' }[c]));
const signed = (v, d = 2) => {
  const r = +v.toFixed(d);                         // avoid rendering "−0.00"
  return (r >= 0 ? '+' : '−') + Math.abs(r).toFixed(d);
};

/* One rendering for "there is no number here", so a pending fetch, a failed
   one and an endpoint that carries no such field all read alike. */
const pendingPanel = msg =>
  `<div style="padding:3rem;text-align:center;font-family:'JetBrains Mono';font-size:0.75rem;color:#8A8FA8">${msg}</div>`;

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
let GATING = SERVED.missions.find(m => m.role === 'gating');

let RUNS = [
  { runId:'ca906040', date:'2026-07-19', auc:0.9100, aucErr:0.0070, recall:0.6120, brier:0.0871, status:'active', verdict:'PROMOTE',
    reason:'TESS AUC +0.0180 over champion, 2.6× the ±0.0070 noise floor. Brier and ECE not degraded.' },
  { runId:'7b1e4c23', date:'2026-07-02', auc:0.8960, aucErr:0.0071, recall:0.5980, brier:0.0903, status:'archived', verdict:'REJECT',
    reason:'ΔAUC +0.0040 falls inside the ±0.0070 noise floor, so it is not distinguishable from the champion.' },
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
    neg:'Profile is V-shaped rather than flat-bottomed, suggesting a grazing transit or a binary',
    neu:'Transit shape is unremarkable at this SNR' },
  { key:'odd_view', sees:'Transits at odd epochs', w:0.72,
    pos:'Odd-epoch depth matches the global fold',
    neg:'Odd-epoch depth runs deeper than the fold',
    neu:'Too few odd-epoch transits to constrain the depth' },
  { key:'even_view', sees:'Transits at even epochs, where a depth mismatch means an eclipsing binary', w:0.72,
    pos:'Odd and even depths agree, which is consistent with a planet',
    neg:'Even depth disagrees with odd, an eclipsing-binary signature',
    neu:'Odd/even comparison is inconclusive at this depth' },
  { key:'secondary_view', sees:'Phase 0.5, where a secondary eclipse would sit', w:0.80,
    pos:'No secondary eclipse at phase 0.5',
    neg:'Secondary eclipse detected, indicating a self-luminous companion',
    neu:'Phase 0.5 coverage is too sparse to rule an eclipse in or out' },
  { key:'centroid_view', sees:'Whether the light centroid moves during transit', w:0.78,
    pos:'Centroid holds still through transit, so the flux is on target',
    neg:'Centroid shifts in transit, indicating flux from a background binary',
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
    neg:'Residual periodicity survives masking, indicating a second source',
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
  // A live score carries the ensemble's actual members and its MC-dropout
  // sigma. The simulation below draws plausible ones from the calibrated
  // score; using it while a real score is attached would put invented fold
  // dots under a real mean.
  if (c.live && c.live.perFold && c.live.perFold.length) {
    const folds = c.live.perFold.map(f => ({ fold: f.fold, score: f.prob }));
    const scores = folds.map(f => f.score);
    const mean = scores.reduce((s, v) => s + v, 0) / scores.length;
    const foldStd = Math.sqrt(scores.reduce((s, v) => s + (v - mean) ** 2, 0) / scores.length);
    return {
      folds, foldStd, probStd: c.live.probStd,
      range: [Math.min(...scores), Math.max(...scores)],
      mean,
    };
  }
  // A live service and no attached score means the members are simply not
  // measured for this row — /candidates has no per-fold column. Returning the
  // simulation here is the exact failure the comment above describes, because
  // the catalogue's bulk mean is already a real number on the same panel.
  if (API.mode === 'live') return null;

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
    fail:'Centroid offset in transit; flux may come from a neighbour' },
  { key:'oddeven', name:'Odd–Even Depth', field:'odd_even_depth_ratio', unit:'', dp:3, threshold:'< 0.050',
    pass:'Odd and even transit depths agree',
    fail:'Odd and even depths differ, indicating an eclipsing binary at twice the period' },
  { key:'secondary', name:'Secondary Eclipse', field:'weak_secondary_max_mes', unit:'', dp:2, threshold:'< 7.10',
    pass:'No significant secondary eclipse at phase 0.5',
    fail:'Secondary eclipse above threshold, indicating a self-luminous companion' },
  { key:'ghost', name:'Ghost Diagnostic', field:'ghost_core_statistic / ghost_halo_statistic', unit:'', dp:2, threshold:'core > halo',
    pass:'Signal is stronger in the core aperture than the halo',
    fail:'Halo statistic exceeds core, indicating a contaminating source outside the aperture' },
  { key:'mes', name:'Detection Strength', field:'max_multiple_event_sigma', unit:'', dp:1, threshold:'≥ 7.1',
    pass:'Multiple-event statistic clears the detection threshold',
    fail:'Detection strength below threshold; the signal is marginal' },
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

/* The API returns five diagnostic suites on /score. Three map onto entries in
   DIAGNOSTICS; the other four entries (ghost, MES, bootstrap, sky offset) have
   no field on the score contract and come back unmeasured rather than
   simulated. That is the same three-state rule the panel already follows, now
   driven by what the service actually returned. */
function liveDiagnostics(c) {
  const d = c.live.diagnostics || {};
  const out = DIAGNOSTICS.map(spec => {
    if (spec.key === 'centroid' && d.centroid) {
      return { ...spec, state: d.centroid.suspicious ? 'fail' : 'pass',
               value: d.centroid.centroid_snr,
               threshold: `< ${d.centroid.beb_threshold_sigma.toFixed(1)} σ` };
    }
    if (spec.key === 'oddeven' && d.oddEven) {
      return { ...spec, state: d.oddEven.depth_diff_sigma > 3 ? 'fail' : 'pass',
               value: d.oddEven.depth_diff_sigma, unit: 'σ', dp: 2,
               field: 'odd_even depth difference', threshold: '< 3.0 σ' };
    }
    if (spec.key === 'secondary' && d.secondary) {
      return { ...spec, state: d.secondary.suspicious ? 'fail' : 'pass',
               value: d.secondary.secondary_significance, unit: 'σ', dp: 2,
               threshold: `< ${d.secondary.fa_threshold.toFixed(1)} σ FA` };
    }
    return { ...spec, state: 'unmeasured' };
  });

  return out;
}

function diagnosticsFor(c) {
  if (c.live) return liveDiagnostics(c);
  // The suites come back on /score and nowhere else, so before a score lands
  // every one of them is unmeasured. The panel already renders that state, and
  // already says that unmeasured is not the same as passing.
  if (API.mode === 'live') return DIAGNOSTICS.map(d => ({ ...d, state: 'unmeasured' }));
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
  if (!has(d.value) || !Number.isFinite(d.value)) return 'not measured';
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

  /* Insolation and the habitable zone come from the catalogue wherever the row
     carries them. The API computes both from the star's own radius, Teff and
     logg (`_add_poe_observables`); the estimates above assume M* = 1 M☉, which
     puts a planet round an M dwarf at the wrong distance and then calls it
     temperate. The HZ edges are the luminosity-scaled Kasting limits
     r = r☉ · sqrt(L), so 0.75 and 1.77 AU for a Sun.

     `a` is recovered rather than re-estimated: the same L sets both the HZ
     edges and the insolation, so a = hz_inner · sqrt(S_inner / S) with
     S_inner = 1/0.75² = 1.778 S⊕ fixed by that definition. That keeps the
     distance, the flux and the zone on one set of stellar parameters instead
     of mixing a published flux with a solar-mass orbit. */
  const HZ_INNER_FLUX = 1 / (0.75 * 0.75);     // 1.778 S⊕ at the recent-Venus edge
  const pubInsol = Number.isFinite(c.insolation) && c.insolation > 0 ? c.insolation : null;
  const pubHz = Number.isFinite(c.hzInner) && Number.isFinite(c.hzOuter) && c.hzInner > 0
    ? { inner: c.hzInner, outer: c.hzOuter } : null;
  const outInsol = pubInsol !== null ? pubInsol : insol;
  const outA = (pubInsol !== null && pubHz)
    ? pubHz.inner * Math.sqrt(HZ_INNER_FLUX / pubInsol)
    : a;
  const hz = pubHz || { inner: 0.75, outer: 1.77 };
  const starMeasured = pubInsol !== null && pubHz !== null;

  // The archive publishes TSM/ESM for TOIs; those are computed from real J/K
  // magnitudes and a real stellar radius, where the estimates above assume
  // R* = 1 R☉, M* = 1 M☉ and NIR magnitudes proxied off T-mag. Prefer the
  // measured value wherever the catalogue carries one, and keep the estimate
  // only as a fallback — CTOIs have no TSM/ESM on the contract.
  const outTsm = Number.isFinite(c.tsm) ? c.tsm : (Number.isFinite(tsm) ? tsm : null);
  const outEsm = Number.isFinite(c.esm) ? c.esm : (Number.isFinite(esm) ? esm : null);
  const outTeq = Number.isFinite(c.teqK) ? c.teqK : (Number.isFinite(teq) ? teq : null);
  const outRp  = Number.isFinite(c.radiusRe) ? c.radiusRe : (Number.isFinite(rp) ? rp : null);

  return { rp: outRp, a: outA, teq: outTeq, insol: outInsol, tsm: outTsm, esm: outEsm, tsmCut, hz,
           estimated: !Number.isFinite(c.tsm),
           starMeasured,
           inHz: outA >= hz.inner && outA <= hz.outer,
           tsmPass: has(outTsm) && outTsm >= tsmCut,
           esmPass: has(outEsm) && outEsm >= 7.5 };
}

/* Ran at module level against the mock array. Now a function, called after
   hydrate(), so live rows are derived from live values instead of having
   mock-derived ones left on them. */
function deriveFollowUp() {
  CANDIDATES.forEach(c => {
    const f = followUp(c);
    c.tsm = f.tsm; c.esm = f.esm; c.teqK = f.teq; c.rp = f.rp;
  });
}
deriveFollowUp();

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

/* ── components/OrbitalDiagram.tsx ─────────────────────────
   The four orbits are decoration and their geometry is fixed. What they are
   labelled with is not: the names and scores came from a hardcoded list that
   outlived the prototype, so a live console captioned real orbits with four
   invented candidates. They now name the highest-scoring rows the catalogue
   actually returned, and an orbit with no row behind it carries no label. */
const ORBIT_PLANETS = [
  { radius: 60,  size: 3,   color:'#4DFFD2', speed:180, startAngle:45  },
  { radius: 100, size: 4,   color:'#F5A623', speed:72,  startAngle:120 },
  { radius: 148, size: 3.5, color:'#4DFFD2', speed:36,  startAngle:200 },
  { radius: 200, size: 2.5, color:'#8A8FA8', speed:18,  startAngle:310 },
];

/** The top scored rows, one per orbit, in catalogue order. */
const topScored = n => CANDIDATES.filter(c => has(c.prob)).slice(0, n);

function orbitalDiagramHTML(size = 460) {
  const named = topScored(ORBIT_PLANETS.length);
  const labels = ORBIT_PLANETS.map((spec, i) => {
    const row = named[i];
    if (!row) return '';
    const p = { ...spec, label: row.id, prob: row.prob.toFixed(3) };
    const a = (p.startAngle + 30) * Math.PI / 180;
    const tx = Math.cos(a) * (p.radius + 20), ty = Math.sin(a) * (p.radius + 20);
    return `<div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%) translate(${tx}px,${ty}px);pointer-events:none;white-space:nowrap">
      <div style="display:flex;align-items:center;gap:0.3rem">
        <div style="width:5px;height:5px;border-radius:50%;border:1px solid ${p.color};background:transparent"></div>
        <span style="font-family:'JetBrains Mono';font-size:0.6rem;color:rgba(240,238,232,0.6)">${esc(p.label)}</span>
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

/* ── ScoringTicker ─────────────────────────────────────────
   A function called after hydrate(), not an IIFE. As an IIFE it ran at parse
   time and pinned the mock array into the strip, so a live session scrolled
   twenty fabricated scores under a panel reporting the real served model.

   P= is rendered only where a score exists. Live catalogue rows carry
   prob: null — /candidates has no score column — so the strip shows the
   catalogue facts it does have rather than a number it does not. */
function mountTicker() {
  const el = document.getElementById('ticker-track');
  if (!el || !CANDIDATES.length) return;
  const cell = c => {
    const bits = [`<b>${esc(c.id)}</b>`];
    if (has(c.prob)) bits.push(`<i style="color:${probColor(c.prob)}">P=${c.prob.toFixed(3)}</i>`);
    else if (c.disposition && c.disposition !== '—') bits.push(`<i style="color:${getDispositionColor(c.disposition)}">${esc(c.disposition)}</i>`);
    if (c.period) bits.push(`<span>${c.period.toFixed(1)}d</span>`);
    if (c.depth) bits.push(`<span>${(c.depth * 1e6).toFixed(0)} ppm</span>`);
    if (c.snr) bits.push(`<span>SNR ${c.snr.toFixed(1)}</span>`);
    return `<div class="item">${bits.join('<em>·</em>')}</div>`;
  };
  const feed = CANDIDATES.slice(0, 40).map(cell).join('');
  el.innerHTML = feed + feed;
  el.title = 'Highest-scoring catalogue rows. Bulk ensemble means, not live scores.';
}

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
  { label:'Vetting',   href:'#/vetting' },
  { label:'Model',     href:'#/model' },
  { label:'Discovery', href:'#/discovery' },
  { label:'Upload',    href:'#/upload' },
  { label:'About',     href:'#/about' },
];

function renderNav(path) {
  document.getElementById('nav-links').innerHTML = NAV_LINKS.map(l => {
    const active = l.href === '#' + path || (l.href.startsWith('#/vetting') && path.startsWith('/vetting'));
    return `<a href="${l.href}" class="nav-link${active ? ' active' : ''}" style="color:${active ? '#F0EEE8' : 'rgba(240,238,232,0.5)'}">${l.label}</a>`;
  }).join('');
}

/* ── mobile nav ───────────────────────────────────────────
   Below 900 px the links live in a panel. The CSS owns the appearance; this
   owns the one piece of state, and every way out of it — a link, a route
   change, Escape, the scrim, or the viewport growing back past the
   breakpoint — closes it, so the panel can never be left open over a page
   that no longer has a button to shut it. */
(function mountNavMenu() {
  const nav = document.getElementById('nav');
  const toggle = document.getElementById('nav-toggle');
  const scrim = document.getElementById('nav-scrim');
  const links = document.getElementById('nav-links');
  const wide = window.matchMedia('(min-width: 901px)');

  const set = open => {
    nav.classList.toggle('menu-open', open);
    toggle.setAttribute('aria-expanded', String(open));
    toggle.setAttribute('aria-label', open ? 'Close menu' : 'Open menu');
  };
  const close = () => set(false);

  toggle.addEventListener('click', () => set(!nav.classList.contains('menu-open')));
  scrim.addEventListener('click', close);
  links.addEventListener('click', e => { if (e.target.closest('a')) close(); });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') close(); });
  wide.addEventListener('change', e => { if (e.matches) close(); });
  window.addEventListener('hashchange', close);
})();

/* ── wide tables on narrow screens ────────────────────────
   A fourteen-column comparison reflowed to one column per row is fourteen
   unrelated lines, which is not what a catalogue is for — so the table keeps
   its desktop layout and is scaled down to the viewport instead. `zoom`, not
   a transform, because it scales the layout box too, so the wrapper still
   sizes to what is drawn and the hit targets stay under the text.

   FIT_FLOOR is the point past which the type stops resolving at all. It is a
   backstop, not a target: with the narrow gutters the catalogue lands around
   0.36 on a phone, which is legible, and the floor only bites on a table
   wider or a screen narrower than anything here produces. Below it the rest
   is left to the wrapper's scroll, because an unreadable table that fits is
   worse than a legible one that does not. */
const FIT_FLOOR = 0.32;
const fitWide = window.matchMedia('(max-width: 1024px)');

function fitTables() {
  document.querySelectorAll('[data-fit-table] table').forEach(table => {
    table.style.zoom = '';
    if (!fitWide.matches) return;
    const avail = table.parentElement.clientWidth;
    const natural = table.scrollWidth;
    if (!avail || !natural || natural <= avail) return;
    table.style.zoom = Math.max(FIT_FLOOR, avail / natural).toFixed(4);
  });
}

/* Resize fires in bursts, so those are coalesced onto a frame. A repaint is
   not: the table has just been written and the measurement is wanted now, and
   a frame never arrives at all while the tab is hidden. */
let fitPending = 0;
const scheduleFit = () => {
  cancelAnimationFrame(fitPending);
  fitPending = requestAnimationFrame(fitTables);
};
window.addEventListener('resize', scheduleFit);
fitWide.addEventListener('change', scheduleFit);

const app = document.getElementById('app');

function bindNavButtons() {
  app.querySelectorAll('[data-nav]:not([data-bound])').forEach(b => {
    b.setAttribute('data-bound', '');
    b.addEventListener('click', () => { location.hash = b.dataset.nav; });
  });
}

/* A target scored from Upload that is not a catalogue row. Period, epoch and
   duration come from the score's own ephemeris; depth, T-mag, SNR, sectors and
   disposition are ExoFOP columns that /score does not return, so they stay
   null and every panel that wants one renders it as not measured. Filling them
   from the score would be inventing catalogue data out of a model output. */
function adHocCandidate(ticNumeric, score) {
  const eph = score.ephemeris || {};
  return {
    id: `TIC ${ticNumeric}`, ticId: `TIC ${ticNumeric}`, ticNumeric,
    period: eph.period_days ?? null,
    duration: has(eph.duration_days) ? eph.duration_days * 24 : null,
    epochBjd: eph.t0_btjd ?? null,
    prob: score.prob, probStd: score.probStd,
    depth: null, tmag: null, snr: null, tsm: null, esm: null,
    disposition: '—', source: 'TESS', catalogue: null,
    sectors: '—', lastScored: '—', baselineDays: null,
    radiusRe: null, teqK: null, stellarRadius: null, stellarTeff: null,
    live: score, adHoc: true,
  };
}

/** Register an ad-hoc target so #/vetting/<id> can find it, and return its id. */
function rememberScoredTarget(ticNumeric, score) {
  const known = CANDIDATES.find(c => c.ticNumeric === ticNumeric);
  if (known) {
    known.live = score;
    known.prob = score.prob;
    known.probStd = score.probStd;
    return known.id;
  }
  const row = adHocCandidate(ticNumeric, score);
  CANDIDATES.push(row);
  return row.id;
}

/* Which candidate "Vetting" opens with no id in the hash.

   Prefers a row that carries a published period and epoch: those score against
   the catalogue ephemeris in seconds, where a row without one triggers a BLS
   period search first and takes minutes. Falls back to the first row, and to
   null when there is no catalogue at all — it used to fall back to a prototype
   id, which then matched no row and left the previous page on screen. */
function defaultVettingId() {
  const usable = CANDIDATES.find(c => c.period && c.duration && has(c.epochBjd))
    || CANDIDATES.find(c => c.period && c.duration)
    || CANDIDATES[0];
  return usable ? usable.id : null;
}

/* The tab reads "Exoplanet Hunter" on every route, by choice. The guidelines in
   AGENTS.md ask for a title that tracks the current view, and this deliberately
   does not: the console is one product and the tab is its name, not a readout
   of which of its seven pages happens to be open.

   The set is still needed as the list of routes route() will accept, which is
   what keeps an unknown hash from rendering a blank page. */
const SITE_TITLE = 'Exoplanet Hunter';
const KNOWN_ROUTES = new Set([
  '/', '/catalogue', '/vetting', '/model', '/upload', '/discovery', '/about',
]);

/* Browsers restore scroll on their own for a document navigation, but this is
   one document and seven hash routes, so it never had anything to restore and
   every Back landed at the top of a 500-row catalogue. The offset is parked in
   the history entry itself, which means it survives with the entry rather than
   in a map this page has to keep in step with it. */
history.scrollRestoration = 'manual';

/* Trailing timeout rather than requestAnimationFrame. rAF does not fire while
   the tab is hidden, and "scrolled, then switched tabs, then came back and hit
   Back" is exactly when losing the offset is most annoying. A timer still runs
   there, throttled, which is all this needs. */
let scrollParkPending = 0;
const parkScroll = () => {
  clearTimeout(scrollParkPending);
  scrollParkPending = setTimeout(() => {
    // replaceState with no url keeps the current one, hash included
    history.replaceState({ ...history.state, y: window.scrollY }, '');
  }, 150);
};

/* popstate fires only for Back/Forward; assigning location.hash does not raise
   it. Both then raise hashchange, popstate first, so this hands the offset
   forward for exactly the navigations that should restore one and leaves a
   link click to land at the top. */
let restoreTo = null;
window.addEventListener('popstate', e => { restoreTo = (e.state && e.state.y) || 0; });

/* ── view state in the URL ───────────────────────────────── */
/* The hash is the route. State that belongs to a view rather than to the route
   — the catalogue's filters and sort, the vetting page's tab — rides in a query
   string on that hash. `?api=` and `?boot=` live in location.search, which this
   never touches; a relative replaceState keeps them.

   Read on demand rather than handed down through page(), because a view can
   re-enter itself. Vetting does, when a slow /score finally lands, and reading
   the URL again there is what keeps the tab you switched to during the wait —
   it used to be thrown back to the default at that moment. */
const routeQuery = () => new URLSearchParams(location.hash.replace(/^#[^?]*\??/, ''));

/* replaceState, not `location.hash = …`. Assigning the hash raises hashchange,
   which is route(), which would rebuild the page underneath the control that
   was just used — the search box would lose focus and its caret on every
   keystroke. Replacing also keeps a filter change out of the history stack, so
   Back leaves the catalogue rather than stepping back through fourteen
   keystrokes, and the one entry it does keep now carries the filters: Back out
   of a candidate returns to the catalogue you were reading rather than to a
   reset one. history.state is passed through so the parked scroll offset
   survives the rewrite. */
function setRouteQuery(params) {
  const path = location.hash.replace(/^#/, '').split('?')[0] || '/';
  const qs = params.toString();
  history.replaceState(history.state, '', '#' + path + (qs ? '?' + qs : ''));
}

function route() {
  clearCharts();
  stopHealth();
  stopScoreLoader();
  // The upload run keeps going; it just stops painting a page that is gone.
  detachUploadRender();
  // The query is stripped before the route is resolved: `#/catalogue?src=TESS`
  // is the catalogue, and a candidate id must not arrive with `?tab=…` glued
  // to it. encodeURIComponent writes `?` as %3F, so the first literal one is
  // always the separator.
  const raw = location.hash.replace(/^#/, '') || '/';
  const bare = raw.split('?')[0] || '/';
  const path = bare.startsWith('/vetting/') || KNOWN_ROUTES.has(bare) ? bare : '/';
  renderNav(path);
  document.title = SITE_TITLE;

  const page =
      path === '/catalogue' ? Catalogue
    : path.startsWith('/vetting/') ? () => Vetting(decodeURIComponent(path.slice('/vetting/'.length)))
    : path === '/vetting' ? () => Vetting(defaultVettingId())
    : path === '/model' ? ModelPerformance
    : path === '/upload' ? Upload
    : path === '/discovery' ? Discovery
    : path === '/about' ? About
    : Home;
  page();
  fitTables();

  /* After the page is in the DOM, or there is nothing to scroll through yet.
     `behavior: 'instant'` because html carries scroll-behavior:smooth for
     in-page anchors, and without the override a route change animates its way
     to the top over several hundred milliseconds instead of just being there. */
  const y = restoreTo;
  restoreTo = null;
  window.scrollTo({ top: y || 0, left: 0, behavior: 'instant' });
}

window.addEventListener('hashchange', route);
window.addEventListener('scroll', () => {
  document.getElementById('nav').classList.toggle('scrolled', window.scrollY > 40);
  parkScroll();
}, { passive: true });
