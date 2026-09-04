/* ═══════════════════════════════════════════════════════════
   API — the live service, and the seam the prototype falls through
   ═══════════════════════════════════════════════════════════

   Base URL, first hit wins: ?api= > window.EH_API_BASE > <meta
   name="eh-api-base"> > '/api'. Live vs mock is decided once by probing
   /healthz, not by a build flag — the same file is opened from disk with no
   service and served from a host that has one.

   Anything without an endpoint degrades to an explicit "not measured", never
   to a mock number: on a screen of measured figures a fabricated one is
   indistinguishable from the rest.

     prob         present, from scripts/score_candidates.py via /candidates.
                  An ENSEMBLE MEAN, not the Platt-calibrated figure /score
                  returns, so a row's catalogue and vetting numbers differ.
                  Rows the scorer has not reached stay null.
     per_mission  present, computed by /model. Missions the run never
                  evaluated are absent, so they get no card.
     branches     ABSENT — occlusion is stage 11. The tab carries the page's
                  in-progress treatment.
     verdicts     ABSENT — no promotion log exists. Archived runs also carry
                  no date; nothing records a run's completion time.
*/

/** Present = neither null nor undefined. `== null` says this in one operator
    but is loose equality, which the repo's slop gate rejects on sight. */
const has = v => v !== null && v !== undefined;

const API = {
  base: '/api',
  mode: 'unknown',            // 'live' | 'mock' | 'unknown'
  timeoutMs: 8000,
  scoreTimeoutMs: 240000,     // a cold score runs a BLS search; minutes, not seconds
  health: null,               // last /healthz body, once probed
};

(function resolveBase() {
  const fromQuery = new URLSearchParams(location.search).get('api');
  const fromGlobal = typeof window !== 'undefined' ? window.EH_API_BASE : null;
  const meta = document.querySelector('meta[name="eh-api-base"]');
  const fromMeta = meta ? meta.getAttribute('content') : null;
  API.base = (fromQuery || fromGlobal || fromMeta || '/api').replace(/\/$/, '');
})();

/** fetch with a deadline — a hung request must not leave the console spinning. */
async function apiFetch(path, { timeoutMs = API.timeoutMs, signal } = {}) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  // A caller's signal is chained onto the deadline's controller rather than
  // passed to fetch directly, so whichever fires first wins and there is still
  // exactly one abort path to clean up.
  const relay = () => ctrl.abort();
  if (signal) signal.addEventListener('abort', relay, { once: true });
  try {
    const res = await fetch(`${API.base}${path}`, { signal: ctrl.signal, cache: 'no-store' });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return await res.json();
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', relay);
  }
}

/* Decide live vs mock once, on the cheapest endpoint. Anything other than a
   clean 200 — offline, CORS, 404, timeout — means mock, because a console that
   half-loads is worse than one that is honestly a prototype.

   fly.toml suspends the machine when it is idle, so the common first request of
   the day pays a resume before it answers; at a single 4 s attempt that resume
   read as "no service" and the console then quietly served prototype data for
   the rest of the session.

   So a slow first attempt is retried once — but only when it was slow. A
   deadline abort means something is there and taking its time, which is what a
   resume looks like; any other rejection is a refused connection, a CORS block
   or a bad host, and none of those get better on a second go. That keeps the
   suspended-machine case alive without making the offline case, which is how
   this file opens from disk, sit through two full timeouts before it renders. */
async function probeApi() {
  for (let attempt = 0; attempt < 2; attempt++) {
    try {
      API.health = await apiFetch('/healthz', { timeoutMs: 8000 });
      API.mode = 'live';
      return API.mode;
    } catch (e) {
      API.probeError = e.message;
      if (e.name !== 'AbortError') break;
    }
  }
  API.mode = 'mock';
  return API.mode;
}

/* ── mappers: API contract → the shapes the pages already render ──────────
   Kept in one place so a contract change lands here and not scattered across
   four render functions. The field names on the right are api/app/schemas.py. */

/** CandidateRow → the catalogue row shape. `prob` is null by contract. */
function mapCandidate(row) {
  return {
    id: row.name,
    ticId: `TIC ${row.tic_id}`,
    ticNumeric: row.tic_id,
    period: row.period_days ?? 0,
    duration: row.duration_hours ?? 0,
    // ephemerisFor() sends this as t0; without it /score has no epoch and
    // falls back to a BLS period search, which is minutes rather than seconds.
    epochBjd: row.epoch_bjd ?? null,
    depth: has(row.depth_ppm) ? row.depth_ppm / 1e6 : 0,
    // Bulk-scored offline by scripts/score_candidates.py. An ENSEMBLE MEAN,
    // not the Platt-calibrated figure /score returns, so this row's number and
    // the one on its vetting page are computed differently and can differ.
    prob: row.prob_mean ?? null,
    probStd: row.prob_std ?? null,
    scoredAt: row.scored_at ?? null,
    disposition: row.disposition || '—',
    source: 'TESS',                          // TOI and CTOI are both TESS products
    catalogue: row.source,                   // 'TOI' | 'CTOI', kept for display
    tmag: row.tess_mag ?? 0,
    sectors: row.sectors || '—',
    lastScored: row.date_modified ? String(row.date_modified).slice(0, 10) : '—',
    snr: row.planet_snr ?? 0,
    tsm: row.tsm ?? null,
    esm: row.esm ?? null,
    // No observing-baseline column exists on the catalogue contract. The
    // console shows a baseline warning wherever a score is, so this stays null
    // and the warning is suppressed rather than shown against a guess.
    baselineDays: null,
    // Published by the API from the star's own radius, Teff and logg, so they
    // do not assume a Sun. followUp() prefers them over its own estimates.
    insolation: row.insolation_earth ?? null,
    hzInner: row.hz_inner_au ?? null,
    hzOuter: row.hz_outer_au ?? null,
    radiusRe: row.planet_radius_re ?? null,
    teqK: row.teq_k ?? null,
    stellarRadius: row.stellar_radius_rsun ?? null,
    stellarTeff: row.stellar_teff_k ?? null,
  };
}

/* The one piece of rendered text this console does not author. The pipeline
   builds its caution verdict as `f"Caution - {concerns}. ..."` with an em dash
   (scoring/diagnostics.py, the "Caution" branch), and it is the only one of the
   four verdict strings that carries one. Normalised here so the console holds
   to one punctuation style end to end. The durable fix is that one line in the
   pipeline; when it lands, this can go. */
const plainDashes = t => (typeof t === 'string' ? t.replace(/\s+\u2014\s+/g, ': ') : t);

/** How much of a binned view actually carries data. `flux` is null in a phase
    bin no cadence landed in, and the count of those is the one honest measure
    of coverage the contract offers — the pipeline panel reports it rather than
    implying every bin was observed. */
function viewCoverage(v) {
  if (!v || !Array.isArray(v.phase)) return null;
  const total = v.phase.length;
  const filled = (v.flux || []).filter(f => has(f) && Number.isFinite(f)).length;
  return { total, filled, span: total ? Math.abs(v.phase[0]) : null };
}

/** ScoreResponse → what the Vetting page needs, with API views preserved. */
function mapScore(s) {
  return {
    prob: s.prob_calibrated,
    probMean: s.prob_mean,
    probStd: s.prob_std,
    perFold: s.per_fold.map(f => ({ fold: f.fold, prob: f.prob })),
    threshold: s.decision_threshold,
    verdict: plainDashes(s.verdict),
    modelVersion: s.model_version,
    nMc: s.n_mc_samples,
    ephemeris: s.ephemeris,
    views: {
      global: s.global_view,
      local: s.local_view,
      odd: s.odd_view,
      even: s.even_view,
      centroidTrack: s.centroid_track,
      periodogram: s.periodogram,
    },
    coverage: {
      global: viewCoverage(s.global_view),
      local: viewCoverage(s.local_view),
      odd: viewCoverage(s.odd_view),
      even: viewCoverage(s.even_view),
      centroidTrack: s.centroid_track ? { total: s.centroid_track.phase.length,
        filled: (s.centroid_track.offset_pixels || []).filter(f => has(f) && Number.isFinite(f)).length } : null,
      periodogram: s.periodogram ? { total: s.periodogram.period_days.length,
        bestPeriodDays: s.periodogram.best_period_days } : null,
    },
    diagnostics: {
      centroid: s.centroid,
      oddEven: s.odd_even,
      secondary: s.secondary,
      duration: s.duration_check,
      falseAlarms: s.false_alarms,
    },
  };
}

/* ── the four real endpoints ─────────────────────────────── */

function loadHealth() {
  return apiFetch('/healthz');
}

/** One page of the catalogue. The console filters and sorts client-side, so
    it asks for a large limit rather than paging.

    Ordered by score descending server-side: the catalogue holds ~11k rows and
    only the bulk-scored ones carry a probability, so an arbitrary page would
    be mostly unscored and the P(planet) column would look broken. This puts
    the scored ones first, which is also what a triage view is for. */
async function loadCandidates({ limit = 500 } = {}) {
  const page = await apiFetch(`/candidates?limit=${limit}&sort_by=prob_mean&order=desc`);
  return { total: page.total, rows: page.rows.map(mapCandidate) };
}

/** Score one target. Slow by nature: a target with no catalogue ephemeris
    runs a BLS search first, which is minutes rather than seconds. */
async function loadScore(ticId, opts = {}, { signal } = {}) {
  const p = new URLSearchParams();
  if (has(opts.periodDays)) p.set('period_days', String(opts.periodDays));
  if (has(opts.t0Btjd)) p.set('t0_btjd', String(opts.t0Btjd));
  if (has(opts.durationHours)) p.set('duration_hours', String(opts.durationHours));
  if (opts.includePeriodogram) p.set('include_periodogram', 'true');
  const qs = p.size ? `?${p}` : '';
  const body = await apiFetch(`/score/${ticId}${qs}`, { timeoutMs: API.scoreTimeoutMs, signal });
  return mapScore(body);
}

function loadReliability() {
  return apiFetch('/reliability');
}

/** CSV export URL matching the current filters, for the catalogue button. */
function candidatesCsvUrl({ search, disposition, source } = {}) {
  const p = new URLSearchParams();
  if (search) p.set('search', search);
  if (disposition && disposition !== 'All') p.set('disposition', disposition);
  if (source && source !== 'All') p.set('source', source);
  const qs = p.size ? `?${p}` : '';
  return `${API.base}/candidates.csv${qs}`;
}

/** Catalogue-row ephemeris in the form /score wants, or {} when unusable.
    Catalogue epochs are full BJD; the API speaks BTJD (BJD − 2457000). */
function ephemerisFor(c) {
  if (!c || !c.period || c.period <= 0 || !c.duration || c.duration <= 0) return {};
  const opts = { periodDays: c.period, durationHours: c.duration };
  if (has(c.epochBjd)) {
    opts.t0Btjd = c.epochBjd > 2440000 ? c.epochBjd - 2457000 : c.epochBjd;
  }
  return opts;
}

function loadModel() {
  return apiFetch('/model');
}

function loadRuns() {
  return apiFetch('/runs?limit=8');
}

/* ── hydration ────────────────────────────────────────────
   Mutates SERVED and CANDIDATES in place before the first route() so every
   page stays synchronous. Returns the notes the UI should show about what
   could not be filled in, because the alternative — silently leaving mock
   values on screen next to live ones — is the failure this whole module
   exists to avoid.

   Declared here but called from app.boot.js: SERVED and CANDIDATES live in
   app.data.js, which is concatenated after this file, so the references
   below only resolve once everything is parsed. That is fine for a function
   body and would not be for top-level code. */
async function hydrate() {
  const notes = [];
  const mode = await probeApi();
  if (mode !== 'live') {
    return { mode, notes: [`No API at ${API.base}${API.probeError ? ` (${API.probeError})` : ''}. Showing the prototype data set.`] };
  }

  const [model, reliability, catalogue, runs] = await Promise.allSettled([
    loadModel(),
    loadReliability(),
    loadCandidates(),
    loadRuns(),
  ]);

  if (model.status === 'fulfilled') {
    const m = model.value;
    const met = m.metrics || {};
    SERVED.runId = String(m.run_id).slice(0, 8);
    SERVED.modelVersion = m.model_version;
    SERVED.promotedAt = (m.promoted_at || '').slice(0, 10);
    SERVED.arch = `${m.n_folds || 5}-fold dual-view CNN ensemble · MC-dropout · Platt calibration`;
    SERVED.metrics = met;

    SERVED.noiseFloor = m.noise_floor || { auc: null, recall: null, measured: false, source: null };
    SERVED.nScored = m.n_scored || 0;
    SERVED.nHighConfidence = m.n_high_confidence || 0;

    const n = reliability.status === 'fulfilled' ? reliability.value.n_examples : null;
    if (Array.isArray(m.per_mission) && m.per_mission.length) {
      SERVED.missions = m.per_mission;
    } else {
      // The run evaluated no resolvable mission slice. One pooled card rather
      // than three invented ones — see the endpoint docstring.
      SERVED.missions = [{
        mission: 'ALL MISSIONS', role: 'gating', evaluation: 'out-of-fold', n: n || 0,
        auc: met.roc_auc ? met.roc_auc.mean : null, aucErr: met.roc_auc ? met.roc_auc.std : null,
        recall: null, recallErr: null,
        brier: met.brier ? met.brier.mean : null, brierErr: met.brier ? met.brier.std : null,
        ece: met.ece ? met.ece.mean : null, eceErr: met.ece ? met.ece.std : null,
      }];
      notes.push('This run records no per-mission split, so metrics are pooled.');
    }
    GATING = SERVED.missions.find(x => x.role === 'gating') || SERVED.missions[0];
  } else {
    // The mock SERVED must not survive a live session: leaving three invented
    // mission cards on screen because /model 404'd is worse than showing none,
    // since everything around them is real and they would not read as mock.
    SERVED.missions = [{
      mission: 'ALL MISSIONS', role: 'gating', evaluation: 'out-of-fold', n: 0,
      auc: null, aucErr: null, recall: null, recallErr: null,
      brier: null, brierErr: null, ece: null, eceErr: null,
    }];
    SERVED.metrics = {};
    SERVED.noiseFloor = { auc: null, recall: null, measured: false, source: null };
    SERVED.nScored = 0;
    SERVED.nHighConfidence = 0;
    GATING = SERVED.missions[0];
    notes.push(`Model summary unavailable (${model.reason && model.reason.message}). Metrics are shown as not measured.`);
  }

  if (reliability.status === 'fulfilled') {
    SERVED.reliability = reliability.value;
  } else {
    notes.push('Reliability curve unavailable.');
  }

  // The prototype's run table must not survive into a live session, so RUNS is
  // emptied whichever way /runs goes. `recall` used to be pinned to null here
  // while the endpoint was returning one, which showed a measured figure as
  // not measured — the same defect as inventing one, pointed the other way.
  if (runs.status === 'fulfilled' && Array.isArray(runs.value.runs)) {
    // verdict and reason come from each run's promotion_log.json. Still null for
    // every run gated before that file existed, which is most of models/cv/.
    RUNS.length = 0;
    RUNS.push(...runs.value.runs.map(r => ({
      runId: r.short_id, date: r.date, auc: r.auc, aucErr: r.aucErr,
      recall: r.recall ?? null, brier: r.brier, status: r.status,
      verdict: r.verdict, reason: r.reason,
    })));
    if (!RUNS.length) notes.push('The service reports no runs.');
  } else {
    RUNS.length = 0;
    notes.push(`Run history unavailable${runs.reason && runs.reason.message ? `: ${runs.reason.message}` : ''}.`);
  }

  if (catalogue.status === 'fulfilled') {
    CANDIDATES.length = 0;
    CANDIDATES.push(...catalogue.value.rows);
    SERVED.catalogueTotal = catalogue.value.total;
    notes.push('Per-row P(planet) is not on the catalogue contract. Open a candidate to score it.');
  } else {
    // Same rule as the run table: eleven prototype rows rendered beside a live
    // model panel would not read as prototype rows.
    CANDIDATES.length = 0;
    SERVED.catalogueTotal = 0;
    notes.push(`Catalogue unavailable: ${catalogue.reason && catalogue.reason.message}`);
  }

  return { mode, notes };
}
