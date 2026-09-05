/* ═══════════════════════════════════════════════════════════
   VETTING · MODEL PERFORMANCE · UPLOAD
   ═══════════════════════════════════════════════════════════ */

// Three verdicts, three chips. An unrecognised verdict is one this console
// cannot interpret, and the honest rendering of that is the amber chip, not red.
const VERDICT_CHIP = { PROMOTE: 'tag-promote', UNRESOLVED: 'tag-unresolved', REJECT: 'tag-reject' };

/* The views the pipeline actually returns. No fitted transit model:
   this project classifies, it does not solve for orbital parameters. */
function generateViews(c) {
  // A live score returns the very arrays the model was fed. The simulation
  // below is a plausible transit drawn from depth and duration; showing it
  // beside a real calibrated probability would caption invented curves with a
  // measured number.
  if (c.live && c.live.views && c.live.views.global) {
    const pack = v => {
      if (!v || !v.phase) return [];
      const out = [];
      for (let i = 0; i < v.phase.length; i++) {
        const f = v.flux[i];
        if (!has(f) || !isFinite(f)) continue;
        // The score contract carries no per-bin scatter, so the band collapses
        // onto the line rather than being invented around it.
        out.push({ phase: v.phase[i], flux: f, hi: f, lo: f });
      }
      return out;
    };
    const g = pack(c.live.views.global);
    const l = pack(c.live.views.local);
    const lSpan = l.length ? Math.max(Math.abs(l[0].phase), Math.abs(l[l.length - 1].phase)) : 0.02;
    const eph = c.live.ephemeris;
    const durPhaseLive = eph && eph.period_days ? (eph.duration_days / eph.period_days) : 0.02;
    return { durPhase: durPhaseLive, localSpan: lSpan, global: g, local: l };
  }
  // Live service, no score attached yet. The binned views arrive with the
  // score and nothing else on the contract carries them, so the panel says so
  // rather than drawing the simulation below under a measured period.
  if (API.mode === 'live') return null;

  const r = rngFor(c.id + '|views');
  // A catalogue row with no published period would divide by zero here.
  const durPhase = c.period ? Math.min(0.45, (c.duration / 24) / c.period) : 0.02;
  const half = durPhase / 2;

  const shape = ph => {
    const a = Math.abs(ph);
    if (a <= half * 0.6) return 1 - c.depth;
    if (a >= half) return 1;
    return 1 - c.depth * (1 - (a - half * 0.6) / (half * 0.4));
  };

  const build = (bins, span) => {
    const out = [];
    for (let i = 0; i < bins; i++) {
      const phase = -span + (2 * span * i) / (bins - 1);
      // fewer bins means more cadences per bin, so per-bin scatter falls as √bins
      const scatter = c.depth * 0.055 * Math.sqrt(bins / 2001);
      const flux = shape(phase) + (r() - 0.5) * 2 * scatter;
      out.push({
        phase: +phase.toFixed(6),
        flux: +flux.toFixed(6),
        hi: +(flux + scatter * 1.6).toFixed(6),
        lo: +(flux - scatter * 1.6).toFixed(6),
      });
    }
    return out;
  };

  const localSpan = Math.min(0.5, Math.max(durPhase * 2.5, 0.004));
  return { durPhase, localSpan, global: build(2001, 0.5), local: build(201, localSpan) };
}

function viewChart(el, data, label, span) {
  renderChart(el, {
    height: 300, data, xKey: 'phase',
    margin: { top: 8, right: 20, bottom: 34, left: 66 },
    xLabel: 'Phase', yLabel: 'Norm. Flux',
    xDomain: [-span, span],
    xFormat: v => span < 0.02 ? v.toExponential(1) : v.toFixed(2),
    yFormat: v => v.toFixed(4),
    yDomain: (min, max) => [min - (max - min) * 0.15, max + (max - min) * 0.15],
    refLines: [{ y: 1.0, stroke: 'rgba(255,255,255,0.1)', dash: '4 4' }],
    series: [
      { key:'hi',   type:'area', fill:'rgba(240,238,232,0.07)', name:'Scatter', hideTooltip:true },
      { key:'lo',   type:'area', fill:'#050608',                name:'lo',      hideTooltip:true },
      { key:'flux', stroke:'#4DFFD2', width: span < 0.05 ? 1.6 : 1, name: label },
    ],
    tooltipFormat: v => v.toFixed(5),
  });
}

function Vetting(candidateId) {
  const c = CANDIDATES.find(x => x.id === candidateId) || CANDIDATES[0];
  // Returning here used to leave the previous route's markup on screen.
  if (!c) {
    app.innerHTML = `
      <div style="min-height:100vh;background:#050608;padding-top:56px;display:flex;align-items:center;justify-content:center">
        <div style="text-align:center;max-width:32rem;padding:2rem">
          <div class="section-label" style="margin-bottom:0.75rem">Vetting Console</div>
          <div style="font-family:'Inter';font-size:0.9rem;line-height:1.7;color:rgba(240,238,232,0.6);margin-bottom:1.5rem">
            ${candidateId ? `No catalogue row matches <span style="font-family:'JetBrains Mono';color:#F0EEE8">${esc(candidateId)}</span>.` : 'The catalogue is empty, so there is nothing to vet.'}
          </div>
          <button class="btn-ghost" data-nav="#/catalogue">← Back to Catalogue</button>
        </div>
      </div>`;
    bindNavButtons();
    return;
  }

  /* Score once per candidate, then re-enter to paint the same page with real
     values. The row arrives from /candidates with no score — that endpoint has
     no score column — so P(planet), the fold dots and every diagnostic are
     absent until this resolves. Re-entry is guarded on the hash so a score
     landing after the user has navigated away does not repaint over them. */
  if (API.mode === 'live' && !c.live && !c.scoring && !c.scoreError) {
    c.scoring = true;
    /* Guarded on the route, not on the whole hash. The tab now rides in a query
       string on that hash, and switching tabs during the wait is exactly what a
       20-60 s score invites — comparing the whole thing would have made the
       result arrive to a hash that no longer matched and never paint at all,
       leaving P(planet) at `···` until the page was left and re-entered. */
    const wanted = location.hash.split('?')[0];
    loadScore(c.ticNumeric, ephemerisFor(c))
      .then(s => { c.live = s; c.prob = s.prob; })
      .catch(e => { c.scoreError = e.message; })
      .finally(() => {
        c.scoring = false;
        if (location.hash.split('?')[0] === wanted) Vetting(candidateId);
      });
  }
  const views = generateViews(c);
  const branches = branchEvidence(c);
  const agree = foldAgreement(c);
  const diags = diagnosticsFor(c);
  // TSM and ESM are functions of depth and T-mag. Both are catalogue columns,
  // so a target that arrived through Upload has neither and the panel is
  // dropped rather than computed from zeros.
  const canFollowUp = has(c.depth) && c.depth > 0 && has(c.tmag) && c.tmag > 0 && has(c.period) && c.period > 0;
  const fu = canFollowUp ? followUp(c) : null;
  const dispColor = getDispositionColor(c.disposition);
  const TABS = [
    ['pipeline',    'Pipeline'],
    ['lightcurve',  'Phase-Folded Views'],
    ['branches',    'Branch Evidence <span class="tag-chip tag-soon" style="margin-left:0.4rem">in progress</span>'],
    ['agreement',   'Model Agreement'],
    ['diagnostics', 'Diagnostic Flags'],
  ];

  // The timeline opens the page: it is the account of where the headline number
  // came from, and every other tab is one stage of it in detail. It is read
  // back from a real score response, so in prototype mode there is nothing for
  // it to describe and the phase-folded views lead instead.
  //
  // `?tab=` outranks that default, which is what makes a link to one stage of
  // one candidate's vetting hold. It also survives the re-entry above: a cold
  // /score takes 20-60 s, switching tabs while it runs is the obvious thing to
  // do, and the repaint that lands the score used to throw you back here.
  // An unrecognised tab falls through to the default rather than to a blank
  // panel, on the same rule as an unrecognised route.
  const tabFromURL = routeQuery().get('tab');
  let activeTab = TABS.some(([k]) => k === tabFromURL)
    ? tabFromURL
    : (API.mode === 'live' ? 'pipeline' : 'lightcurve');

  /* An ad-hoc target from Upload carries the score's ephemeris and nothing
     else, so every catalogue column below can legitimately be absent. */
  const num = (v, dp, unit = '') => (has(v) && Number.isFinite(v) && v !== 0
    ? v.toFixed(dp) + unit : 'not measured');

  const KEY_PARAMS = [
    { label:'Period',      value:num(c.period, 1, ' d') },
    { label:'Duration',    value:num(c.duration, 1, ' h') },
    { label:'Depth',       value:num(c.depth, 4) },
    { label:'T-mag',       value:num(c.tmag, 1) },
    { label:'SNR',         value:num(c.snr, 1) },
    { label:'Last Scored', value:c.lastScored || 'not measured' },  // scored_at, not the catalogue edit date
  ];

  const longBaseline = has(c.baselineDays) && c.baselineDays >= 1000;
  const fuVerdict = !fu ? ''
    : fu.tsmPass && fu.esmPass ? 'High priority: viable for both transmission and emission spectroscopy'
    : fu.tsmPass ? 'Transmission target: TSM clears the Kempton threshold for this radius bin'
    : fu.esmPass ? 'Emission target: ESM above the GJ 1132 b benchmark'
    : 'Below Kempton thresholds: not competitive for JWST time';

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">

      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:2rem">
        <button class="crumb" data-nav="#/catalogue" style="font-family:'Ailerons';font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;background:none;border:none;transition:color 150ms ease">← Catalogue</button>
        <span style="color:rgba(255,255,255,0.2);font-size:0.6rem">/</span>
        <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2">${c.id}</span>
      </div>

      <div class="vet-head" style="display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:start;margin-bottom:2.5rem">
        <div>
          <div class="section-label" style="margin-bottom:0.75rem">Vetting Console</div>
          <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;flex-wrap:wrap">
            <h1 style="font-family:'Ailerons';font-size:clamp(2rem, 4vw, 3rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0">${c.id}</h1>
            <span style="font-family:'JetBrains Mono';font-size:0.7rem;font-weight:600;color:${dispColor};background:${dispColor}18;border:1px solid ${dispColor}44;padding:0.25rem 0.75rem;border-radius:2px">${c.disposition}</span>
          </div>
          <div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#8A8FA8">${c.ticId} · ${c.source} · Sectors ${c.sectors} · scored by <span style="color:#4DFFD2">${SERVED.runId}</span></div>
        </div>
        <div class="prob-big" style="text-align:right;display:flex;flex-direction:column;align-items:flex-end">
          <div class="stat-label" style="margin-bottom:0.4rem">P(planet)</div>
          <div style="font-family:'JetBrains Mono';font-size:3.5rem;font-weight:500;color:${!has(c.prob) ? '#8A8FA8' : probColor(c.prob)};line-height:1;letter-spacing:-0.02em;font-variant-numeric:tabular-nums">${!has(c.prob) ? (c.scoring ? '···' : '—') : c.prob.toFixed(3)}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.68rem;color:#8A8FA8;margin-top:0.35rem">${
            !has(c.prob)
              ? (c.scoring ? 'scoring: light curve → 5-fold ensemble' : (c.scoreError ? esc(c.scoreError) : 'not scored'))
              : agree && has(agree.probStd)
                ? `± ${agree.probStd.toFixed(3)} ${agree.source === 'bulk' ? 'within-fold · bulk ensemble mean' : 'MC-dropout · Platt-calibrated'}`
                : 'spread not measured'}</div>
          <div style="margin-top:0.75rem;display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0.65rem;border:1px solid ${longBaseline ? 'rgba(245,166,35,0.35)' : 'rgba(255,255,255,0.10)'};background:${longBaseline ? 'rgba(245,166,35,0.05)' : 'transparent'}">
            <span style="font-family:'JetBrains Mono';font-size:0.62rem;color:${longBaseline ? '#F5A623' : '#8A8FA8'}">
              ${!has(c.baselineDays)
                ? 'baseline not derivable for this row'
                : `${Math.round(c.baselineDays).toLocaleString()} d observed span, ${c.baselineSource === 'ephemeris-derived' ? 'ephemeris-derived, whole periods' : 'source not stated'}${longBaseline ? ' · long baselines inflate detectability, so scores are not comparable across very different ones' : ''}`}
            </span>
          </div>
        </div>
      </div>

      <div class="six-up" style="display:grid;grid-template-columns:repeat(6, 1fr);gap:0;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.08)">
        ${KEY_PARAMS.map((p, i) => `
          <div style="padding:1.25rem 1.5rem;border-right:${i < 5 ? '1px solid rgba(255,255,255,0.08)' : 'none'};background:rgba(255,255,255,0.01)">
            <div class="stat-label" style="margin-bottom:0.4rem">${p.label}</div>
            <div class="stat-value" style="font-size:1.1rem;font-weight:500">${p.value}</div>
          </div>`).join('')}
      </div>

      ${!fu ? `
      <div class="panel" style="margin-bottom:2.5rem;padding:1.1rem 1.25rem">
        <div class="section-label" style="margin-bottom:0.4rem">Follow-up Priority</div>
        <div style="font-family:'Inter';font-size:0.78rem;line-height:1.6;color:rgba(240,238,232,0.55)">
          TSM and ESM are functions of transit depth and host T-mag, which are catalogue
          columns, and /score does not return them. This target did not arrive with a catalogue
          row, so they are not computed.
        </div>
      </div>` : `
      <div class="panel" style="margin-bottom:2.5rem">
        <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;padding:1.1rem 1.25rem;border-bottom:1px solid rgba(255,255,255,0.08)">
          <div class="section-label">Follow-up Priority</div>
          <div style="font-family:'Inter';font-size:0.75rem;color:${fu.tsmPass || fu.esmPass ? '#4DFFD2' : '#8A8FA8'}">${fuVerdict}</div>
        </div>
        <div class="fu-grid">
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">TSM</div>
            <div class="stat-value" style="font-size:1.35rem;color:${fu.tsmPass ? '#4DFFD2' : '#F0EEE8'}">${fu.tsm.toFixed(1)}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">threshold ${fu.tsmCut} · R<sub>p</sub> ${fu.rp.toFixed(1)} R⊕</div>
          </div>
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">ESM</div>
            <div class="stat-value" style="font-size:1.35rem;color:${fu.esmPass ? '#4DFFD2' : '#F0EEE8'}">${fu.esm.toFixed(2)}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">benchmark 7.50</div>
          </div>
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">Equilibrium T</div>
            <div class="stat-value" style="font-size:1.35rem">${Math.round(fu.teq)} K</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">Bond albedo 0.3</div>
          </div>
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">Insolation</div>
            <div class="stat-value" style="font-size:1.35rem">${fu.insol < 1000 ? fu.insol.toFixed(1) : fu.insol.toExponential(1)}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">S⊕ · a = ${fu.a.toFixed(3)} AU${fu.starMeasured ? '' : ' · assumes a Sun'}</div>
          </div>
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">Habitable Zone</div>
            ${fu.starMeasured
              ? `<div class="stat-value" style="font-size:1.35rem;color:${fu.inHz ? '#4DFFD2' : '#F0EEE8'}">${fu.inHz ? 'Inside' : 'Outside'}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">${fu.hz.inner.toFixed(2)}–${fu.hz.outer.toFixed(2)} AU · this star</div>`
              : `<div class="stat-value" style="font-size:1.35rem;color:#8A8FA8">not published</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">needs the star's radius, Teff and logg</div>`}
          </div>
        </div>
      </div>

`}

      <div style="display:flex;gap:0;margin-bottom:2rem;border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap" id="vet-tabs">
        ${TABS.map(([k, label]) => `<button data-tab="${k}" style="font-family:'Ailerons';font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;padding:0.75rem 1.5rem;background:none;border:none;transition:background-color 150ms ease,color 150ms ease,border-color 150ms ease,border-bottom-color 150ms ease;margin-bottom:-1px">${label}</button>`).join('')}
      </div>

      <div id="vet-panel" style="margin-bottom:3rem"></div>

      <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.08)">
        <button class="btn-ghost" data-nav="#/catalogue">← Back to Catalogue</button>
        <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
          ${CANDIDATES.slice(0, 5).filter(x => x.id !== c.id).map(x =>
            `<button class="cand-btn" data-nav="#/vetting/${encodeURIComponent(x.id)}" style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;background:none;border:1px solid rgba(255,255,255,0.08);padding:0.4rem 0.75rem;transition:background-color 150ms ease,color 150ms ease,border-color 150ms ease">${x.id}</button>`).join('')}
        </div>
      </div>
    </div>
  </div>`;

  const panel = document.getElementById('vet-panel');

  const paintTabs = () => {
    document.querySelectorAll('#vet-tabs button').forEach(b => {
      const on = b.dataset.tab === activeTab;
      b.style.borderBottom = `2px solid ${on ? '#4DFFD2' : 'transparent'}`;
      b.style.color = on ? '#F0EEE8' : '#8A8FA8';
    });
  };

  const branchPanel = () => {
    if (API.mode === 'live') {
      return `
      <div class="soon" style="margin-bottom:1.5rem">
        <div class="h">Per-branch contributions not measured yet <span class="tag-chip tag-soon" style="margin-left:0.4rem">in progress</span></div>
        <div class="d">
          The score on this page is real. Attributing it across the eleven input
          views needs branch-occlusion at serving time, which is not built. Rather
          than show a plausible split, this tab shows none. What each view
          feeds the model is listed below.
        </div>
      </div>
      <div style="display:grid;gap:0.5rem">
        ${BRANCHES.map(b => `
          <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;padding:0.7rem 0;border-bottom:1px solid rgba(255,255,255,0.06)">
            <div>
              <div style="font-family:'JetBrains Mono';font-size:0.72rem;color:#F0EEE8">${esc(b.key)}</div>
              <div style="font-family:'Inter';font-size:0.72rem;color:rgba(240,238,232,0.5);margin-top:0.15rem">${esc(b.sees)}</div>
            </div>
            <div style="font-family:'JetBrains Mono';font-size:0.68rem;color:#8A8FA8;white-space:nowrap">not measured</div>
          </div>`).join('')}
      </div>`;
    }
    const max = Math.max(...branches.map(b => Math.abs(b.value)));
    const sum = branches.reduce((s, b) => s + b.value, 0);
    return `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1.25rem">
        <div>
          <div class="stat-label" style="margin-bottom:0.25rem">Per-Branch Contribution</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
            11 input views · attributions on the probability scale, sorted by magnitude
          </div>
        </div>
        <div style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8">
          catalogue mean ${BASE_RATE.toFixed(3)} <span style="color:rgba(255,255,255,0.25)">+</span>
          <span style="color:${sum >= 0 ? '#4DFFD2' : '#FF4D4D'}">${signed(sum, 3)}</span>
          <span style="color:rgba(255,255,255,0.25)">=</span>
          <span style="color:${!has(c.prob) ? '#8A8FA8' : probColor(c.prob)}">${!has(c.prob) ? '—' : c.prob.toFixed(3)}</span>
        </div>
      </div>

      <div class="panel" style="padding:1.5rem">
        <div class="branch-scale">
          <div class="scale-spacer"></div>
          <div class="ends">
            <span style="color:#FF4D4D">← favours false positive</span>
            <span style="color:#4DFFD2">favours planet →</span>
          </div>
          <div class="scale-spacer"></div>
        </div>
        ${branches.map(b => {
          const pos = b.value >= 0;
          const w = (Math.abs(b.value) / max) * 50;
          const col = pos ? '#4DFFD2' : '#FF4D4D';
          return `
          <div class="branch-row">
            <div>
              <div class="branch-key">${b.key}</div>
              <div class="branch-sees">${esc(b.sees)}</div>
            </div>
            <div class="branch-mid">
              <div class="branch-bar">
                <div class="branch-fill" style="${pos ? 'left:50%' : 'right:50%'};width:${w}%;background:${col};opacity:0.75"></div>
              </div>
              <div class="branch-read">${esc(b.reading)}</div>
            </div>
            <div class="branch-val" style="color:${col}">${signed(b.value)}</div>
          </div>`;
        }).join('')}
      </div>`;
  };

  const agreementPanel = () => {
    // `agree` is null whenever the fold members are not measured. A catalogue
    // row carries a bulk ensemble mean but no members, so the headline number
    // can be real while there is still nothing to put under it.
    if (!has(c.prob) || !agree) {
      return pendingPanel(
        c.scoring ? 'Scoring: the ensemble members arrive with the score.'
        : c.scoreError ? `Not scored: ${esc(c.scoreError)}`
        : 'Fold members not measured: /candidates carries the bulk ensemble mean, not the members behind it.');
    }
    const lo = Math.max(0, Math.min(agree.range[0], c.prob - agree.probStd) - 0.05);
    const hi = Math.min(1, Math.max(agree.range[1], c.prob + agree.probStd) + 0.05);
    const pct = v => Math.max(0, Math.min(100, ((v - lo) / (hi - lo)) * 100));
    const verdict = agree.foldStd < 0.02
      ? 'The five fold models agree closely, so the ensemble score is well determined.'
      : agree.foldStd < 0.05
        ? 'Moderate disagreement between folds: the ensemble mean is less firm than the headline figure suggests.'
        : 'Genuine disagreement between folds. Treat the ensemble mean with caution and prefer manual vetting.';

    return `
      <div style="margin-bottom:1.25rem">
        <div class="stat-label" style="margin-bottom:0.25rem">Fold Agreement and Score Spread</div>
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
          ${agree.source === 'bulk'
            ? 'Five fold models from the bulk scorer · uncalibrated ensemble means, a different measurement from the live score above'
            : 'Five independently trained fold models, each scored live with MC-dropout'}
        </div>
      </div>

      <div class="panel" style="padding:1.75rem 1.75rem 1.25rem">
        <div class="fold-axis">
          <div class="fold-track"></div>
          ${has(agree.probStd) ? `<div class="fold-band" style="left:${pct(c.prob - agree.probStd)}%;width:${pct(c.prob + agree.probStd) - pct(c.prob - agree.probStd)}%"></div>` : ''}
          <div class="fold-mean" style="left:${pct(c.prob)}%"></div>
          ${agree.folds.map(f => `<div class="fold-dot" style="left:${pct(f.score)}%" title="fold ${f.fold} · ${f.score.toFixed(3)}"></div>`).join('')}
          <!-- The end ticks sit on the axis ends, and .fold-tick centres itself
               on its position, so each hung half its width outside the panel.
               Anchored to their own edge instead; the middle one still centres. -->
          <div class="fold-tick" style="left:0%;transform:translateX(0)">${lo.toFixed(2)}</div>
          <div class="fold-tick" style="left:50%">${((lo + hi) / 2).toFixed(2)}</div>
          <div class="fold-tick" style="left:100%;transform:translateX(-100%)">${hi.toFixed(2)}</div>
        </div>

        <div style="display:flex;gap:1.75rem;flex-wrap:wrap;margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(255,255,255,0.06)">
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:13px;height:13px;border-radius:50%;border:1px solid #4DFFD2;background:rgba(5,6,8,0.9)"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">per_fold score</span></div>
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:1px;height:14px;background:#4DFFD2;box-shadow:0 0 8px rgba(77,255,210,0.7);margin:0 6px"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">ensemble mean</span></div>
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:14px;height:10px;background:rgba(77,255,210,0.12);border-left:1px solid rgba(77,255,210,0.3);border-right:1px solid rgba(77,255,210,0.3)"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">± σ</span></div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:0;border:1px solid rgba(255,255,255,0.08);border-top:none">
        ${(() => {
          /* Two spreads, named for what each is about rather than both called
             sigma. Within-fold is the model's own uncertainty on this target;
             across-fold is how much the answer depended on which split scored
             it. The live score's prob_std is the TOTAL of the two, so it is
             labelled differently from the bulk row's pure within-fold figure. */
          const tiles = [
            { k:'Calibrated score', v:!has(c.prob) ? '—' : c.prob.toFixed(3), accent:true },
            { k: agree.source === 'bulk' ? 'Within-fold σ' : 'Total σ (MC + fold)',
              v: has(agree.probStd) ? agree.probStd.toFixed(4) : 'not measured' },
            { k:'Across-fold σ', v: has(agree.foldStd) ? agree.foldStd.toFixed(4) : 'not measured' },
            { k:'Fold range', v:`${agree.range[0].toFixed(3)} – ${agree.range[1].toFixed(3)}` },
          ];
          if (has(agree.p10) && has(agree.p90)) {
            tiles.push({ k:'MC 10th–90th pct', v:`${agree.p10.toFixed(3)} – ${agree.p90.toFixed(3)}` });
          }
          return tiles;
        })().map((m, i, all) => `
          <div style="padding:1.1rem 1.25rem;border-right:${i < all.length - 1 ? '1px solid rgba(255,255,255,0.08)' : 'none'};background:rgba(255,255,255,0.01)">
            <div class="stat-label" style="margin-bottom:0.35rem">${m.k}</div>
            <div class="stat-value" style="font-size:1.05rem;color:${m.accent ? '#4DFFD2' : '#F0EEE8'}">${m.v}</div>
          </div>`).join('')}
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(8rem,1fr));gap:1px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.06);border-top:none">
        ${agree.folds.map(f => `
          <div style="background:#050608;padding:1rem 1.1rem">
            <div class="stat-label" style="margin-bottom:0.3rem">fold ${f.fold}</div>
            <div class="stat-value" style="font-size:1rem;color:${probColor(f.score)}">${f.score.toFixed(3)}</div>
          </div>`).join('')}
      </div>

      <div style="font-family:'Inter';font-size:0.8rem;line-height:1.6;color:rgba(240,238,232,0.6);margin-top:1.25rem">${verdict}</div>`;
  };

  const diagnosticsPanel = () => {
    const measured = diags.filter(d => d.state !== 'unmeasured').length;
    const failed = diags.filter(d => d.state === 'fail').length;
    const noReport = measured === 0;

    return `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1.25rem">
        <div class="stat-label">Automated Vetting Diagnostics</div>
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
          ${measured} of ${diags.length} measured${failed ? ` · <span style="color:#FF4D4D">${failed} failing</span>` : ''}
        </div>
      </div>

      ${noReport ? `
      <div class="note" style="margin-bottom:1.5rem">
        <span class="ico">▲</span>
        <span class="txt">${c.scoring
          ? 'The suites come back with the score, which is still running.'
          : c.scoreError
            ? `This target was not scored (${esc(c.scoreError)}), so none of these tests have been run.`
            : 'No Data Validation report exists for this target, so none of these tests have been run.'}
        <b style="color:rgba(240,238,232,0.85);font-weight:500">Absent is not the same as passing</b>. An unmeasured diagnostic carries no evidence either way, and the score below was produced without it.</span>
      </div>` : ''}

      <div class="diag-grid" style="display:grid;grid-template-columns:repeat(2, 1fr);gap:1px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.06)">
        ${diags.map(d => `
          <div class="diag-card ${d.state}">
            <div class="diag-marker ${d.state}"></div>
            <div style="flex:1;min-width:0">
              <div style="display:flex;justify-content:space-between;align-items:baseline;gap:0.75rem;flex-wrap:wrap">
                <div>
                  <div class="diag-name">${d.name}</div>
                  <div class="diag-field">${esc(d.field)}</div>
                </div>
                <div style="display:flex;gap:0.75rem;align-items:baseline">
                  <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:${d.state === 'pass' ? '#4DFFD2' : d.state === 'fail' ? '#FF4D4D' : 'rgba(138,143,168,0.75)'}">${esc(diagValue(d))}</span>
                  <span style="font-family:'JetBrains Mono';font-size:0.62rem;color:${d.state === 'unmeasured' ? 'rgba(138,143,168,0.4)' : '#8A8FA8'}">${esc(d.threshold)}</span>
                </div>
              </div>
              <div class="diag-desc">${d.state === 'unmeasured' ? 'Not measured: no value returned for this field.' : esc(d.state === 'pass' ? d.pass : d.fail)}</div>
            </div>
          </div>`).join('')}
      </div>`;
  };

  const viewsPanel = () => !views ? pendingPanel(
        c.scoring ? 'Scoring: the binned views arrive with the score.'
      : c.scoreError ? `Not scored: ${esc(c.scoreError)}`
      : 'Not scored: no photometry has been fetched for this target.') : `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
        <div>
          <div class="stat-label" style="margin-bottom:0.25rem">Phase-Folded Photometry</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
            P = ${num(c.period, 3, ' d')} · depth = ${num(c.depth, 4)} · SNR = ${num(c.snr, 1)} · transit spans ${(views.durPhase * 100).toFixed(3)}% of phase
          </div>
        </div>
        <div style="display:flex;gap:1.5rem">
          <div style="display:flex;align-items:center;gap:0.4rem"><div style="width:20px;height:2px;background:#4DFFD2"></div><span style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8">Binned median</span></div>
          <div style="display:flex;align-items:center;gap:0.4rem"><div style="width:14px;height:9px;background:rgba(240,238,232,0.12)"></div><span style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8">Per-bin scatter</span></div>
        </div>
      </div>

      <div class="charts-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:1.5rem">
        <div class="panel" style="padding:1.25rem 1rem 0.5rem">
          <div style="display:flex;justify-content:space-between;align-items:baseline;padding:0 0.5rem 0.5rem">
            <div class="stat-label">Global view</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8">2001 bins · full phase</div>
          </div>
          <div class="chart-wrap" id="chart-global"></div>
        </div>
        <div class="panel" style="padding:1.25rem 1rem 0.5rem">
          <div style="display:flex;justify-content:space-between;align-items:baseline;padding:0 0.5rem 0.5rem">
            <div class="stat-label">Local view</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8">201 bins · ±${views.localSpan.toFixed(4)} phase</div>
          </div>
          <div class="chart-wrap" id="chart-local"></div>
        </div>
      </div>

      <div class="note" style="margin-top:1.25rem">
        <span class="ico">▸</span>
        <span class="txt">These are the binned views the network is fed, not a fitted transit model.
        This pipeline classifies light curves, it does not solve for orbital parameters. The line is the per-bin median; the band is the per-bin scatter.</span>
      </div>`;

  /* ── the pipeline timeline ──────────────────────────────
     Every stage between "a TIC went in" and "this number came out", with what
     each one actually produced. The figures are all from the score itself; the
     one stage the service does not retain says so, because a timeline with a
     silent gap in it is worse than one that names the gap. */
  const step = (n, title, what, outs, absent) => `
    <div class="pipe-step${absent ? ' absent' : ''}">
      <div class="n">STAGE ${String(n).padStart(2, '0')}</div>
      <h4>${title}</h4>
      <div class="what">${what}</div>
      ${outs.length ? `<div class="pipe-out">${outs.map(o => `
        <div><div class="k">${o.k}</div><div class="v${o.dim ? ' dim' : ''}">${o.v}</div></div>`).join('')}</div>` : ''}
    </div>`;

  const pipelinePanel = () => {
    const live = c.live;
    if (!live) {
      return pendingPanel(
        c.scoring ? 'Scoring: the pipeline record is written as the score is produced.'
        : c.scoreError ? `Not scored: ${esc(c.scoreError)}`
        : 'Not scored: there is no run to describe yet.');
    }

    const eph = live.ephemeris || {};
    const cov = live.coverage || {};
    const folds = live.perFold || [];
    const fs = folds.map(f => f.prob);
    const foldSd = fs.length
      ? Math.sqrt(fs.reduce((a, v) => a + (v - fs.reduce((x, y) => x + y, 0) / fs.length) ** 2, 0) / fs.length) : null;
    const shift = has(live.prob) && has(live.probMean) ? live.prob - live.probMean : null;
    const ephSource = { catalogue: 'from the catalogue row', user: 'supplied with the request', bls: 'solved by a BLS period search' }[eph.source] || 'unrecorded';
    const suites = [
      ['centroid', live.diagnostics.centroid], ['odd/even', live.diagnostics.oddEven],
      ['secondary', live.diagnostics.secondary], ['duration', live.diagnostics.duration],
      ['false alarms', live.diagnostics.falseAlarms],
    ];
    const returned = suites.filter(([, v]) => v);
    const flagged = returned.filter(([, v]) => v.suspicious);
    const binCov = k => cov[k] ? `${cov[k].filled} / ${cov[k].total} bins` : 'not returned';

    return `
      <div style="margin-bottom:1.5rem">
        <div class="stat-label" style="margin-bottom:0.25rem">How this number was produced</div>
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
          ${esc(live.modelVersion || SERVED.modelVersion)} · one <span style="color:#4DFFD2">GET /score/${c.ticNumeric}</span> · every figure below is from that response
        </div>
      </div>
      <div class="pipe">
        ${step(1, 'Target and ephemeris',
          `The period, epoch and duration fix where in the light curve a transit should fall. Everything downstream is folded on them, so a wrong ephemeris makes every later stage wrong in the same way. This one was <b style="color:rgba(240,238,232,0.8);font-weight:500">${ephSource}</b>.`,
          [{ k: 'TIC', v: c.ticNumeric },
           { k: 'Period', v: has(eph.period_days) ? `${eph.period_days.toFixed(5)} d` : 'not measured' },
           { k: 'Epoch t0', v: has(eph.t0_btjd) ? `${eph.t0_btjd.toFixed(4)} BTJD` : 'not measured' },
           { k: 'Duration', v: has(eph.duration_days) ? `${(eph.duration_days * 24).toFixed(2)} h` : 'not measured' },
           { k: 'Source', v: eph.source || 'not recorded', dim: !eph.source }])}

        ${step(2, 'Raw photometry',
          `SPOC or PDC photometry is pulled from MAST and detrended before anything else happens. <b style="color:rgba(240,238,232,0.8);font-weight:500">The series itself is not part of the score contract</b>: /score returns the binned views built from it, not the cadences that went in, so the unprocessed curve cannot be drawn here. It is the one stage on this page with no numbers of its own.`,
          [{ k: 'Cadences returned', v: 'not measured', dim: true },
           { k: 'Sectors', v: c.sectors && c.sectors !== '—' ? esc(c.sectors) : 'not measured', dim: !c.sectors || c.sectors === '—' }],
          true)}

        ${step(3, 'Detrend and phase-fold',
          `The detrended flux is folded on the ephemeris above and binned twice: a global view over the whole phase, and a local view zoomed on the transit. These two arrays are what the network is fed. Bins with no cadence in them come back empty, so the counts below are coverage, not array length.`,
          [{ k: 'Global view', v: binCov('global') },
           { k: 'Local view', v: binCov('local') },
           { k: 'Local span', v: cov.local && has(cov.local.span) ? `±${cov.local.span.toFixed(4)} phase` : 'not measured' },
           { k: 'Transit width', v: views ? `${(views.durPhase * 100).toFixed(3)}% of phase` : 'not measured' }])}

        ${step(4, 'The other input views',
          `Four more views carry the evidence the odd/even, secondary and centroid tests read. A view the service could not build is absent rather than zero-filled, and the branch that reads it then has nothing to contribute.`,
          [{ k: 'Odd epochs', v: binCov('odd'), dim: !cov.odd },
           { k: 'Even epochs', v: binCov('even'), dim: !cov.even },
           { k: 'Centroid track', v: binCov('centroidTrack'), dim: !cov.centroidTrack },
           { k: 'Periodogram', v: cov.periodogram ? `${cov.periodogram.total} samples · peak ${cov.periodogram.bestPeriodDays.toFixed(4)} d` : 'not returned', dim: !cov.periodogram }],
          !cov.odd && !cov.even && !cov.centroidTrack && !cov.periodogram)}

        ${step(5, 'The ensemble',
          `Each of the ${folds.length || 5} fold models scores the views independently. They were trained on different splits, so their spread is the part of the uncertainty that comes from which data the model happened to see. Wide spread here means the answer depends on the split, and the mean deserves less weight than its own precision suggests.`,
          folds.length
            ? folds.map(f => ({ k: `fold ${f.fold}`, v: f.prob.toFixed(4) }))
                .concat([{ k: 'Fold σ', v: has(foldSd) ? foldSd.toFixed(4) : 'not measured' }])
            : [{ k: 'Members', v: 'not returned', dim: true }])}

        ${step(6, 'MC-dropout',
          `Dropout is left on at inference and the ensemble is run ${has(live.nMc) ? live.nMc : 'n'} times, so each pass sees a slightly different network. The spread across those passes is the model's own uncertainty about this target, separate from the disagreement between folds above.`,
          [{ k: 'Stochastic passes', v: has(live.nMc) ? live.nMc : 'not measured' },
           { k: 'Ensemble mean', v: has(live.probMean) ? live.probMean.toFixed(4) : 'not measured' },
           { k: 'MC-dropout σ', v: has(live.probStd) ? live.probStd.toFixed(4) : 'not measured' }])}

        ${step(7, 'Platt calibration',
          `The raw mean is not a probability: a network trained to separate classes is free to be confident and wrong. Platt scaling maps it onto the frequency actually observed out of fold, which is what makes "0.9" mean nine in ten. The reliability diagram on the Model page is the evidence for this step.`,
          [{ k: 'Before', v: has(live.probMean) ? live.probMean.toFixed(4) : 'not measured' },
           { k: 'After', v: has(live.prob) ? live.prob.toFixed(4) : 'not measured' },
           { k: 'Shift', v: has(shift) ? signed(shift, 4) : 'not measured' }])}

        ${step(8, 'Decision',
          `The calibrated score is compared with the promoted run's operating threshold. The threshold is a policy choice about how many false positives a shortlist can carry, not a property of this target; a score either side of it is still the same score.`,
          [{ k: 'Calibrated P(planet)', v: has(live.prob) ? live.prob.toFixed(4) : 'not measured' },
           { k: 'Threshold', v: has(live.threshold) ? live.threshold.toFixed(3) : 'not measured' },
           { k: 'Verdict', v: esc(live.verdict || 'not returned'), dim: !live.verdict }])}

        ${step(9, 'Automated diagnostics',
          `Run beside the model, not inside it: these are independent tests on the same photometry, and they can disagree with the score. ${returned.length ? `${flagged.length} of ${returned.length} returned suites flagged this target.` : 'None of the suites returned for this target.'} An absent suite is not a pass.`,
          suites.map(([name, v]) => ({ k: name, v: v ? (v.suspicious ? 'flagged' : 'clear') : 'not measured', dim: !v })),
          !returned.length)}
      </div>

      <div class="note" style="margin-top:1.5rem">
        <span class="ico">▸</span>
        <span class="txt">Stages 1 and 3–9 are read back from this target's own score response. Stage 2 is the
        gap: the service does not return the cadences it worked from, so the unprocessed light curve
        cannot be shown and is marked as such rather than reconstructed.</span>
      </div>`;
  };

  const paintPanel = () => {
    clearCharts();
    if (activeTab === 'pipeline') {
      panel.innerHTML = pipelinePanel();
    } else if (activeTab === 'lightcurve') {
      panel.innerHTML = viewsPanel();
      if (views) {
        viewChart(document.getElementById('chart-global'), views.global, 'Global view', 0.5);
        viewChart(document.getElementById('chart-local'), views.local, 'Local view', views.localSpan);
      }
    } else if (activeTab === 'branches') {
      panel.innerHTML = branchPanel();
    } else if (activeTab === 'agreement') {
      panel.innerHTML = agreementPanel();
    } else {
      panel.innerHTML = diagnosticsPanel();
    }
  };

  document.querySelectorAll('#vet-tabs button').forEach(b => b.addEventListener('click', () => {
    activeTab = b.dataset.tab;
    // Written for every tab including the one that is this mode's default,
    // because the default is not the same in live and prototype and a shared
    // link has to open on the stage the sender was reading rather than on
    // whichever one the recipient's mode would have picked.
    const p = routeQuery();
    p.set('tab', activeTab);
    setRouteQuery(p);
    paintTabs(); paintPanel();
  }));
  paintTabs(); paintPanel(); bindNavButtons();
}

/* ── MODEL PERFORMANCE ───────────────────────────────────── */
/* Abramowitz & Stegun 7.1.26 */
const erf = x => {
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  return s * (1 - poly * Math.exp(-x * x));
};
const normCdf = x => 0.5 * (1 + erf(x / Math.SQRT2));
const normInv = p => {
  let lo = -6, hi = 6;
  for (let i = 0; i < 60; i++) { const m = (lo + hi) / 2; if (normCdf(m) < p) lo = m; else hi = m; }
  return (lo + hi) / 2;
};

/* The measured curve where /model serves one, and only then the binormal
   stand-in. A drawn curve reads as a measurement whatever the caption says, so
   the two cases are kept apart and the chart says which it is showing. */
function rocFor(m) {
  return Array.isArray(m.roc) && m.roc.length ? m.roc : binormalRoc(m.auc);
}

/* Binormal ROC with the mission's measured AUC: the family of curves that AUC
   implies, not this run's. Used only where no per-threshold points are served. */
function binormalRoc(auc) {
  if (!has(auc) || !Number.isFinite(auc)) return [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];
  const a = Math.SQRT2 * normInv(auc);
  const pts = [{ fpr: 0, tpr: 0 }];
  for (let i = 1; i < 80; i++) {
    const fpr = i / 80;
    pts.push({ fpr: +fpr.toFixed(4), tpr: +Math.min(1, normCdf(a + normInv(fpr))).toFixed(4) });
  }
  pts.push({ fpr: 1, tpr: 1 });
  return pts;
}

function calibrationFor(m) {
  // /reliability returns the promoted run's actual reliability diagram: mean
  // predicted probability against observed positive fraction, per bin, over
  // its out-of-fold predictions. Scattering points around the diagonal by ECE
  // draws a curve with the right summary statistic and the wrong shape, which
  // is the one chart on this page whose whole purpose is its shape.
  const live = SERVED.reliability;
  if (live && Array.isArray(live.bins) && live.bins.length) {
    return live.bins.map(b => ({
      predicted: b.prob_mean,
      actual: b.frac_positive,
      perfect: b.prob_mean,
      count: b.count,
    }));
  }
  if (API.mode === 'live') return null;
  const r = rngFor(m.mission + '|calib');
  const ece = m.ece || 0;
  return [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95].map(p => ({
    predicted: p,
    actual: +Math.min(1, Math.max(0, p + (r() - 0.5) * ece * 6)).toFixed(3),
    perfect: p,
  }));
}

/* The four cells as /model measured them, or nothing.

   This used to assume a positive rate per mission and derive four
   measured-looking counts from it, which is why it refused to run live at all.
   /model now cuts the matrix at the same threshold it cuts the published
   recall at, so the cells and the recall beside them are one measurement.

   All four are required together: a matrix with a hole in it cannot be read,
   and filling the hole is how the assumption got in the first time. */
function confusionFor(m) {
  const cells = [m.tp, m.fp, m.fn, m.tn];
  if (!cells.every(v => has(v) && Number.isFinite(v))) return null;
  return { tp: m.tp, fp: m.fp, fn: m.fn, tn: m.tn };
}

function metricBlock(label, value, err, accent) {
  // A metric the run did not record renders as not-measured. Formatting null
  // through toFixed would print 0.0000, which reads as a measured zero — for
  // ECE that is perfect calibration, the most flattering possible misreading.
  if (!has(value) || !has(err)) {
    return `
    <div>
      <div class="stat-label" style="margin-bottom:0.4rem">${label}</div>
      <div style="display:flex;align-items:baseline;gap:0.4rem">
        <span class="stat-value" style="font-size:1.6rem;color:#8A8FA8">—</span>
        <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8">not measured for this run</span>
      </div>
    </div>`;
  }
  const lo = value - err, hi = value + err;
  return `
    <div>
      <div class="stat-label" style="margin-bottom:0.4rem">${label}</div>
      <div style="display:flex;align-items:baseline;gap:0.4rem">
        <span class="stat-value" style="font-size:1.6rem;color:${accent ? '#4DFFD2' : '#F0EEE8'}">${value.toFixed(4)}</span>
        <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8">±${err.toFixed(4)}</span>
      </div>
      <div class="interval">
        <span style="left:${(lo * 100).toFixed(1)}%;width:${((hi - lo) * 100).toFixed(1)}%"></span>
        <i style="left:${(value * 100).toFixed(1)}%"></i>
      </div>
    </div>`;
}

function ModelPerformance() {
  let mission = GATING.mission;

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">

      <div style="margin-bottom:2.5rem">
        <div class="section-label" style="margin-bottom:0.75rem">Model Performance</div>
        <h1 style="font-family:'Ailerons';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">${has(GATING.auc) ? GATING.auc.toFixed(4) : '—'} ON ${GATING.mission}.</h1>
        <h2 style="font-family:'Ailerons';font-size:clamp(1.2rem, 2.5vw, 2rem);font-weight:300;letter-spacing:-0.02em;color:rgba(240,238,232,0.25);line-height:1.0;margin-bottom:0.6rem">THE MISSION THAT GATES</h2>
        <p style="font-family:'Inter';font-size:0.85rem;color:rgba(240,238,232,0.45)">
          Serving <span style="font-family:'JetBrains Mono';color:#4DFFD2">${SERVED.runId}</span> since ${SERVED.promotedAt} · ${SERVED.arch}
        </p>
      </div>

      <div class="mission-grid" style="margin-bottom:1rem">
        ${SERVED.missions.map(m => `
          <div class="mission-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:1.25rem;flex-wrap:wrap">
              <div style="display:flex;align-items:baseline;gap:0.6rem">
                <span style="font-family:'Ailerons';font-size:1.05rem;font-weight:700;letter-spacing:0.04em;color:#F0EEE8">${m.mission}</span>
                <span class="tag-chip tag-gate">${m.role}</span>
              </div>
              <span class="tag-chip ${m.evaluation === 'zero-shot' ? 'tag-zeroshot' : 'tag-oof'}">${m.evaluation}</span>
            </div>
            ${metricBlock('ROC-AUC', m.auc, m.aucErr, m.role === 'gating')}
            <div style="height:1.25rem"></div>
            ${metricBlock('Recall @ 1% FPR', m.recall, m.recallErr, m.role === 'gating')}
            <div style="display:flex;gap:1.5rem;flex-wrap:wrap;margin-top:1.25rem;padding-top:1rem;border-top:1px solid rgba(255,255,255,0.06)">
              <div><div class="stat-label" style="margin-bottom:0.2rem">Brier</div><div class="stat-value" style="font-size:0.85rem">${has(m.brier) ? m.brier.toFixed(4) : '—'} <span style="color:#8A8FA8;font-size:0.7rem">${has(m.brierErr) ? '±' + m.brierErr.toFixed(4) : ''}</span></div></div>
              <div><div class="stat-label" style="margin-bottom:0.2rem">ECE</div><div class="stat-value" style="font-size:0.85rem">${has(m.ece) ? m.ece.toFixed(4) : '—'} <span style="color:#8A8FA8;font-size:0.7rem">${has(m.eceErr) ? '±' + m.eceErr.toFixed(4) : ''}</span></div></div>
              <div><div class="stat-label" style="margin-bottom:0.2rem">n</div><div class="stat-value" style="font-size:0.85rem">${m.n.toLocaleString()}</div></div>
            </div>
          </div>`).join('')}
      </div>

      <div class="note" style="margin-bottom:2.5rem">
        <span class="ico">▸</span>
        <span class="txt">
          There is no pooled headline: the missions have different label provenance and different class balance, so a single averaged figure would not mean anything.
          <b style="color:rgba(240,238,232,0.85);font-weight:500">${SERVED.missions.filter(m => m.evaluation === 'zero-shot').map(m => m.mission).join(', ') || 'None'}</b>
          ${SERVED.missions.some(m => m.evaluation === 'zero-shot') ? 'has no out-of-fold evaluation for this run, so its numbers are zero-shot transfer and are not comparable with the out-of-fold columns.' : 'runs are all out-of-fold.'}
${SERVED.noiseFloor.measured && has(SERVED.noiseFloor.auc)
            ? `Noise floor, measured on this run over ${SERVED.noiseFloor.n_models_per_fold} members per fold: AUC ±${SERVED.noiseFloor.auc.toFixed(4)}, shortlist recall ±${has(SERVED.noiseFloor.recall) ? SERVED.noiseFloor.recall.toFixed(4) : '—'}. Differences smaller than these are not differences.`
            : `Noise floor: not measured for this run. It trains one model per fold, so there is no seed spread to take the floor from, and a floor measured on another architecture would not apply to these numbers.`}
        </span>
      </div>

      <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:1.25rem">
        <div class="section-label">Per-Mission Detail</div>
        <div class="seg" id="mission-seg">
          ${SERVED.missions.map(m => `<button data-m="${m.mission}">${m.mission}</button>`).join('')}
        </div>
      </div>
      <div id="mission-detail"></div>

      <div style="margin-top:2.5rem" class="panel">
        <div style="padding:1.5rem 1.5rem 0">
          <div class="stat-label" style="margin-bottom:0.5rem">Training History</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-bottom:1.25rem">Per-epoch loss and accuracy</div>
        </div>
        <div style="padding:0 1.5rem 1.5rem">
          <div class="soon">
            <div class="h">Not yet available <span class="tag-chip tag-soon" style="margin-left:0.4rem">coming</span></div>
            <div class="d">Per-epoch loss and accuracy are not persisted by the training job yet, so there is nothing to plot. Queued behind the running block; this panel will fill in once the metrics land in the run artefacts.</div>
          </div>
        </div>
      </div>

      <div style="margin-top:2.5rem" class="panel">
        <div style="padding:1.5rem 1.5rem 1rem">
          <div class="stat-label">Run Registry &amp; Promotion Gate</div>
        </div>
        <div data-fit-table="runs" style="overflow-x:auto;padding:0 1.5rem 1.5rem">
          <table style="width:100%;border-collapse:collapse;min-width:820px">
            <thead><tr>
              ${['Run','Date','TESS AUC','Recall @1% FPR','Brier','Verdict','Reason'].map(h =>
                `<th style="padding:0.5rem 0.75rem;text-align:left;font-family:'Ailerons';font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap">${h}</th>`).join('')}
            </tr></thead>
            <tbody>
              ${RUNS.map(v => `
                <tr class="data-row" style="cursor:default">
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:${v.status === 'active' ? '#4DFFD2' : '#8A8FA8'}">${v.runId}</span>${v.status === 'active' ? '<div class="tag-chip tag-oof" style="display:inline-block;margin-left:0.4rem">served</div>' : ''}</td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">${v.date || '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.auc) ? v.auc.toFixed(4) : '—'} ${has(v.aucErr) ? `<span style="color:#8A8FA8">±${v.aucErr.toFixed(4)}</span>` : ''}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.recall) ? v.recall.toFixed(4) : '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.brier) ? v.brier.toFixed(4) : '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top">${v.verdict ? `<span class="tag-chip ${VERDICT_CHIP[v.verdict] || 'tag-unresolved'}">${esc(v.verdict)}</span>` : `<span style="color:#8A8FA8;font-family:'JetBrains Mono';font-size:0.65rem">—</span>`}</td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top;max-width:30rem"><span style="font-family:'Inter';font-size:0.75rem;line-height:1.5;color:rgba(240,238,232,0.55)">${esc(v.reason || 'No promotion log is written yet, so no reason is on record.')}</span></td>
                </tr>`).join('')}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>`;

  const detail = document.getElementById('mission-detail');

  const paintDetail = () => {
    clearCharts();
    const m = SERVED.missions.find(x => x.mission === mission);
    const cm = confusionFor(m);
    // All three are served, with their own fold spreads. They were derived
    // here from assumed cells, and their error bars were the recall's scaled
    // by 0.6 and 0.7 — two numbers on screen that nothing had measured.
    const precision = has(m.precision) ? m.precision : null;
    const recall = has(m.recall) ? m.recall : null;
    const f1 = has(m.f1) ? m.f1 : null;

    detail.innerHTML = `
      ${m.evaluation === 'zero-shot' ? `
      <div class="note" style="margin-bottom:1.25rem;border-color:rgba(245,166,35,0.35)">
        <span class="ico">▲</span>
        <span class="txt"><b style="color:#F5A623;font-weight:500">Zero-shot slice.</b> ${m.mission} was never in a training fold for ${SERVED.runId}; these curves show transfer, not held-out performance. Do not compare them against the out-of-fold missions.</span>
      </div>` : ''}

      <div class="charts-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin-bottom:2rem">
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.5rem">${m.mission} ROC Curve</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-bottom:0.35rem">AUC = ${has(m.auc) ? m.auc.toFixed(4) : '—'} ± ${has(m.aucErr) ? m.aucErr.toFixed(4) : '—'} · ${m.evaluation}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:rgba(138,143,168,0.75);margin-bottom:1rem">${Array.isArray(m.roc) && m.roc.length
            ? `${m.roc.length} measured thresholds${has(m.fprActual) ? ` · shortlist cut marked at ${(m.fprActual * 100).toFixed(2)}% FPR` : ''}`
            : 'binormal curve implied by the AUC · per-threshold points not measured'}</div>
          <div class="chart-wrap" id="chart-roc"></div>
        </div>
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.5rem">${m.mission} Calibration</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-bottom:1rem">Brier ${has(m.brier) ? m.brier.toFixed(4) : '—'} ± ${has(m.brierErr) ? m.brierErr.toFixed(4) : '—'} · ECE ${has(m.ece) ? m.ece.toFixed(4) : '—'}</div>
          <div class="chart-wrap" id="chart-calib"></div>
        </div>
      </div>

      <div class="cm-grid" style="display:grid;grid-template-columns:1fr 2fr;gap:2rem">
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.35rem">${m.mission} Confusion Matrix</div>
          <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-bottom:1.25rem">${cm && has(m.fprActual)
            ? `at the shortlist cut, score &gt; ${m.threshold.toFixed(4)}, ${(m.fprActual * 100).toFixed(2)}% of ${(m.n - m.nPositive).toLocaleString()} false positives · ${m.evaluation}`
            : `at the 1% FPR operating point · ${m.evaluation}`}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:rgba(255,255,255,0.08)">
            ${[
              { label:'True Positive',  value:cm && cm.tp, color:'#4DFFD2' },
              { label:'False Positive', value:cm && cm.fp, color:'#FF4D4D' },
              { label:'False Negative', value:cm && cm.fn, color:'#F5A623' },
              { label:'True Negative',  value:cm && cm.tn, color:'#4DFFD2' },
            ].map(cell => `
              <div style="background:#050608;padding:1.25rem;text-align:center">
                <div style="font-family:'JetBrains Mono';font-size:${cm ? '1.5rem' : '0.85rem'};font-weight:500;color:${cm ? cell.color : '#8A8FA8'};margin-bottom:0.25rem;font-variant-numeric:tabular-nums">${cm ? cell.value.toLocaleString() : 'not measured'}</div>
                <div class="stat-label">${cell.label}</div>
              </div>`).join('')}
          </div>
          ${cm ? '' : `
          <div class="note" style="margin-top:1.25rem">
            <span class="ico">▲</span>
            <span class="txt">This run reports no confusion matrix for the ${esc(m.mission)} slice. /model cuts one wherever the slice has both classes, so an absent matrix means the slice has only one.</span>
          </div>`}
        </div>
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:1.5rem">${m.mission} Derived Metrics</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:2rem">
            ${metricBlock('Precision', precision, m.precisionErr, false)}
            ${metricBlock('Recall', recall, m.recallErr, false)}
            ${metricBlock('F1', f1, m.f1Err, false)}
          </div>
          <div style="font-family:'Inter';font-size:0.75rem;line-height:1.6;color:rgba(240,238,232,0.5);margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(255,255,255,0.06)">
            Intervals are the ±1σ spread over the five folds. Recall @ 1% FPR is the promotion criterion: it is what "would this candidate reach the shortlist" actually means, and it is the number that rejected all five architecture arms.
          </div>
        </div>
      </div>`;

    renderChart(document.getElementById('chart-roc'), {
      height: 260, data: rocFor(m), xKey: 'fpr', fontSize: 9,
      margin: { top: 5, right: 10, bottom: 40, left: 52 },
      xLabel: 'False Positive Rate', yLabel: 'True Positive Rate',
      xDomain: [0, 1], yDomain: () => [0, 1],
      xFormat: v => v.toFixed(2), yFormat: v => v.toFixed(2),
      refLines: [{ segment: [{ x:0, y:0 }, { x:1, y:1 }], stroke:'rgba(255,255,255,0.15)', dash:'4 4' }],
      series: [{ key:'tpr', stroke:'#4DFFD2', width:2, name:'TPR' }],
      tooltipFormat: v => v.toFixed(4), tooltipValueColor: '#F0EEE8',
    });

    const calib = calibrationFor(m);
    if (!calib) {
      document.getElementById('chart-calib').innerHTML =
        pendingPanel('Reliability diagram unavailable: /reliability did not answer.');
    } else renderChart(document.getElementById('chart-calib'), {
      height: 260, data: calib, xKey: 'predicted', fontSize: 9,
      margin: { top: 5, right: 10, bottom: 40, left: 52 },
      xLabel: 'Mean Predicted Probability', yLabel: 'Fraction of Positives',
      xDomain: [0, 1], yDomain: () => [0, 1],
      xFormat: v => v.toFixed(2), yFormat: v => v.toFixed(2),
      series: [
        { key:'perfect', stroke:'rgba(255,255,255,0.2)', width:1, dash:'4 4', name:'Perfect' },
        { key:'actual',  stroke:'#F5A623', width:2, dots:true, name:'Observed' },
      ],
      tooltipFormat: v => v.toFixed(4), tooltipValueColor: '#F0EEE8',
    });

    document.querySelectorAll('#mission-seg button').forEach(b => b.classList.toggle('on', b.dataset.m === mission));
  };

  document.querySelectorAll('#mission-seg button').forEach(b => b.addEventListener('click', () => {
    mission = b.dataset.m; paintDetail();
  }));
  paintDetail();
}

/* ── UPLOAD ──────────────────────────────────────────────── */
/* The page offered three ways in and two of them were labelled "coming": a
   file upload the API cannot take, and a coordinate lookup nothing resolves.
   Two thirds of a control that does nothing is not a roadmap, it is a page
   pretending to be larger than it is. The one real path is now the page, and
   what the service does not do is stated once in prose below rather than sold
   as a tab. Discovery keeps its page because a whole deferred capability is
   worth naming; a greyed-out button beside a working one is not.  */

/* ── the scoring loader ───────────────────────────────────
   Six copies of the console's own motif — flat baseline, transit dip, flat
   baseline — drawn on in sequence with `svg.createDrawable`, so the wait reads
   as light curves arriving rather than as a spinner. It measures nothing and
   is not meant to: the honest count is the elapsed seconds above it.

   One handle at module scope, reverted by stopScoreLoader(), because the
   panel that holds it is torn down by a cancel, a completed run or a route
   change, and a looping animation on a detached target never stops on its own. */
const LOADER_UNITS = 6, LOADER_W = 80;
let scoreLoader = null;

function scoreLoaderHTML() {
  const paths = Array.from({ length: LOADER_UNITS }, (_, i) => {
    const x = i * LOADER_W;
    return `<path d="M ${x} 8 L ${x + 24} 8 L ${x + 29} 17 L ${x + 51} 17 L ${x + 56} 8 L ${x + LOADER_W} 8"/>`;
  }).join('');
  return `<svg id="score-loader" viewBox="0 0 ${LOADER_UNITS * LOADER_W} 24" preserveAspectRatio="none"
       aria-hidden="true" fill="none" stroke="#4DFFD2" stroke-width="1.6"
       stroke-linecap="round" stroke-linejoin="round"
       style="display:block;width:100%;height:22px;margin-top:1.25rem;opacity:0.75">${paths}</svg>`;
}

function mountScoreLoader() {
  stopScoreLoader();
  if (REDUCED || !document.getElementById('score-loader')) return;
  scoreLoader = animate(svg.createDrawable('#score-loader path'), {
    draw: ['0 0', '0 1', '1 1'],
    delay: stagger(40),
    ease: 'inOut(3)',
    duration: 1500,
    loop: true,
  });
}

function stopScoreLoader() {
  if (!scoreLoader) return;
  scoreLoader.revert();
  scoreLoader = null;
}

const RATE_LIMIT_S = 60;
let lastScoreAt = 0;

/* The run outlives the page. Navigating away used to clear the interval and
   drop the closure that held the request, so a score you had waited 40 s for
   was abandoned the moment you looked at anything else — and /score is rate
   limited to one call a minute, so starting again cost another wait. State and
   timers live out here; the page registers a render hook while it is mounted
   and clears it when it is not, and the run carries on either way. */
const UPLOAD = { ticId:'', status:'idle', progress:0, stage:0, elapsed:0,
                 script:0, awaiting:false, waitedOn:0, result:null, error:null,
                 cooldown:0, controller:null, cancelled:false, vettingId:null,
                 answeredAt:0, ephemeris:{}, blsExpected:false };
let uploadTimer = null, uploadCooldown = null, uploadRender = null;

/** Called by route() on the way out: stop painting, keep running. */
function detachUploadRender() { uploadRender = null; }

const paintUpload = () => { if (uploadRender) uploadRender(); };

/* `holds` marks the stage the run actually waits in. /score is one blocking
     request that reports nothing about its own progress, so the client knows
     exactly two things: when it was sent, and when it answered. The MAST fetch
     is what that gap is — 20-60 s cold against about three seconds for
     everything after it — so the run stops on that stage and counts real
     seconds there, instead of running the bar to 100% on 'Platt calibration'
     and waiting behind a finished animation. */
const STAGES = [
  { label:'Resolving target in the TIC',      short:'Resolve',    to:8,   ms:600 },
  { label:'Downloading photometry from MAST', short:'Download',   to:56,  ms:3400, holds:true },
  { label:'Detrending and phase-folding',     short:'Detrend',    to:72,  ms:1100 },
  { label:'Scoring 11-branch ensemble',       short:'Score',      to:91,  ms:1500 },
  { label:'Platt calibration · MC-dropout',   short:'Calibrate',  to:100, ms:900 },
];
const HOLD_AT = STAGES.findIndex(x => x.holds);
const HOLD_MS = STAGES.slice(0, HOLD_AT + 1).reduce((a, x) => a + x.ms, 0);
const TOTAL_MS = STAGES.reduce((a, x) => a + x.ms, 0);

/** Scripted milliseconds → which stage is running and how full the bar is. */
const stageAt = ms => {
  let acc = 0;
  for (let i = 0; i < STAGES.length; i++) {
    if (ms < acc + STAGES[i].ms || i === STAGES.length - 1) {
      const from = i === 0 ? 0 : STAGES[i - 1].to;
      const f = Math.max(0, Math.min(1, (ms - acc) / STAGES[i].ms));
      return { stage: i, progress: Math.round(from + (STAGES[i].to - from) * f) };
    }
    acc += STAGES[i].ms;
  }
};

function Upload() {
  const state = UPLOAD;

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:900px;margin:0 auto;padding:3rem 3rem 0">

      <div style="margin-bottom:2.5rem">
        <div class="section-label" style="margin-bottom:0.75rem">Data Submission</div>
        <h1 style="font-family:'Ailerons';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">NAME A TARGET.</h1>
        <h2 style="font-family:'Ailerons';font-size:clamp(1.2rem, 2.5vw, 2rem);font-weight:300;letter-spacing:-0.02em;color:rgba(240,238,232,0.25);line-height:1.0;margin-bottom:0.6rem">GET A CALIBRATED ANSWER.</h2>
        <p style="font-family:'Inter';font-size:0.85rem;color:rgba(240,238,232,0.45);line-height:1.7">
          Give the pipeline a TIC or KIC identifier. It pulls the photometry from MAST, rebuilds the eleven input views, and scores the target against the promoted model.
        </p>
      </div>

      <div id="up-input"></div>
      <div id="up-action"></div>

      <div style="margin-top:3rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.08)">
        <div class="stat-label" style="margin-bottom:1.25rem">What the endpoint does</div>
        <div class="fmt-grid" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:1rem">
          ${[
            { h:'GET /score/{tic_id}', t:'live', d:'Resolves the target, fetches SPOC or PDC photometry from MAST, builds the eleven views and returns a calibrated score with per-fold detail.' },
            { h:'Rate limit', t:'', d:'One scoring request per 60 seconds per client. The MAST fetch dominates the latency; expect 20–60 s on a cold cache.' },
            { h:'Returned', t:'', d:'probability, prob_std, per_fold[5], branch contributions, and whichever Data Validation fields exist for the target.' },
          ].map(f => `
            <div style="padding:1.25rem;border:1px solid rgba(255,255,255,0.06);background:rgba(255,255,255,0.01)">
              <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:0.5rem">
                <span style="font-family:'JetBrains Mono';font-size:0.72rem;font-weight:500;color:#F0EEE8">${f.h}</span>
                ${f.t ? `<span class="tag-chip tag-oof">${f.t}</span>` : ''}
              </div>
              <div style="font-family:'Inter';font-size:0.75rem;color:rgba(240,238,232,0.45);line-height:1.5">${esc(f.d)}</div>
            </div>`).join('')}
        </div>
      </div>
    </div>
  </div>`;

  const inputEl = document.getElementById('up-input');
  const actionEl = document.getElementById('up-action');

  const paintInput = () => {
    inputEl.innerHTML = `
      <div style="margin-bottom:1.5rem">
        <label for="up-tic" style="font-family:'Ailerons';font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;display:block;margin-bottom:0.5rem">TIC ID or KIC ID</label>
        <input type="text" id="up-tic" placeholder="e.g. TIC 43288669 or KIC 8120608" value="${esc(state.ticId)}"
          style="width:100%;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);color:#F0EEE8;font-family:'JetBrains Mono';font-size:0.9rem;padding:1rem 1.25rem;outline:none;box-sizing:border-box">
        <div style="font-family:'Inter';font-size:0.75rem;color:#8A8FA8;margin-top:0.5rem">
          Photometry is fetched from MAST on demand, so a first scoring run on an uncached target typically takes 20–60 seconds.
        </div>
      </div>`;
    document.getElementById('up-tic').addEventListener('input', e => { state.ticId = e.target.value; paintAction(); });
  };

  const cancelRun = () => {
    state.cancelled = true;
    if (state.controller) state.controller.abort();
    clearInterval(uploadTimer);
    uploadTimer = null;
    state.status = 'idle';
    state.awaiting = false;
    state.error = null;
    document.body.classList.remove('scoring');
    paintAction();
  };

  const paintAction = () => {
    if (state.status !== 'running') stopScoreLoader();
    if (state.status === 'ratelimited') {
      actionEl.innerHTML = `
        <div class="note" style="border-color:rgba(245,166,35,0.35)">
          <span class="ico">▲</span>
          <span class="txt"><b style="color:#F5A623;font-weight:500">Rate limited.</b>
          The scoring endpoint accepts one request per ${RATE_LIMIT_S} seconds, and each one pulls fresh photometry from MAST.
          Try again in <span style="font-family:'JetBrains Mono';color:#F5A623">${state.cooldown}s</span>.</span>
        </div>`;
      return;
    }

    if (state.status === 'idle') {
      const ok = state.ticId.trim().length > 3;
      if (state.error) {
        actionEl.innerHTML = `
          <div class="note" style="margin-bottom:1rem;border-color:rgba(255,77,77,0.35)">
            <span class="ico" style="color:#FF4D4D">▲</span>
            <span class="txt">Scoring failed: ${esc(state.error)}</span>
          </div>
          <button class="btn-teal" id="up-go" ${ok ? '' : 'disabled'} style="font-size:0.75rem;padding:0.875rem 2rem">Try again →</button>`;
        if (ok) document.getElementById('up-go').addEventListener('click', run);
        return;
      }
      actionEl.innerHTML = `<button class="btn-teal" id="up-go" ${ok ? '' : 'disabled'} style="font-size:0.75rem;padding:0.875rem 2rem">Score this target →</button>`;
      if (ok) document.getElementById('up-go').addEventListener('click', run);
      return;
    }

    /* Rendered once on entry, then patched. The panel used to be rebuilt on
       every 220 ms tick, which restarted the rail's waiting sweep and would
       have rebuilt the loader with it — the same detached-target leak the boot
       overlay had, five times a second. */
    if (state.status === 'running') {
      if (!actionEl.querySelector('#up-run')) {
        actionEl.innerHTML = `
        <div id="up-run" style="margin-top:2rem">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem">
            <div style="display:flex;align-items:center;gap:0.5rem">
              <div class="live-dot"></div>
              <span class="stat-label" id="up-stage" style="color:#4DFFD2"></span>
            </div>
            <div style="display:flex;gap:1rem;align-items:baseline">
              <span id="up-elapsed" style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8"></span>
              <span id="up-count" style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2;font-variant-numeric:tabular-nums"></span>
              <button class="btn-ghost" id="up-cancel" style="font-size:0.6rem;padding:0.3rem 0.7rem">Cancel</button>
            </div>
          </div>
          <div class="up-rail" id="up-railel"><div class="up-fill" id="up-fill"></div></div>
          <div id="up-chips" style="margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:0.5rem">
            ${STAGES.map(st => `<div style="display:flex;align-items:center;gap:0.4rem">
                <div class="up-chip-dot" style="width:6px;height:6px;border-radius:50%"></div>
                <span class="up-chip-text" style="font-family:'JetBrains Mono';font-size:0.62rem">${esc(st.short)}</span>
              </div>`).join('')}
          </div>
          ${scoreLoaderHTML()}
          <div id="up-foot" style="font-family:'Inter';font-size:0.72rem;color:#8A8FA8;margin-top:1rem"></div>
        </div>`;
        document.getElementById('up-cancel').addEventListener('click', cancelRun);
        mountScoreLoader();
      }

      document.getElementById('up-stage').textContent = `${STAGES[state.stage].label}…`;
      document.getElementById('up-elapsed').textContent = `${state.elapsed.toFixed(1)}s elapsed`;
      document.getElementById('up-count').textContent = state.awaiting
        ? `waiting on MAST · ${state.waitedOn.toFixed(1)}s`
        : `stage ${state.stage + 1} of ${STAGES.length}`;
      document.getElementById('up-railel').classList.toggle('waiting', state.awaiting);
      document.getElementById('up-fill').style.width = `${state.progress}%`;
      document.querySelectorAll('#up-chips .up-chip-dot').forEach((dot, i) => {
        const done = i < state.stage, now = i === state.stage;
        dot.style.background = done || now ? '#4DFFD2' : 'rgba(255,255,255,0.15)';
        dot.style.opacity = now ? '1' : done ? '0.6' : '1';
        dot.style.boxShadow = done || now ? '0 0 6px rgba(77,255,210,0.6)' : 'none';
      });
      document.querySelectorAll('#up-chips .up-chip-text').forEach((t, i) => {
        t.style.color = i <= state.stage ? '#4DFFD2' : '#8A8FA8';
      });
      document.getElementById('up-foot').innerHTML = !state.awaiting
        ? `Scoring runs server-side in one request; the stages are its sequence. The run will stop on <b style="color:rgba(240,238,232,0.75);font-weight:500">${esc(STAGES[HOLD_AT].short)}</b> and count real seconds there until MAST answers.`
        : 'Holding the connection open while MAST serves the light curve. /score is a single blocking request and reports nothing about its own progress, so this is real elapsed time in the fetch, not a percentage of it. The bar stops here until the answer arrives.'
          // Past a minute the wait has almost certainly stopped being the fetch.
          // The client cannot see inside one blocking request, so it says which
          // of the two it is rather than going on naming the fetch.
          + (state.blsExpected && state.waitedOn > 45
              ? ' This target carries no catalogue ephemeris, so its period is being solved by a BLS search before it can be scored. Past about a minute, that is what the wait is.'
              : '');
      return;
    }


    if (state.status === 'complete' && state.result) {
      const r = state.result;
      actionEl.innerHTML = `
        <div style="margin-top:2rem;border:1px solid rgba(77,255,210,0.2);background:rgba(77,255,210,0.02);padding:2rem">
          <div style="display:flex;justify-content:space-between;align-items:flex-start;gap:1rem;flex-wrap:wrap;margin-bottom:1.75rem">
            <div>
              <div class="section-label" style="margin-bottom:0.5rem">Scored in ${r.seconds.toFixed(1)}s</div>
              <div style="font-family:'JetBrains Mono';font-size:1.1rem;color:#F0EEE8">${esc(r.target)}</div>
              <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-top:0.3rem">${SERVED.runId} · ${r.sectors}</div>
            </div>
            <div style="text-align:right">
              <div class="stat-label" style="margin-bottom:0.25rem">P(planet)</div>
              <div style="font-family:'JetBrains Mono';font-size:2.5rem;font-weight:500;color:${probColor(r.prob)};line-height:1;font-variant-numeric:tabular-nums">${r.prob.toFixed(3)}</div>
              <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-top:0.3rem">± ${r.probStd.toFixed(3)} MC-dropout</div>
            </div>
          </div>

          <div class="stat-label" style="margin-bottom:0.6rem">Fold agreement</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(6rem,1fr));gap:1px;background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.06);margin-bottom:1.5rem">
            ${r.folds.map(f => `
              <div style="background:#050608;padding:0.8rem 0.9rem">
                <div class="stat-label" style="margin-bottom:0.25rem;font-size:0.58rem">fold ${f.fold}</div>
                <div class="stat-value" style="font-size:0.9rem;color:${probColor(f.score)}">${f.score.toFixed(3)}</div>
              </div>`).join('')}
          </div>

          <div class="stat-label" style="margin-bottom:0.6rem">Data Validation flags</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(15rem,1fr));gap:0.75rem;margin-bottom:1.5rem">
            ${r.diags.map(d => `
              <div style="display:flex;align-items:center;gap:0.75rem;padding:0.75rem;background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06)">
                <div class="diag-marker ${d.state}" style="margin-top:0"></div>
                <span style="font-family:'Ailerons';font-size:0.75rem;font-weight:500;color:${d.state === 'unmeasured' ? 'rgba(240,238,232,0.45)' : '#F0EEE8'};flex:1">${d.name}</span>
                <span style="font-family:'JetBrains Mono';font-size:0.68rem;color:${d.state === 'pass' ? '#4DFFD2' : d.state === 'fail' ? '#FF4D4D' : 'rgba(138,143,168,0.7)'}">${esc(diagValue(d))}</span>
              </div>`).join('')}
          </div>
          ${r.unmeasured ? `
          <div class="note" style="margin-bottom:1.5rem">
            <span class="ico">▲</span>
            <span class="txt">${r.unmeasured} of ${r.diags.length} diagnostics were not measured for this target: no value was returned, which is not the same as passing.</span>
          </div>` : ''}

          <div style="display:flex;gap:1rem;flex-wrap:wrap">
            <!-- was "View in Catalogue", which dropped the user on 500 unrelated
                 rows with nothing pointing at the target they had just scored. -->
            ${state.vettingId
              ? `<button class="btn-teal" data-nav="#/vetting/${encodeURIComponent(state.vettingId)}">Open full vetting →</button>`
              : `<button class="btn-teal" data-nav="#/catalogue">View in Catalogue →</button>`}
            <button class="btn-ghost" id="up-again">Score another</button>
          </div>
        </div>`;
      bindNavButtons();
      document.getElementById('up-again').addEventListener('click', () => {
        const wait = Math.ceil((RATE_LIMIT_S * 1000 - (Date.now() - lastScoreAt)) / 1000);
        if (wait > 0) { state.status = 'ratelimited'; state.cooldown = wait; startCooldown(); }
        else { state.status = 'idle'; state.result = null; }
        paintAction();
      });
    }
  };

  const startCooldown = () => {
    clearInterval(uploadCooldown);
    uploadCooldown = setInterval(() => {
      state.cooldown -= 1;
      if (state.cooldown <= 0) {
        clearInterval(uploadCooldown);
        uploadCooldown = null;
        state.status = 'idle';
        state.result = null;
      }
      paintUpload();
    }, 1000);
  };

  const run = () => {
    const wait = Math.ceil((RATE_LIMIT_S * 1000 - (Date.now() - lastScoreAt)) / 1000);
    if (lastScoreAt && wait > 0) { state.status = 'ratelimited'; state.cooldown = wait; startCooldown(); paintAction(); return; }

    state.status = 'running'; state.progress = 0; state.stage = 0;
    state.elapsed = 0; state.script = 0; state.result = null;
    state.awaiting = false; state.waitedOn = 0; state.vettingId = null;
    // The nav marks a run in flight, so leaving the page to browse does not
    // mean losing track of whether one is still going.
    document.body.classList.add('scoring');
    paintAction();

    const t0 = performance.now();
    let last = t0;
    clearInterval(uploadTimer);

    const tic = Number(String(state.ticId).replace(/[^0-9]/g, ''));
    state.cancelled = false;
    state.controller = API.mode === 'live' && tic ? new AbortController() : null;

    /* Send the catalogue ephemeris when the target is a row already held. The
       Vetting page has always done this; Upload sent {} for everything, so a
       target whose period and epoch were sitting in memory still had its period
       solved from scratch by a BLS search — seconds turned into minutes for no
       reason. Unknown targets still fall back to the search, as they must. */
    const known = CANDIDATES.find(c => c.ticNumeric === tic);
    state.ephemeris = known ? ephemerisFor(known) : {};
    state.blsExpected = !has(state.ephemeris.periodDays);

    /* The request and the staged bar run together. The bar is a readable
       account of the server's sequence, not a measurement of it — so it is
       allowed to run ahead of the request up to the MAST stage, and there it
       stops until the answer lands. */
    /* `answeredAt` is stamped where the promise settles, not where the 220 ms
       timer next runs. Browsers throttle timers in a background tab, so the
       reported "scored in" was the delay until the console noticed rather than
       the time the request took — it read 52.8 s for a run that had answered
       long before. */
    let answer = null;
    state.answeredAt = 0;
    if (state.controller) {
      loadScore(tic, state.ephemeris, { signal: state.controller.signal })
        .then(sc => { state.answeredAt = performance.now(); answer = { sc }; })
        .catch(e => { state.answeredAt = performance.now(); answer = { err: e.name === 'AbortError' ? null : e.message }; });
    }

    uploadTimer = setInterval(() => {
      const now = performance.now();
      const dt = now - last;
      last = now;
      state.elapsed = (now - t0) / 1000;

      // A mock run has nothing to wait for, so its script never stops.
      const settled = !state.controller || Boolean(answer);
      // Once the answer is in hand the remaining stages are narration, so they
      // run at 3x. A cached score answers in about 0.1 s and used to sit behind
      // seven and a half seconds of animation describing work already done.
      state.script = Math.min(settled ? TOTAL_MS : HOLD_MS, state.script + dt * (settled ? 3 : 1));
      state.awaiting = !settled && state.script >= HOLD_MS;
      if (state.awaiting) state.waitedOn = state.elapsed - HOLD_MS / 1000;

      const at = stageAt(state.script);
      // At exactly HOLD_MS the mapper has already rolled to the next stage, so
      // a run parked on the MAST fetch would be captioned "Detrending".
      state.stage = state.awaiting ? HOLD_AT : at.stage;
      state.progress = state.awaiting ? STAGES[HOLD_AT].to : at.progress;

      if (settled && state.script >= TOTAL_MS) {
        clearInterval(uploadTimer);
        uploadTimer = null;
        state.awaiting = false;
        document.body.classList.remove('scoring');
        if (!state.controller) {
          lastScoreAt = Date.now();
          state.status = 'complete';
          state.result = mockScore(state.ticId, state.elapsed);
        } else {
          // A cancelled run never happened: it must not start the rate-limit
          // cooldown, and it has nothing to report.
          if (state.cancelled || (!has(answer.err) && !has(answer.sc))) {
            state.status = 'idle';
          } else {
            lastScoreAt = Date.now();
            if (answer.err) {
              state.status = 'idle';
              state.error = answer.err;
            } else {
              state.status = 'complete';
              state.error = null;
              state.result = liveScoreResult(state.ticId, (state.answeredAt - t0) / 1000, answer.sc);
              /* The score is kept against the target so the vetting page can
                 open on it without scoring again — same endpoint, same answer,
                 and /score is rate limited to one call a minute. */
              state.vettingId = rememberScoredTarget(tic, answer.sc);
            }
          }
        }
      }

      paintUpload();
    }, 220);
  };

  // Painting resumes from wherever the run got to while the page was away.
  uploadRender = paintAction;
  paintInput(); paintAction();
}

/* The live equivalent of mockScore: the same shape, filled from /score.

   The Data Validation flags come from the score's own diagnostic suites, so a
   suite the contract does not carry reads as unmeasured rather than as a pass —
   the panel already counts those and says so underneath. */
function liveScoreResult(ticId, seconds, score) {
  const c = { id: ticId, live: score };
  const diags = diagnosticsFor(c).slice(0, 4);
  const eph = score.ephemeris || {};
  return {
    target: String(ticId).trim().toUpperCase(),
    seconds,
    sectors: `P = ${eph.period_days ? eph.period_days.toFixed(4) : '—'} d · ephemeris from ${eph.source || 'catalogue'}`,
    prob: score.prob,
    probStd: score.probStd,
    folds: (score.perFold || []).map(f => ({ fold: f.fold, score: f.prob })),
    diags,
    unmeasured: diags.filter(d => d.state === 'unmeasured').length,
  };
}

function mockScore(ticId, seconds) {
  const target = ticId.trim().toUpperCase();
  const stub = { id: target, prob: 0.847, depth: 0.0062, period: 14.3, duration: 2.8, tmag: 11.4 };
  const agree = foldAgreement(stub);
  const diags = diagnosticsFor(stub).slice(0, 4);
  return {
    target, seconds,
    sectors: 'sectors 14, 41 · 2-min cadence',
    prob: stub.prob, probStd: agree.probStd, folds: agree.folds,
    diags, unmeasured: diags.filter(d => d.state === 'unmeasured').length,
  };
}

/* ═══════════════════════════════════════════════════════════
   DISCOVERY — placeholder. The search exists; serving it does not.
   ═══════════════════════════════════════════════════════════ */
function Discovery() {
  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">
      <div class="section-label" style="margin-bottom:0.75rem">Discovery</div>
      <h1 style="font-family:'Ailerons';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">SEARCH A LIGHT CURVE.</h1>
      <div style="font-family:'Ailerons';font-size:clamp(1.4rem, 2.6vw, 2.2rem);font-weight:600;color:rgba(240,238,232,0.30);letter-spacing:-0.02em;margin-bottom:1.25rem">FIND THE SIGNAL YOURSELF</div>
      <p style="font-family:'Inter';font-size:0.9rem;line-height:1.7;color:#8A8FA8;max-width:64ch;margin-bottom:2.5rem">
        Every other page on this console vets a signal somebody else detected: a
        TOI, a KOI, a SPOC threshold-crossing event. Discovery is the other half:
        take a raw light curve with no ephemeris and search it for a periodic
        transit, then score whatever it finds.
      </p>

      <div class="soon" style="margin-bottom:2.5rem">
        <div class="h">Not yet served <span class="tag-chip tag-soon" style="margin-left:0.4rem">coming</span></div>
        <div class="d">
          The search itself is built: a box-least-squares period search over a
          configurable grid, the same one the scoring path falls back to when a
          target has no catalogue ephemeris. What is missing is the serving side:
          a survey-scale search is minutes of CPU per target, and the API has no
          job queue to run one outside a request.
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(17rem,1fr));gap:1.25rem">
        ${[
          ['What exists', 'A BLS period search with a configurable grid, and a scoring path that already uses it whenever a target arrives without an ephemeris.'],
          ['What is missing', 'A job queue. A search is minutes of CPU, so it cannot be held open inside an HTTP request the way a score with a known ephemeris can.'],
          ['Why it matters', 'ExoMiner and the SPOC pipeline vet events that were already detected. Detection plus vetting in one place is the thing neither of them offers.'],
        ].map(([h, d]) => `
          <div class="panel" style="padding:1.5rem">
            <div class="stat-label" style="margin-bottom:0.6rem">${h}</div>
            <div style="font-family:'Inter';font-size:0.78rem;line-height:1.65;color:#8A8FA8">${d}</div>
          </div>`).join('')}
      </div>
    </div>
  </div>`;
}

/* ═══════════════════════════════════════════════════════════
   ABOUT — scaffold. Sections are placeholders to be written.
   ═══════════════════════════════════════════════════════════ */
const ABOUT_SECTIONS = [
  ['What this is', 'A one-paragraph statement of the project: what it classifies, for whom, and what it does not do.'],
  ['How it works', 'The pipeline end to end: catalogue refresh, validation gates, the eleven input views, the five-fold ensemble, calibration, the promotion gate.'],
  ['How to read a score', 'What a calibrated probability means here, what the MC-dropout band is, and why a margin under the noise floor is not a difference.'],
  ['Known limits', 'The measured defects, stated plainly: observation baseline correlates with the label on TESS, and the model scores the star as well as the transit.'],
  ['Data and provenance', 'Where every input comes from (MAST, ExoFOP, the NASA archive, Gaia), and which model version served any given number.'],
  ['Credits and licence', 'Attribution, the ExoMiner work this builds on, and the licence this is released under.'],
];

function About() {
  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">
      <div class="section-label" style="margin-bottom:0.75rem">About</div>
      <h1 style="font-family:'Anurati';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:1.25rem">EXOPLANET HUNTER.</h1>
      <p style="font-family:'Inter';font-size:0.9rem;line-height:1.7;color:#8A8FA8;max-width:64ch;margin-bottom:2.5rem">
        A calibrated deep-learning pipeline for vetting transit candidates in NASA
        TESS, Kepler and K2 photometry.
      </p>

      <div class="soon" style="margin-bottom:2.5rem">
        <div class="h">Scaffold <span class="tag-chip tag-soon" style="margin-left:0.4rem">to write</span></div>
        <div class="d">
          The sections below are placeholders. Each one is a heading that needs
          its prose written; nothing here is a claim yet.
        </div>
      </div>

      <div style="display:grid;gap:1.25rem">
        ${ABOUT_SECTIONS.map(([h, d], i) => `
          <div class="panel" style="padding:1.75rem">
            <div style="display:flex;align-items:baseline;gap:0.9rem;margin-bottom:0.6rem">
              <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2">${String(i + 1).padStart(2, '0')}</span>
              <span style="font-family:'Ailerons';font-size:1rem;font-weight:600;color:#F0EEE8">${h}</span>
            </div>
            <div style="font-family:'Inter';font-size:0.8rem;line-height:1.7;color:rgba(138,143,168,0.75);padding-left:2.1rem">${d}</div>
          </div>`).join('')}
      </div>
    </div>
  </div>`;
}
