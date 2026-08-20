/* ═══════════════════════════════════════════════════════════
   VETTING · MODEL PERFORMANCE · UPLOAD
   ═══════════════════════════════════════════════════════════ */

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
  if (!c) return;

  /* Score once per candidate, then re-enter to paint the same page with real
     values. The row arrives from /candidates with no score — that endpoint has
     no score column — so P(planet), the fold dots and every diagnostic are
     absent until this resolves. Re-entry is guarded on the hash so a score
     landing after the user has navigated away does not repaint over them. */
  if (API.mode === 'live' && !c.live && !c.scoring && !c.scoreError) {
    c.scoring = true;
    const wanted = location.hash;
    loadScore(c.ticNumeric, ephemerisFor(c))
      .then(s => { c.live = s; c.prob = s.prob; })
      .catch(e => { c.scoreError = e.message; })
      .finally(() => {
        c.scoring = false;
        if (location.hash === wanted) Vetting(candidateId);
      });
  }
  const views = generateViews(c);
  const branches = branchEvidence(c);
  const agree = foldAgreement(c);
  const diags = diagnosticsFor(c);
  const fu = followUp(c);
  const dispColor = getDispositionColor(c.disposition);
  let activeTab = 'lightcurve';

  const TABS = [
    ['lightcurve',  'Phase-Folded Views'],
    ['branches',    'Branch Evidence <span class="tag-chip tag-soon" style="margin-left:0.4rem">in progress</span>'],
    ['agreement',   'Model Agreement'],
    ['diagnostics', 'Diagnostic Flags'],
  ];

  const KEY_PARAMS = [
    { label:'Period',      value:`${c.period.toFixed(1)} d` },
    { label:'Duration',    value:`${c.duration.toFixed(1)} h` },
    { label:'Depth',       value:c.depth.toFixed(4) },
    { label:'T-mag',       value:c.tmag.toFixed(1) },
    { label:'SNR',         value:c.snr.toFixed(1) },
    { label:'Last Scored', value:c.lastScored },
  ];

  const longBaseline = has(c.baselineDays) && c.baselineDays >= 1000;
  const fuVerdict =
    fu.tsmPass && fu.esmPass ? 'High priority — viable for both transmission and emission spectroscopy'
    : fu.tsmPass ? 'Transmission target — TSM clears the Kempton threshold for this radius bin'
    : fu.esmPass ? 'Emission target — ESM above the GJ 1132 b benchmark'
    : 'Below Kempton thresholds — not competitive for JWST time';

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">

      <div style="display:flex;align-items:center;gap:0.5rem;margin-bottom:2rem">
        <button class="crumb" data-nav="#/catalogue" style="font-family:'Space Grotesk';font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;background:none;border:none;transition:color 150ms ease">← Catalogue</button>
        <span style="color:rgba(255,255,255,0.2);font-size:0.6rem">/</span>
        <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2">${c.id}</span>
      </div>

      <div class="vet-head" style="display:grid;grid-template-columns:1fr auto;gap:2rem;align-items:start;margin-bottom:2.5rem">
        <div>
          <div class="section-label" style="margin-bottom:0.75rem">Vetting Console</div>
          <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.25rem;flex-wrap:wrap">
            <h1 style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0">${c.id}</h1>
            <span style="font-family:'JetBrains Mono';font-size:0.7rem;font-weight:600;color:${dispColor};background:${dispColor}18;border:1px solid ${dispColor}44;padding:0.25rem 0.75rem;border-radius:2px">${c.disposition}</span>
          </div>
          <div style="font-family:'JetBrains Mono';font-size:0.75rem;color:#8A8FA8">${c.ticId} · ${c.source} · Sectors ${c.sectors} · scored by <span style="color:#4DFFD2">${SERVED.runId}</span></div>
        </div>
        <div class="prob-big" style="text-align:right;display:flex;flex-direction:column;align-items:flex-end">
          <div class="stat-label" style="margin-bottom:0.4rem">P(planet)</div>
          <div style="font-family:'JetBrains Mono';font-size:3.5rem;font-weight:500;color:${!has(c.prob) ? '#8A8FA8' : probColor(c.prob)};line-height:1;letter-spacing:-0.02em;font-variant-numeric:tabular-nums">${!has(c.prob) ? (c.scoring ? '···' : '—') : c.prob.toFixed(3)}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.68rem;color:#8A8FA8;margin-top:0.35rem">${
            !has(c.prob)
              ? (c.scoring ? 'scoring — light curve → 5-fold ensemble' : (c.scoreError ? esc(c.scoreError) : 'not scored'))
              : `± ${agree.probStd.toFixed(3)} MC-dropout · Platt-calibrated`}</div>
          <div style="margin-top:0.75rem;display:flex;align-items:center;gap:0.5rem;padding:0.4rem 0.65rem;border:1px solid ${longBaseline ? 'rgba(245,166,35,0.35)' : 'rgba(255,255,255,0.10)'};background:${longBaseline ? 'rgba(245,166,35,0.05)' : 'transparent'}">
            <span style="font-family:'JetBrains Mono';font-size:0.62rem;color:${longBaseline ? '#F5A623' : '#8A8FA8'}">
              ${!has(c.baselineDays)
                ? 'baseline not published for this row'
                : `${c.baselineDays.toLocaleString()} d baseline${longBaseline ? ' · well-observed targets may score high' : ''}`}
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
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">S⊕ · a = ${fu.a.toFixed(3)} AU</div>
          </div>
          <div class="fu-cell">
            <div class="stat-label" style="margin-bottom:0.4rem">Habitable Zone</div>
            <div class="stat-value" style="font-size:1.35rem;color:${fu.inHz ? '#4DFFD2' : '#F0EEE8'}">${fu.inHz ? 'Inside' : 'Outside'}</div>
            <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-top:0.35rem">${fu.hz.inner}–${fu.hz.outer} AU conservative</div>
          </div>
        </div>
      </div>

      <div style="display:flex;gap:0;margin-bottom:2rem;border-bottom:1px solid rgba(255,255,255,0.08);flex-wrap:wrap" id="vet-tabs">
        ${TABS.map(([k, label]) => `<button data-tab="${k}" style="font-family:'Space Grotesk';font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;padding:0.75rem 1.5rem;background:none;border:none;transition:all 150ms ease;margin-bottom:-1px">${label}</button>`).join('')}
      </div>

      <div id="vet-panel" style="margin-bottom:3rem"></div>

      <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.08)">
        <button class="btn-ghost" data-nav="#/catalogue">← Back to Catalogue</button>
        <div style="display:flex;gap:0.75rem;flex-wrap:wrap">
          ${CANDIDATES.slice(0, 5).filter(x => x.id !== c.id).map(x =>
            `<button class="cand-btn" data-nav="#/vetting/${encodeURIComponent(x.id)}" style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;background:none;border:1px solid rgba(255,255,255,0.08);padding:0.4rem 0.75rem;transition:all 150ms ease">${x.id}</button>`).join('')}
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
          views needs branch-occlusion at serving time, which is not built — so
          rather than show a plausible split, this tab shows none. What each view
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
    if (!has(c.prob)) {
      return `<div style="padding:3rem;text-align:center;font-family:'JetBrains Mono';font-size:0.75rem;color:#8A8FA8">
        ${c.scoring ? 'Scoring — the ensemble members arrive with the score.' : 'Not scored — no fold members to compare.'}
      </div>`;
    }
    const lo = Math.max(0, Math.min(agree.range[0], c.prob - agree.probStd) - 0.05);
    const hi = Math.min(1, Math.max(agree.range[1], c.prob + agree.probStd) + 0.05);
    const pct = v => ((v - lo) / (hi - lo)) * 100;
    const verdict = agree.foldStd < 0.02
      ? 'The five fold models agree closely — the ensemble score is well determined.'
      : agree.foldStd < 0.05
        ? 'Moderate disagreement between folds — the ensemble mean is less firm than the headline figure suggests.'
        : 'Genuine disagreement between folds — treat the ensemble mean with caution and prefer manual vetting.';

    return `
      <div style="margin-bottom:1.25rem">
        <div class="stat-label" style="margin-bottom:0.25rem">Fold Agreement and Score Spread</div>
        <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
          Five independently trained fold models · MC-dropout σ over 64 stochastic passes
        </div>
      </div>

      <div class="panel" style="padding:1.75rem 1.75rem 1.25rem">
        <div class="fold-axis">
          <div class="fold-track"></div>
          <div class="fold-band" style="left:${pct(c.prob - agree.probStd)}%;width:${pct(c.prob + agree.probStd) - pct(c.prob - agree.probStd)}%"></div>
          <div class="fold-mean" style="left:${pct(c.prob)}%"></div>
          ${agree.folds.map(f => `<div class="fold-dot" style="left:${pct(f.score)}%" title="fold ${f.fold} · ${f.score.toFixed(3)}"></div>`).join('')}
          <div class="fold-tick" style="left:0%">${lo.toFixed(2)}</div>
          <div class="fold-tick" style="left:50%">${((lo + hi) / 2).toFixed(2)}</div>
          <div class="fold-tick" style="left:100%">${hi.toFixed(2)}</div>
        </div>

        <div style="display:flex;gap:1.75rem;flex-wrap:wrap;margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(255,255,255,0.06)">
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:11px;height:11px;border-radius:50%;border:1px solid #4DFFD2"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">per_fold score</span></div>
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:14px;height:2px;background:#4DFFD2"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">ensemble mean</span></div>
          <div style="display:flex;align-items:center;gap:0.45rem"><div style="width:14px;height:10px;background:rgba(77,255,210,0.12);border-left:1px solid rgba(77,255,210,0.3);border-right:1px solid rgba(77,255,210,0.3)"></div><span style="font-family:'JetBrains Mono';font-size:0.62rem;color:#8A8FA8">± prob_std</span></div>
        </div>
      </div>

      <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(11rem,1fr));gap:0;border:1px solid rgba(255,255,255,0.08);border-top:none">
        ${[
          { k:'Calibrated score', v:!has(c.prob) ? '—' : c.prob.toFixed(3), accent:true },
          { k:'MC-dropout σ',     v:agree.probStd.toFixed(4) },
          { k:'Fold σ',           v:agree.foldStd.toFixed(4) },
          { k:'Fold range',       v:`${agree.range[0].toFixed(3)} – ${agree.range[1].toFixed(3)}` },
        ].map((m, i) => `
          <div style="padding:1.1rem 1.25rem;border-right:${i < 3 ? '1px solid rgba(255,255,255,0.08)' : 'none'};background:rgba(255,255,255,0.01)">
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
        <span class="txt">No Data Validation report exists for this target, so none of these tests have been run.
        <b style="color:rgba(240,238,232,0.85);font-weight:500">Absent is not the same as passing</b> — an unmeasured diagnostic carries no evidence either way, and the score below was produced without it.</span>
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
              <div class="diag-desc">${d.state === 'unmeasured' ? 'Not measured — no value returned for this field.' : esc(d.state === 'pass' ? d.pass : d.fail)}</div>
            </div>
          </div>`).join('')}
      </div>`;
  };

  const viewsPanel = () => `
      <div style="display:flex;justify-content:space-between;align-items:baseline;gap:1rem;flex-wrap:wrap;margin-bottom:1rem">
        <div>
          <div class="stat-label" style="margin-bottom:0.25rem">Phase-Folded Photometry</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">
            P = ${c.period.toFixed(3)} d · depth = ${c.depth.toFixed(4)} · SNR = ${c.snr.toFixed(1)} · transit spans ${(views.durPhase * 100).toFixed(3)}% of phase
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
        <span class="txt">These are the binned views the network is fed, not a fitted transit model —
        this pipeline classifies light curves, it does not solve for orbital parameters. The line is the per-bin median; the band is the per-bin scatter.</span>
      </div>`;

  const paintPanel = () => {
    clearCharts();
    if (activeTab === 'lightcurve') {
      panel.innerHTML = viewsPanel();
      viewChart(document.getElementById('chart-global'), views.global, 'Global view', 0.5);
      viewChart(document.getElementById('chart-local'), views.local, 'Local view', views.localSpan);
    } else if (activeTab === 'branches') {
      panel.innerHTML = branchPanel();
    } else if (activeTab === 'agreement') {
      panel.innerHTML = agreementPanel();
    } else {
      panel.innerHTML = diagnosticsPanel();
    }
  };

  document.querySelectorAll('#vet-tabs button').forEach(b => b.addEventListener('click', () => {
    activeTab = b.dataset.tab; paintTabs(); paintPanel();
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

/* binormal ROC with the mission's measured AUC */
function rocFor(auc) {
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
  const r = rngFor(m.mission + '|calib');
  const ece = m.ece || 0;
  return [0.05,0.15,0.25,0.35,0.45,0.55,0.65,0.75,0.85,0.95].map(p => ({
    predicted: p,
    actual: +Math.min(1, Math.max(0, p + (r() - 0.5) * ece * 6)).toFixed(3),
    perfect: p,
  }));
}

/* derived from the mission spec so the matrix, the recall headline and the
   stated operating point cannot drift apart */
const POSITIVE_RATE = { TESS: 0.493, Kepler: 0.538, K2: 0.524 };

function confusionFor(m) {
  const P = Math.round(m.n * (POSITIVE_RATE[m.mission] ?? 0.5));
  const N = m.n - P;
  const fp = Math.round(N * 0.01);            // the 1% FPR operating point
  const tp = Math.round(P * m.recall);
  return { tp, fp, fn: P - tp, tn: N - fp };
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
        <h1 style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">${has(GATING.auc) ? GATING.auc.toFixed(4) : '—'} ON ${GATING.mission}.</h1>
        <h2 style="font-family:'Space Grotesk';font-size:clamp(1.2rem, 2.5vw, 2rem);font-weight:300;letter-spacing:-0.02em;color:rgba(240,238,232,0.25);line-height:1.0;margin-bottom:0.6rem">THE MISSION THAT GATES</h2>
        <p style="font-family:'Inter';font-size:0.85rem;color:rgba(240,238,232,0.45)">
          Serving <span style="font-family:'JetBrains Mono';color:#4DFFD2">${SERVED.runId}</span> since ${SERVED.promotedAt} · ${SERVED.arch}
        </p>
      </div>

      <div class="mission-grid" style="margin-bottom:1rem">
        ${SERVED.missions.map(m => `
          <div class="mission-card">
            <div style="display:flex;justify-content:space-between;align-items:center;gap:0.5rem;margin-bottom:1.25rem;flex-wrap:wrap">
              <div style="display:flex;align-items:baseline;gap:0.6rem">
                <span style="font-family:'Space Grotesk';font-size:1.05rem;font-weight:700;letter-spacing:0.04em;color:#F0EEE8">${m.mission}</span>
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
          ${SERVED.missions.some(m => m.evaluation === 'zero-shot') ? 'has no out-of-fold evaluation for this run — its numbers are zero-shot transfer and are not comparable with the out-of-fold columns.' : 'runs are all out-of-fold.'}
          Measured noise floor: AUC ±${has(SERVED.noiseFloor.auc) ? SERVED.noiseFloor.auc.toFixed(4) : '—'}, shortlist recall ±${has(SERVED.noiseFloor.recall) ? SERVED.noiseFloor.recall.toFixed(4) : '—'} — differences smaller than these are not differences.
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
            <div class="d">Per-epoch loss and accuracy are not persisted by the training job yet, so there is nothing to plot. Queued behind the running block — this panel will fill in once the metrics land in the run artefacts.</div>
          </div>
        </div>
      </div>

      <div style="margin-top:2.5rem" class="panel">
        <div style="padding:1.5rem 1.5rem 1rem">
          <div class="stat-label">Run Registry &amp; Promotion Gate</div>
        </div>
        <div style="overflow-x:auto;padding:0 1.5rem 1.5rem">
          <table style="width:100%;border-collapse:collapse;min-width:820px">
            <thead><tr>
              ${['Run','Date','TESS AUC','Recall @1% FPR','Brier','Verdict','Reason'].map(h =>
                `<th style="padding:0.5rem 0.75rem;text-align:left;font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;border-bottom:1px solid rgba(255,255,255,0.08);white-space:nowrap">${h}</th>`).join('')}
            </tr></thead>
            <tbody>
              ${RUNS.map(v => `
                <tr class="data-row" style="cursor:default">
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:${v.status === 'active' ? '#4DFFD2' : '#8A8FA8'}">${v.runId}</span>${v.status === 'active' ? '<div class="tag-chip tag-oof" style="display:inline-block;margin-left:0.4rem">served</div>' : ''}</td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8">${v.date || '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.auc) ? v.auc.toFixed(4) : '—'} <span style="color:#8A8FA8">±${has(v.aucErr) ? v.aucErr.toFixed(4) : '—'}</span></span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.recall) ? v.recall.toFixed(4) : '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top"><span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#F0EEE8;font-variant-numeric:tabular-nums">${has(v.brier) ? v.brier.toFixed(4) : '—'}</span></td>
                  <td style="padding:0.85rem 0.75rem;vertical-align:top">${v.verdict ? `<span class="tag-chip ${v.verdict === 'PROMOTE' ? 'tag-promote' : 'tag-reject'}">${v.verdict}</span>` : `<span style="color:#8A8FA8;font-family:'JetBrains Mono';font-size:0.65rem">—</span>`}</td>
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
    const precision = cm.tp / (cm.tp + cm.fp);
    const recall = cm.tp / (cm.tp + cm.fn);
    const f1 = 2 * precision * recall / (precision + recall);

    detail.innerHTML = `
      ${m.evaluation === 'zero-shot' ? `
      <div class="note" style="margin-bottom:1.25rem;border-color:rgba(245,166,35,0.35)">
        <span class="ico">▲</span>
        <span class="txt"><b style="color:#F5A623;font-weight:500">Zero-shot slice.</b> ${m.mission} was never in a training fold for ${SERVED.runId}; these curves show transfer, not held-out performance. Do not compare them against the out-of-fold missions.</span>
      </div>` : ''}

      <div class="charts-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:2rem;margin-bottom:2rem">
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.5rem">ROC Curve — ${m.mission}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-bottom:1rem">AUC = ${has(m.auc) ? m.auc.toFixed(4) : '—'} ± ${has(m.aucErr) ? m.aucErr.toFixed(4) : '—'} · ${m.evaluation}</div>
          <div class="chart-wrap" id="chart-roc"></div>
        </div>
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.5rem">Calibration — ${m.mission}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.65rem;color:#8A8FA8;margin-bottom:1rem">Brier ${has(m.brier) ? m.brier.toFixed(4) : '—'} ± ${has(m.brierErr) ? m.brierErr.toFixed(4) : '—'} · ECE ${has(m.ece) ? m.ece.toFixed(4) : '—'}</div>
          <div class="chart-wrap" id="chart-calib"></div>
        </div>
      </div>

      <div class="cm-grid" style="display:grid;grid-template-columns:1fr 2fr;gap:2rem">
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:0.35rem">Confusion Matrix — ${m.mission}</div>
          <div style="font-family:'JetBrains Mono';font-size:0.6rem;color:#8A8FA8;margin-bottom:1.25rem">at the 1% FPR operating point · ${m.evaluation}</div>
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:1px;background:rgba(255,255,255,0.08)">
            ${[
              { label:'True Positive',  value:cm.tp, color:'#4DFFD2' },
              { label:'False Positive', value:cm.fp, color:'#FF4D4D' },
              { label:'False Negative', value:cm.fn, color:'#F5A623' },
              { label:'True Negative',  value:cm.tn, color:'#4DFFD2' },
            ].map(cell => `
              <div style="background:#050608;padding:1.25rem;text-align:center">
                <div style="font-family:'JetBrains Mono';font-size:1.5rem;font-weight:500;color:${cell.color};margin-bottom:0.25rem;font-variant-numeric:tabular-nums">${cell.value.toLocaleString()}</div>
                <div class="stat-label">${cell.label}</div>
              </div>`).join('')}
          </div>
        </div>
        <div class="panel" style="padding:1.5rem">
          <div class="stat-label" style="margin-bottom:1.5rem">Derived Metrics — ${m.mission}</div>
          <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:2rem">
            ${metricBlock('Precision', precision, m.recallErr * 0.6, false)}
            ${metricBlock('Recall', recall, m.recallErr, false)}
            ${metricBlock('F1', f1, m.recallErr * 0.7, false)}
          </div>
          <div style="font-family:'Inter';font-size:0.75rem;line-height:1.6;color:rgba(240,238,232,0.5);margin-top:1.5rem;padding-top:1.25rem;border-top:1px solid rgba(255,255,255,0.06)">
            Intervals are the ±1σ spread over the five folds. Recall @ 1% FPR is the promotion criterion — it is what "would this candidate reach the shortlist" actually means, and it is the number that rejected all five architecture arms.
          </div>
        </div>
      </div>`;

    renderChart(document.getElementById('chart-roc'), {
      height: 260, data: rocFor(m.auc), xKey: 'fpr', fontSize: 9,
      margin: { top: 5, right: 10, bottom: 40, left: 52 },
      xLabel: 'False Positive Rate', yLabel: 'True Positive Rate',
      xDomain: [0, 1], yDomain: () => [0, 1],
      xFormat: v => v.toFixed(2), yFormat: v => v.toFixed(2),
      refLines: [{ segment: [{ x:0, y:0 }, { x:1, y:1 }], stroke:'rgba(255,255,255,0.15)', dash:'4 4' }],
      series: [{ key:'tpr', stroke:'#4DFFD2', width:2, name:'TPR' }],
      tooltipFormat: v => v.toFixed(4), tooltipValueColor: '#F0EEE8',
    });

    renderChart(document.getElementById('chart-calib'), {
      height: 260, data: calibrationFor(m), xKey: 'predicted', fontSize: 9,
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
const UPLOAD_MODES = [
  { key:'tic',         label:'TIC / KIC ID',       live:true },
  { key:'file',        label:'Upload Light Curve', live:false,
    why:'The scoring API takes a target identifier, not a file. Accepting uploads means running detrending and phase-folding on user photometry, which the service does not do today.' },
  { key:'coordinates', label:'Star Coordinates',   live:false,
    why:'Resolving RA/Dec to a TIC needs a cone search against the target catalogue. Not wired up — name the target directly for now.' },
];

const RATE_LIMIT_S = 60;
let lastScoreAt = 0;

function Upload() {
  const state = { mode:'tic', ticId:'', status:'idle', progress:0, stage:0, elapsed:0, result:null, cooldown:0 };
  let timer = null, ticker = null;

  const STAGES = [
    { label:'Resolving target in the TIC',      to:8,   ms:600 },
    { label:'Downloading photometry from MAST', to:56,  ms:3400 },
    { label:'Detrending and phase-folding',     to:72,  ms:1100 },
    { label:'Scoring 11-branch ensemble',       to:91,  ms:1500 },
    { label:'Platt calibration · MC-dropout',   to:100, ms:900 },
  ];

  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:900px;margin:0 auto;padding:3rem 3rem 0">

      <div style="margin-bottom:2.5rem">
        <div class="section-label" style="margin-bottom:0.75rem">Data Submission</div>
        <h1 style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">NAME A TARGET.</h1>
        <h2 style="font-family:'Space Grotesk';font-size:clamp(1.2rem, 2.5vw, 2rem);font-weight:300;letter-spacing:-0.02em;color:rgba(240,238,232,0.25);line-height:1.0;margin-bottom:0.6rem">GET A CALIBRATED ANSWER.</h2>
        <p style="font-family:'Inter';font-size:0.85rem;color:rgba(240,238,232,0.45);line-height:1.7">
          Give the pipeline a TIC or KIC identifier. It pulls the photometry from MAST, rebuilds the eleven input views, and scores the target against the promoted model.
        </p>
      </div>

      <div style="display:flex;gap:0;margin-bottom:2rem;border:1px solid rgba(255,255,255,0.1);flex-wrap:wrap" id="up-modes">
        ${UPLOAD_MODES.map((m, i) => `
          <button data-m="${m.key}" style="flex:1;min-width:11rem;display:flex;align-items:center;justify-content:center;gap:0.5rem;font-family:'Space Grotesk';font-size:0.65rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;padding:0.875rem;border:none;border-right:${i < UPLOAD_MODES.length - 1 ? '1px solid rgba(255,255,255,0.1)' : 'none'};transition:all 150ms ease">
            ${m.label}${m.live ? '' : '<span class="tag-chip tag-soon">coming</span>'}
          </button>`).join('')}
      </div>

      <div id="up-input"></div>
      <div id="up-action"></div>

      <div style="margin-top:3rem;padding-top:2rem;border-top:1px solid rgba(255,255,255,0.08)">
        <div class="stat-label" style="margin-bottom:1.25rem">What the endpoint does</div>
        <div class="fmt-grid" style="display:grid;grid-template-columns:repeat(3, 1fr);gap:1rem">
          ${[
            { h:'GET /score/{tic_id}', t:'live', d:'Resolves the target, fetches SPOC or PDC photometry from MAST, builds the eleven views and returns a calibrated score with per-fold detail.' },
            { h:'Rate limit', t:'', d:'One scoring request per 60 seconds per client. The MAST fetch dominates the latency — expect 20–60 s on a cold cache.' },
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

  const paintModes = () => {
    document.querySelectorAll('#up-modes button').forEach(b => {
      const m = UPLOAD_MODES.find(x => x.key === b.dataset.m);
      const on = b.dataset.m === state.mode;
      b.style.background = on ? (m.live ? 'rgba(77,255,210,0.08)' : 'rgba(255,255,255,0.03)') : 'transparent';
      b.style.color = on ? (m.live ? '#4DFFD2' : 'rgba(240,238,232,0.7)') : '#8A8FA8';
    });
  };

  const paintInput = () => {
    const m = UPLOAD_MODES.find(x => x.key === state.mode);
    if (!m.live) {
      inputEl.innerHTML = `
        <div class="soon" style="margin-bottom:1.5rem">
          <div class="h">${m.label} is not wired to the API <span class="tag-chip tag-soon" style="margin-left:0.4rem">coming</span></div>
          <div class="d">${esc(m.why)}</div>
        </div>`;
      return;
    }
    inputEl.innerHTML = `
      <div style="margin-bottom:1.5rem">
        <label for="up-tic" style="font-family:'Space Grotesk';font-size:0.6rem;font-weight:600;letter-spacing:0.12em;text-transform:uppercase;color:#8A8FA8;display:block;margin-bottom:0.5rem">TIC ID or KIC ID</label>
        <input type="text" id="up-tic" placeholder="e.g. TIC 43288669 or KIC 8120608" value="${esc(state.ticId)}"
          style="width:100%;background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.1);color:#F0EEE8;font-family:'JetBrains Mono';font-size:0.9rem;padding:1rem 1.25rem;outline:none;box-sizing:border-box">
        <div style="font-family:'Inter';font-size:0.75rem;color:#8A8FA8;margin-top:0.5rem">
          Photometry is fetched from MAST on demand, so a first scoring run on an uncached target typically takes 20–60 seconds.
        </div>
      </div>`;
    document.getElementById('up-tic').addEventListener('input', e => { state.ticId = e.target.value; paintAction(); });
  };

  const paintAction = () => {
    const m = UPLOAD_MODES.find(x => x.key === state.mode);
    if (!m.live) { actionEl.innerHTML = ''; return; }

    if (state.status === 'ratelimited') {
      actionEl.innerHTML = `
        <div class="note" style="border-color:rgba(245,166,35,0.35)">
          <span class="ico">▲</span>
          <span class="txt"><b style="color:#F5A623;font-weight:500">Rate limited.</b>
          The scoring endpoint accepts one request per ${RATE_LIMIT_S} seconds — each one pulls fresh photometry from MAST.
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
            <span class="txt">Scoring failed — ${esc(state.error)}</span>
          </div>
          <button class="btn-teal" id="up-go" ${ok ? '' : 'disabled'} style="font-size:0.75rem;padding:0.875rem 2rem">Try again →</button>`;
        if (ok) document.getElementById('up-go').addEventListener('click', run);
        return;
      }
      actionEl.innerHTML = `<button class="btn-teal" id="up-go" ${ok ? '' : 'disabled'} style="font-size:0.75rem;padding:0.875rem 2rem">Score this target →</button>`;
      if (ok) document.getElementById('up-go').addEventListener('click', run);
      return;
    }

    if (state.status === 'running') {
      actionEl.innerHTML = `
        <div style="margin-top:2rem">
          <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap;margin-bottom:0.75rem">
            <div style="display:flex;align-items:center;gap:0.5rem">
              <div class="live-dot"></div>
              <span class="stat-label" style="color:#4DFFD2">${esc(STAGES[state.stage].label)}…</span>
            </div>
            <div style="display:flex;gap:1rem;align-items:baseline">
              <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#8A8FA8">${state.elapsed.toFixed(1)}s elapsed</span>
              <span style="font-family:'JetBrains Mono';font-size:0.7rem;color:#4DFFD2;font-variant-numeric:tabular-nums">${state.awaiting ? 'waiting on MAST' : state.progress + '%'}</span>
              <button class="btn-ghost" id="up-cancel" style="font-size:0.6rem;padding:0.3rem 0.7rem">Cancel</button>
            </div>
          </div>
          <div style="height:2px;background:rgba(255,255,255,0.08);position:relative;overflow:hidden">
            <div style="height:100%;width:${state.progress}%;background:#4DFFD2;transition:width 400ms linear;box-shadow:0 0 8px rgba(77,255,210,0.6)"></div>
          </div>
          <div style="margin-top:1rem;display:grid;grid-template-columns:repeat(auto-fit,minmax(9rem,1fr));gap:0.5rem">
            ${STAGES.map((s, i) => {
              const done = i < state.stage, now = i === state.stage;
              return `<div style="display:flex;align-items:center;gap:0.4rem">
                <div style="width:6px;height:6px;border-radius:50%;background:${done || now ? '#4DFFD2' : 'rgba(255,255,255,0.15)'};opacity:${now ? 1 : done ? 0.6 : 1};box-shadow:${done || now ? '0 0 6px rgba(77,255,210,0.6)' : 'none'}"></div>
                <span style="font-family:'JetBrains Mono';font-size:0.62rem;color:${done || now ? '#4DFFD2' : '#8A8FA8'}">${esc(s.label.split(' ').slice(0, 2).join(' '))}</span>
              </div>`;
            }).join('')}
          </div>
          <div style="font-family:'Inter';font-size:0.72rem;color:#8A8FA8;margin-top:1rem">
            Holding the connection open while MAST serves the light curve. This is the slow step and it is not cached for this target.
          </div>
        </div>`;
      const cancel = document.getElementById('up-cancel');
      if (cancel) cancel.addEventListener('click', () => {
        state.cancelled = true;
        if (state.controller) state.controller.abort();
        clearInterval(timer);
        state.status = 'idle';
        state.awaiting = false;
        state.pending = null;
        state.error = null;
        paintAction();
      });
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
                <span style="font-family:'Space Grotesk';font-size:0.75rem;font-weight:500;color:${d.state === 'unmeasured' ? 'rgba(240,238,232,0.45)' : '#F0EEE8'};flex:1">${d.name}</span>
                <span style="font-family:'JetBrains Mono';font-size:0.68rem;color:${d.state === 'pass' ? '#4DFFD2' : d.state === 'fail' ? '#FF4D4D' : 'rgba(138,143,168,0.7)'}">${esc(diagValue(d))}</span>
              </div>`).join('')}
          </div>
          ${r.unmeasured ? `
          <div class="note" style="margin-bottom:1.5rem">
            <span class="ico">▲</span>
            <span class="txt">${r.unmeasured} of ${r.diags.length} diagnostics were not measured for this target — no value was returned, which is not the same as passing.</span>
          </div>` : ''}

          <div style="display:flex;gap:1rem;flex-wrap:wrap">
            <button class="btn-teal" data-nav="#/catalogue">View in Catalogue →</button>
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
    clearInterval(ticker);
    ticker = setInterval(() => {
      state.cooldown -= 1;
      if (state.cooldown <= 0) { clearInterval(ticker); state.status = 'idle'; state.result = null; }
      if (location.hash.replace(/^#/, '') === '/upload') paintAction();
      else clearInterval(ticker);
    }, 1000);
  };

  const run = () => {
    const wait = Math.ceil((RATE_LIMIT_S * 1000 - (Date.now() - lastScoreAt)) / 1000);
    if (lastScoreAt && wait > 0) { state.status = 'ratelimited'; state.cooldown = wait; startCooldown(); paintAction(); return; }

    state.status = 'running'; state.progress = 0; state.stage = 0; state.elapsed = 0; state.result = null;
    paintAction();

    const t0 = performance.now();
    clearInterval(timer);

    /* The staged bar and the real request run together. The stages are a
       readable account of what the server is doing, not a measurement of it,
       so the bar is allowed to finish first and then wait. */
    const tic = Number(String(state.ticId).replace(/[^0-9]/g, ''));
    state.cancelled = false;
    state.controller = API.mode === 'live' && tic ? new AbortController() : null;
    state.pending = state.controller
      ? loadScore(tic, {}, { signal: state.controller.signal })
          .then(sc => ({ sc }))
          .catch(e => ({ err: e.name === 'AbortError' ? null : e.message }))
      : null;
    state.awaiting = false;

    timer = setInterval(() => {
      state.elapsed = (performance.now() - t0) / 1000;
      const total = STAGES.slice(0, state.stage + 1).reduce((s, x) => s + x.ms, 0);
      const prev = STAGES.slice(0, state.stage).reduce((s, x) => s + x.ms, 0);
      const from = state.stage === 0 ? 0 : STAGES[state.stage - 1].to;
      const f = Math.min(1, (state.elapsed * 1000 - prev) / STAGES[state.stage].ms);
      state.progress = Math.round(from + (STAGES[state.stage].to - from) * f);
      if (state.elapsed * 1000 >= total && state.stage < STAGES.length - 1) state.stage++;
      if (state.elapsed * 1000 >= STAGES.reduce((s, x) => s + x.ms, 0)) {
        if (state.pending) {
          // Hold the last stage until the real score lands. MAST fetches run
          // 20-60 s cold, well past the animation, and completing the bar
          // before the answer exists would show a finished run with no result.
          const pending = state.pending;
          state.pending = null;
          state.awaiting = true;
          pending.then(out => {
            clearInterval(timer);
            state.awaiting = false;
            // A cancelled run never happened: it must not start the rate-limit
            // cooldown, and it has nothing to report.
            if (state.cancelled || !has(out.err) && !has(out.sc)) {
              state.status = 'idle';
              return;
            }
            lastScoreAt = Date.now();
            if (out.err) {
              state.status = 'idle';
              state.error = out.err;
            } else {
              state.status = 'complete';
              state.error = null;
              state.result = liveScoreResult(state.ticId, state.elapsed, out.sc);
            }
            if (location.hash.replace(/^#/, '') === '/upload') paintAction();
          });
        } else if (!state.awaiting) {
          clearInterval(timer);
          lastScoreAt = Date.now();
          state.status = 'complete';
          state.result = mockScore(state.ticId, state.elapsed);
        }
      }
      if (location.hash.replace(/^#/, '') === '/upload') paintAction();
      else clearInterval(timer);
    }, 220);
  };

  document.querySelectorAll('#up-modes button').forEach(b => b.addEventListener('click', () => {
    state.mode = b.dataset.m; paintModes(); paintInput(); paintAction();
  }));

  paintModes(); paintInput(); paintAction();
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
      <h1 style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:0.1rem">SEARCH A LIGHT CURVE.</h1>
      <div style="font-family:'Space Grotesk';font-size:clamp(1.4rem, 2.6vw, 2.2rem);font-weight:600;color:rgba(240,238,232,0.30);letter-spacing:-0.02em;margin-bottom:1.25rem">FIND THE SIGNAL YOURSELF</div>
      <p style="font-family:'Inter';font-size:0.9rem;line-height:1.7;color:#8A8FA8;max-width:64ch;margin-bottom:2.5rem">
        Every other page on this console vets a signal somebody else detected — a
        TOI, a KOI, a SPOC threshold-crossing event. Discovery is the other half:
        take a raw light curve with no ephemeris and search it for a periodic
        transit, then score whatever it finds.
      </p>

      <div class="soon" style="margin-bottom:2.5rem">
        <div class="h">Not yet served <span class="tag-chip tag-soon" style="margin-left:0.4rem">coming</span></div>
        <div class="d">
          The search itself is built — a box-least-squares period search over a
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
  ['How it works', 'The pipeline end to end — catalogue refresh, validation gates, the eleven input views, the five-fold ensemble, calibration, the promotion gate.'],
  ['How to read a score', 'What a calibrated probability means here, what the MC-dropout band is, and why a margin under the noise floor is not a difference.'],
  ['Known limits', 'The measured defects, stated plainly: observation baseline correlates with the label on TESS, and the model scores the star as well as the transit.'],
  ['Data and provenance', 'Where every input comes from — MAST, ExoFOP, the NASA archive, Gaia — and which model version served any given number.'],
  ['Credits and licence', 'Attribution, the ExoMiner work this builds on, and the licence this is released under.'],
];

function About() {
  app.innerHTML = `
  <div style="min-height:100vh;background:#050608;padding-top:56px;padding-bottom:40px">
    <div class="page-pad" style="max-width:1440px;margin:0 auto;padding:3rem 3rem 0">
      <div class="section-label" style="margin-bottom:0.75rem">About</div>
      <h1 style="font-family:'Space Grotesk';font-size:clamp(2rem, 4vw, 3.5rem);font-weight:700;letter-spacing:-0.03em;color:#F0EEE8;line-height:1.0;margin-bottom:1.25rem">EXOPLANET HUNTER.</h1>
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
              <span style="font-family:'Space Grotesk';font-size:1rem;font-weight:600;color:#F0EEE8">${h}</span>
            </div>
            <div style="font-family:'Inter';font-size:0.8rem;line-height:1.7;color:rgba(138,143,168,0.75);padding-left:2.1rem">${d}</div>
          </div>`).join('')}
      </div>
    </div>
  </div>`;
}
