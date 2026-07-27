# API

FastAPI serving. Live at `https://exoplanet-hunter-api.fly.dev`.

```bash
make api          # uvicorn on :8000, reload
```

## Endpoints

| route | returns |
|---|---|
| `GET /healthz` | liveness + the promoted run id |
| `GET /score/{tic_id}` | full vetting result for one target — the console's core call |
| `GET /candidates` | paginated candidate catalogue (sortable, capped at 1000 rows) |
| `GET /candidates.csv` | the same rows as CSV |
| `GET /reliability` | pooled calibration curve + ECE/Brier for the promoted run |

There is no `/` route — `{"detail":"Not Found"}` at the root is expected.

## The pinned contract

`app/schemas.py` and `frontend/src/api/types.ts` describe the same wire format
and **change together or not at all**. Adding an optional field to
`ScoreResponse` without the matching TypeScript is how the console silently
stops rendering a panel.

## Serving shape

One process, one model. Deliberate choices, each with a scar behind it:

- **The ensemble preloads in a lifespan thread**, so a cold Fly machine warms
  TensorFlow while the user is still browsing.
- **`_score_lock` serialises scoring.** The box has one vCPU, and two
  concurrent scores of the same TIC once rewrote a FITS under the other's
  memory-map — SIGBUS, exit 135. This is not a throughput bug to optimise away.
- **The response cache is LRU, bounded, and lock-guarded.** Check-then-pop and
  evict-then-insert are both multi-step; unguarded they raise `KeyError` and
  `RuntimeError` under contention (reproduced: 2,276 and 1,605 in 320k
  iterations) and both surface as 500s.
- **MC-Dropout draws in one batched forward pass.** Sequential passes took
  >12 min on shared CPU and scale-to-zero killed the abandoned requests.
- **The headline probability is deterministic**; MC only feeds `prob_std`.
  Feeding MC means to deterministically-fitted calibrators cost ~0.08 ECE.
- **Error bodies name no server paths.** Both the 503s and the 404 (whose
  download reason interpolates an `OSError` carrying its path) are redacted.

## Tests

```bash
pytest api/tests
```

The scorer is stubbed, so these pin HTTP semantics without TensorFlow. The
autouse fixture clears the module-level response cache between tests — without
it a TIC scored by one test is served from cache to the next and never reaches
its stub.
