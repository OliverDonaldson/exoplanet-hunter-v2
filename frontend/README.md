# Console

The vetting console lives in [`design-console/`](design-console/README.md) and
is live at `https://exoplanet-hunter-console.onrender.com`. It is plain HTML
and JavaScript built into one static file; there is no bundler.

```bash
make frontend     # npm install (anime.js), build.py, then serve dist/ on :5173
```

Open `http://localhost:5173/?api=http://localhost:8000` with `make api`
running. The `?api=` query overrides the API base for one session; Render bakes
the production base in through `EH_API_BASE` (see `render.yaml`).

`package.json` exists only to fetch anime.js, which `build.py` inlines. The wire
contract the console reads is `api/app/schemas.py`; the client that reads it is
`design-console/src/app.api.js`, and the two change together.

The earlier React console (`frontend/src`) was removed on 2026-09-04. It had not
shipped since 2026-08-20.
