# Conntrail demo (site + live service)

Everything marketing/demo lives here, kept separate from the SDK packages in
`src/` so the library, collector, dashboard and reports stay uncluttered.

```
demo/
  site/         static marketing site + interactive demo (Cloudflare Pages)
  service/      FastAPI service that runs the real SDK live on free models
  fixtures/     generator for the site's precomputed TraceRecord fixtures
  render.yaml   Render blueprint for the live service
```

- **Site** → see `site/README.md` (deploys to `conntrail.ibzie.dev`).
- **Live service** → see `service/README.md` (Render free plan, free models only).
- **Fixtures** → `python demo/fixtures/build_fixtures.py --out demo/site/data/demo.json`.

Quick local run:

```bash
# live service (needs a free key in .env, gitignored)
set -a; . ./.env; set +a
uvicorn demo.service.app:app --port 8000

# static site
python demo/site/dev_server.py          # http://localhost:8090
```
