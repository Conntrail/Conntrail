# Conntrail website demo

A zero-cost, no-signup static page that makes the positioning statement
interactive: **"Is this routing decision stable?"**

The page loads precomputed `TraceRecord` fixtures — it makes **no LLM calls and
needs no server**, so it is instant and free to serve even when a provider is
down.

## Files

```
web/
  index.html      the single page (input → perturb → variants → entropy → attribution)
  style.css       visual language shared with the dashboard (same stability colors)
  app.js          loads data/demo.json and renders; no framework, no build step
  data/demo.json  frozen TraceRecord fixtures (generated, committed)
```

## Regenerating the fixtures

The fixtures are built from the hand-authored scenarios in
`examples/website/scenarios.py`.

```bash
# Offline (deterministic, no LLM, no cost) — what is committed:
python examples/website/build_fixtures.py --out web/data/demo.json

# Live (real contrast generation + divergence analysis; needs a provider key):
python examples/website/build_fixtures.py --live --out web/data/demo.json
```

Offline mode computes entropy from the scenarios' hand-authored route labels
via the same `routing_entropy` the SDK uses, so the entropy/attribution/stability
shown on the page are the real formulas — only the inputs are frozen.

## Serving locally

Any static file server works; the page needs `data/demo.json` served alongside
it:

```bash
python -m http.server 8080 --directory web
# open http://localhost:8080
```

## Constraints (do not drift)

- Precomputed fixtures only. No live calls in the default page.
- No signup, no key management.
- The improvement loop (CPE-GEPA) stays **experimental** — one discreet link,
  never part of the demo flow or the pitch.
- Every figure shown is derived from a signal Conntrail measured; the page
  recommends nothing.
