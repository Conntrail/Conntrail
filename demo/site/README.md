# Conntrail website demo

The marketing site and interactive demo for Conntrail, served at
**https://conntrail.ibzie.dev** (a subdomain of an existing `ibzie.dev` — no new
domain needed).

It is a static site: precomputed `TraceRecord` fixtures for the use-case pages,
**no LLM calls and no server-side rendering** for those. Cloudflare Pages
Functions provide the contact endpoint (`POST /api/contact`). The **Live demo**
page (`live.html`) additionally calls the separate `demo/service` API to run
the real Conntrail analysis on the visitor's message using free models.

This whole folder (and `../service`, `../fixtures`) is the demo/marketing
material — it is intentionally kept out of the SDK packages in `src/`.

## Files

```
demo/site/
  index.html            hero + positioning + links to the other pages
  use-cases.html        the four incidents (picker + incident/after-fix view)
  live.html             run Conntrail live on your own message (free models)
  how-it-works.html     method, stability labels, failure classifier
  pricing.html          self-hosted vs managed enterprise
  contact.html          contact form
  style.css             visual language: one stylesheet, custom properties, light/dark
  app.js                per-page init: demo (use-cases), contact form, active nav
  live.js               live-demo page logic (calls the demo/service API)
  config.js             runtime config: demo API URL + Turnstile site key
  favicon.svg           brand mark (one node, three routes)
  data/demo.json        frozen use-case fixtures (generated, committed)
  functions/api/contact.js   Cloudflare Pages Function for the contact form
  dev_server.py         zero-dependency local preview (static + form stub)
  _headers              security headers (CSP, etc.)
  robots.txt
  wrangler.toml         Pages config (output dir = this folder)
```

## Live demo

`live.html` runs the **real** Conntrail analysis on the visitor's message via
the `demo/service` API (see `../service/README.md`). Free models only, capped,
with a bring-your-own-key escape hatch. Set the deployed API URL in
`config.js` (`CONNTRAIL_DEMO_API`); when previewing on localhost the page calls
`http://127.0.0.1:8000` automatically.

## Use cases

Four real-world failure patterns (fictional companies, real patterns). Each
shows a simulated user-facing surface, Conntrail's detection, and — via the
**Incident / After fix** toggle — the user-facing outcome after the fix (e.g.
the WhatsApp chat that escalates and cancels in time instead of looping):

1. Support chatbot invents a refund policy (Cursor / Air Canada / Priceline)
2. WhatsApp order bot misses the escalation window (Zomato + WhatsApp API errors)
3. Lead-gen bot buries a high-intent buyer (false confidence)
4. Provider outage goes unnoticed until users churn — Conntrail classifies and escalates (OpenAI/Anthropic status incidents)

Entropy, stability, attribution and the counterfactual are **derived** from the
frozen route labels by `demo/fixtures/build_fixtures.py` using the SDK's own
`routing_entropy` and attribution table — they are not hand-typed.

## Regenerate the fixtures

```bash
python demo/fixtures/build_fixtures.py --out demo/site/data/demo.json
```

## Preview locally

Zero-dependency preview (static site + a stub that makes the contact form
succeed locally):

```bash
python demo/site/dev_server.py            # http://localhost:8090
```

To preview the **live demo** page, also run the demo service (needs a free key
in `.env`):

```bash
uvicorn demo.service.app:app --port 8000    # from the repo root
python demo/site/dev_server.py              # then open http://localhost:8090/live.html
```

Full fidelity, including the real Cloudflare Pages Function:

```bash
cd demo/site && npx wrangler pages dev .  # http://localhost:8788
```

A plain static server also works for everything except the form:

```bash
python -m http.server 8080 --directory demo/site
```

## Deploy to Cloudflare Pages (conntrail.ibzie.dev)

1. **Create the Pages project** — Cloudflare dashboard → Workers & Pages →
   Create → Pages → connect the Git repo.
   - **Root directory:** `demo/site`
   - **Build command:** *(leave empty)*
   - **Build output directory:** `/` (the root dir is already the site)
   - `demo/site/functions/` is auto-detected as the Functions directory.

   Or deploy directly with Wrangler:

   ```bash
   npx wrangler pages deploy demo/site --project-name conntrail
   ```

2. **Add the custom domain** — Pages project → Custom domains → Set up a
   domain → `conntrail.ibzie.dev`. If `ibzie.dev` is on Cloudflare DNS, the
   `CNAME` record is created automatically. Otherwise add it manually:

   ```
   Type:  CNAME
   Name:  conntrail
   Value: <your-project>.pages.dev
   Proxy: on (recommended)
   ```

3. **Contact form env vars** — Pages project → Settings → Environment
   variables (Production). Set **one** delivery path:

   | Variable | Purpose |
   |---|---|
   | `RESEND_API_KEY` | Resend API key (recommended) |
   | `CONTACT_TO_EMAIL` | Where enquiries go (default `hello@ibzie.dev`) |
   | `CONTACT_FROM_EMAIL` | Verified Resend sender, e.g. `Conntrail <hello@ibzie.dev>` |
   | `CONTACT_WEBHOOK_URL` | Alternative: any JSON webhook (Slack, Zapier, …) |
   | `TURNSTILE_SECRET` | Optional Cloudflare Turnstile bot check |

   If none are set, `/api/contact` returns `503` and the page falls back to the
   `mailto:hello@ibzie.dev` link.

## Constraints (do not drift)

- Precomputed fixtures only; no live calls in the use-case pages.
- The live demo runs free models only, capped, never a paid model.
- The improvement loop (CPE-GEPA) is **experimental** — one footer link, never
  in the pitch, pricing, or demo flow.
- Self-hosted is available now; managed enterprise is labelled "in progress".
- Every figure shown is derived from a signal Conntrail measured; the page
  recommends nothing.
