# Conntrail website demo

The marketing site and interactive demo for Conntrail, served at
**https://conntrail.ibzie.dev** (a subdomain of an existing `ibzie.dev` — no new
domain needed).

It is a static site: precomputed `TraceRecord` fixtures, **no LLM calls and no
server-side rendering**. Cloudflare Pages Functions provide the one dynamic
endpoint (`POST /api/contact`).

## Files

```
web/
  index.html            hero + positioning + links to the other pages
  use-cases.html        the four incidents (picker + incident/after-fix view)
  how-it-works.html     method, stability labels, failure classifier
  pricing.html          self-hosted vs managed enterprise
  contact.html          contact form
  style.css             visual language shared with the dashboard
  app.js                per-page init: demo (use-cases), contact form, active nav
  data/demo.json        frozen use-case fixtures (generated, committed)
  functions/api/contact.js   Cloudflare Pages Function for the contact form
  dev_server.py         zero-dependency local preview (static + form stub)
  _headers              security headers (CSP, etc.)
  robots.txt
  wrangler.toml         Pages config (output dir = this folder)
```

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
frozen route labels by `examples/website/build_fixtures.py` using the SDK's own
`routing_entropy` and attribution table — they are not hand-typed.

## Regenerate the fixtures

```bash
python examples/website/build_fixtures.py --out web/data/demo.json
```

## Preview locally

Zero-dependency preview (static site + a stub that makes the contact form
succeed locally):

```bash
python web/dev_server.py            # http://localhost:8090
```

Full fidelity, including the real Cloudflare Pages Function:

```bash
cd web && npx wrangler pages dev .  # http://localhost:8788
```

A plain static server also works for everything except the form:

```bash
python -m http.server 8080 --directory web
```

## Deploy to Cloudflare Pages (conntrail.ibzie.dev)

1. **Create the Pages project** — Cloudflare dashboard → Workers & Pages →
   Create → Pages → connect the Git repo.
   - **Root directory:** `web`
   - **Build command:** *(leave empty)*
   - **Build output directory:** `/` (the root dir is already the site)
   - `web/functions/` is auto-detected as the Functions directory.

   Or deploy directly with Wrangler:

   ```bash
   npx wrangler pages deploy web --project-name conntrail
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

- Precomputed fixtures only; no live calls in the default page.
- The improvement loop (CPE-GEPA) is **experimental** — one footer link, never
  in the pitch, pricing, or demo flow.
- Self-hosted is available now; managed enterprise is labelled "in progress".
- Every figure shown is derived from a signal Conntrail measured; the page
  recommends nothing.
