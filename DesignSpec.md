# Conntrail website — Design Spec & Handoff

This document describes the redesign of the Conntrail marketing site (`web/`),
what has already been implemented, and exactly what remains. Read it fully
before touching anything. If a choice seems arbitrary, it is documented here.

## 0. Current state (read this first)

**The redesign is implemented and verified.** All five pages, the stylesheet,
the JS renderer, the fixtures and the favicon are done. As of the last commit
of this doc:

- `node --check web/app.js` passes
- `522 passed` in `pytest tests/unit -m "not integration"`
- `ruff check .` is clean
- Screenshots reviewed: all 5 pages light + index/use-cases dark, all 4 use
  cases, the Incident/After-fix states, 390px mobile

**Remaining work is polish-only**, listed in §8. Do not re-do finished work.

## 1. Design rationale (the "why")

Conntrail is a **measurement product**. It measures how stable an agent's
routing decision is under semantic perturbation. So the site is designed like
a **precision instrument**, not a SaaS landing page:

- **Graph-paper motif**: a faint 28px grid + intersection dots behind the
  hero and page heads (`.grid-bg`), faded with a radial mask. It says "we
  measure things" without a single word.
- **One voice for severity**: the only saturated colors on the page are the
  three stability colors (green/amber/red). Everything else is near-neutral.
  Severity reads instantly because nothing else competes.
- **Data looks measured**: all numbers, route names, timestamps and
  metadata use the monospace stack with tabular numerals; uppercase
  letterspaced micro-labels for panel headers ("WHAT THE USER SAW").
- **The pitch is the instrument**: the homepage hero shows a real frozen
  measurement (gauge 0.75, FRAGILE, flipped variants, counterfactual) next
  to the headline. No stock illustration.
- **Calm, EU-technical tone**: no gradients, no emoji decoration, no hype
  words, hairline borders, restrained shadows.

## 2. Hard constraints (do not violate, ever)

1. Plain HTML + CSS + vanilla JS. **No build step, no framework, no npm
   runtime dependency, no preprocessor.** Must work as static files.
2. **No external requests at runtime.** No Google Fonts, no CDNs, no
   analytics. System font stack only. The only outbound links are `<a href>`
   to GitHub and source references.
3. Must satisfy `web/_headers` CSP:
   `default-src 'self'; style-src 'self' 'unsafe-inline'; script-src 'self';
   img-src 'self' data:; connect-src 'self'; base-uri 'self'; form-action 'self'`.
   Consequences: inline `style=""` attributes are allowed; inline `<script>`
   is NOT; all JS lives in `app.js`; SVG is inline markup or `favicon.svg`.
4. Accessible: semantic HTML, WCAG AA contrast, `:focus-visible` rings,
   keyboard-operable tabs, `prefers-reduced-motion` support, content works
   without JS except the interactive case viewer (which has a `<noscript>`
   note).
5. Responsive from 320px up. Light and dark themes via
   `prefers-color-scheme` only (no manual toggle; do not add one casually).
6. **Copy rule: NO dashes in content.** No em dash (—), no en dash (–), no
   spaced hyphen used as a pause. Verify with:
   `grep -rn $'—\|–' web/*.html web/app.js web/data/demo.json` → must print
   nothing. Hyphens inside compound words (self-hosted, retry_loop,
   pay-as-you-go) are words, not dashes; keep them but don't invent new ones.
   The "·" middot and "→" arrow are allowed and used deliberately.
7. Positioning guardrails:
   - Never sell tracing. Conntrail is positioned against tracing, not as it.
   - The prompt-optimization loop is always labelled **"Experimental:"** and
     lives only in the footer, never in the pitch/pricing/demo flow.
   - Anything AI-Act-shaped says **"evidence, not legal advice"** (currently
     in `pricing.html` pricing-note).
   - Self-hosted = "Available now"; managed enterprise = "In progress".
   - Findings recommend, never mutate. The site recommends nothing itself.

## 3. Design tokens (all in `web/style.css`, section "1. Tokens")

Colors are CSS custom properties with light + dark values. **Never hardcode
a hex outside the token blocks.**

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#ffffff` | `#0b0e13` | page background |
| `--bg-raised` | `#fcfdff` | `#0f1319` | cards, inputs |
| `--panel` | `#f6f8fb` | `#131823` | recessed panels, analysis column |
| `--fg` | `#10151d` | `#e6eaf2` | primary text |
| `--muted` | `#556070` | `#9aa4b5` | secondary text (AA on bg) |
| `--faint` | `#7d8694` | `#6d7686` | tertiary/meta text only |
| `--border` | `#e3e8f0` | `#232b38` | hairlines |
| `--border-strong` | `#c8d1de` | `#333e50` | input borders, hover |
| `--accent` | `#1d4ed8` | `#8aabff` | links, active states |
| `--accent-btn` | `#1d4ed8` | `#2563eb` | primary button bg (white text) |
| `--accent-soft` | `#edf2fe` | `rgba(122,162,255,.12)` | accent tint |
| `--confident/-soft/-ink` | `#15803d / #e8f6ec / #166534` | `#4ade80 / 12% / #7fe6a4` | stable |
| `--boundary/-soft/-ink` | `#b45309 / #fcf1e1 / #92400e` | `#fbbf24 / 12% / #fcd34d` | at risk |
| `--fragile/-soft/-ink` | `#dc2626 / #fdebeb / #b91c1c` | `#f87171 / 12% / #fca5a5` | unstable |
| `--wa-bg/out/in` | `#ece0d4 / #d2f8c6 / #ffffff` | `#0b141a / #005c4b / #1f2c34` | WhatsApp surface |
| `--grid-line/dot` | low-alpha dark | low-alpha light | graph-paper |

Type: `--font-ui` (system stack), `--font-mono` (ui-monospace stack, always
with `font-variant-numeric: tabular-nums`). Scale: `--t-xs` .75rem, `--t-sm`
.85rem, `--t-md` 1rem, `--t-lg` 1.125rem, `--t-xl` 1.375rem, `--t-2xl`
1.75rem, `--t-3xl` clamp(→2.55rem), `--t-hero` clamp(→2.85rem).

Spacing: `--s1..--s8` = .25/.5/.75/1/1.5/2/3/4.5rem. Radii: `--r-sm` 6,
`--r-md` 10, `--r-lg` 14, `--r-xl` 20, `--r-pill` 999. Shadows:
`--shadow-sm/md/lg`. Layout: `--maxw` 1080px, `--topbar-h` 60px.

## 4. Architecture (what lives where)

- `web/style.css` — the only stylesheet. 18 numbered sections; keep new CSS
  in the right section. Dark values only inside the existing
  `@media (prefers-color-scheme: dark)` token block. All decorative
  animation inside `@media (prefers-reduced-motion: no-preference)`; a
  `reduce` block kills transitions globally.
- `web/app.js` — the only script. Renders use cases from
  `fetch("data/demo.json")`, plus the contact form POST to `/api/contact`
  and the active-nav marker. ES5-style (`var`, no modules) by choice.
- `web/data/demo.json` — generated fixture bundle. **Never hand-edit.** It
  is built by `examples/website/build_fixtures.py` from
  `examples/website/use_cases.py` (entropy/attribution derived by the SDK).
  After editing copy in `use_cases.py`, regenerate:
  `.venv/bin/python examples/website/build_fixtures.py --out web/data/demo.json`
- `web/favicon.svg` — brand mark: one node (blue) splitting into three
  routes (green/amber/red). Same geometry as `.brand-mark` in the topbar.
- `web/functions/api/contact.js`, `web/_headers`, `web/robots.txt`,
  `web/wrangler.toml`, `web/dev_server.py` — untouched infrastructure.
  Do not modify without a separate reason.

### IDs/classes `app.js` and tests depend on (breaking these breaks things)

`#case-picker`, `#case-view`, `#contact-form`, `#form-status`,
`#contact-submit`, `.topbar nav a` (+`.active`), `.fix-toggle`
(`.after` drives the sliding thumb), honeypot field `name="website"`.
Tests (`tests/unit/test_demo_fixtures.py`) validate the demo.json schema:
`use_cases[]` fields, `surface_after`, derived entropy/stability, failure
categories in {exception, retry_loop, timeout, malformed_output, none},
and flat `records` round-tripping `TraceRecord`. **Do not change the
schema.**

## 5. Component notes (for consistent extension)

- **Brand mark**: 32×32 SVG, dot at (9,16), three routes to x=24.5 colored
  confident/boundary/fragile. Reused identically in topbar, footer, favicon.
- **Badges** (`.badge-*`): tinted chip + 7px dot, uppercase 0.75rem. Never
  white text on saturated bg (fails AA); always tint + `-ink` text.
- **Gauge**: SVG r=52, C≈326.73, `stroke-dashoffset = C×(1−entropy)`,
  animated via double-rAF in `app.js:gauge()`. Ticks at 0/.25/.5/.75,
  dashed threshold markers at .25/.60. Entropy 0 draws a `.gauge-zero-dot`
  at 12 o'clock instead of an arc (an empty ring looked broken).
- **Attribution table**: flipped rows get `.flip` (red tint + inset bar)
  and a `res-flip` chip; stable rows `res-hold`. "First flip:" line below.
- **Fix toggle**: two buttons + sliding `::before` thumb (red left, green
  right); `.after` class on the container slides it. Delta chips
  (`.delta-chip delta-<stability>`) show `0.75 FRAGILE → 0.00 CONFIDENT`;
  the outage case shows `Unacknowledged → Escalated in 38s` (no label span).
- **Case picker**: pill tabs, `role=tab`, roving `tabindex`, arrow/Home/End
  keys in `app.js:onPickerKey`.
- **Surfaces**: web_chat uses per-message avatars (B / M); WhatsApp does
  NOT (authentic); leads use `route → should_be` chips + verdict label;
  outage uses status dots (pulsing on `down`) + `cat-<category>` chips.
- **Case enter animation**: `.case.enter` retriggered by class
  remove/reflow/add in `renderCase()`.

## 6. Old → new mapping

All original class names were preserved (`.topbar`, `.hero`, `.pos-card`,
`.case-picker`, `.case-grid`, `.bubble`, `.wa`, `.lead`, `.svc`, `.gauge*`,
`.badge-*`, `.attr-table`, `.fix-toggle`, `.alerts`, `.cost-line`,
`.site-footer`, `.exp-link`, …). New classes are additive: `.skip-link`,
`.brand-mark`, `.grid-bg`, `.page-head`, `.section-head`, `.hero-grid`,
`.hero-stats`, `.specimen*`, `.chip`, `.res*`, `.icon`, `.msg-row`,
`.avatar`, `.lead-verdict`, `.cat-<category>`, `.cf-label`, `.delta-*`,
`.fix-top`, `.case-meta`, `.steps`, `.label-meter`, `.contact-grid`,
`.footer-inner`, `.footer-col`, `.footer-fine`, `.h-label`, `.h-tag`.
Footer markup changed from two `<span>`s to `.footer-inner` grid —
`.site-footer` and `.exp-link` still exist as hooks.

## 7. What changed (prioritized, done)

1. Full token system + 18-section stylesheet; graph-grid motif; type scale.
2. Brand mark + favicon; topbar with blur, active-nav underline, skip link.
3. Hero: two-column, headline pitch + static specimen measurement card +
   mono proof stats.
4. Use-case view: instrument-style panel headers with icons, premium
   surfaces (chat avatars/flags, authentic WhatsApp, verdict-labelled
   leads, pulsing outage dots), tab a11y.
5. Gauge ticks/thresholds/zero-dot; tinted badges; flipped-row treatment;
   counterfactual as a labelled callout; failure-signal chips.
6. Incident/After-fix: sliding red/green segmented toggle + delta chips;
   case re-render transition (reduced-motion safe).
7. Footer: 3-column + mono fine print. Pricing/contact/pages restyled.
8. All copy rewritten dash-free; fixtures de-dashed at the source
   (`examples/website/use_cases.py`) and regenerated.

## 8. Remaining work (polish only, in priority order)

1. **Mobile meta separators**: on 390px the `.case-meta` middot separators
   (`::before`) can start a wrapped line. Either hide separators under
   560px (`@media (max-width:560px){ .case-meta > span::before{content:none} }`)
   or accept it. Cosmetic.
2. **Optional: keyboard focus order check** with a real keyboard on
   use-cases (tab through picker → toggle → links). Expected fine.
3. **Do NOT commit binary screenshots to the repo** unless the user asks;
   regenerate them any time with §9.
4. **Deploy note**: nothing in the redesign changes deploy config;
   `wrangler.toml`, `_headers`, `functions/` are untouched. Deploy as before
   (`npx wrangler pages deploy web --project-name conntrail`).
5. `AGENTS.md` does not mention `web/`; no update needed there.
   `web/README.md` was already updated (favicon + stylesheet description).

## 9. Verification harness (exact commands)

```bash
# serve (already running at the time of writing; restart with:)
python web/dev_server.py            # http://localhost:8090

# static checks
node --check web/app.js
.venv/bin/python -c "import json; json.load(open('web/data/demo.json'))"
.venv/bin/python -m pytest tests/unit -m "not integration" -q
.venv/bin/ruff check .

# dash check (must print nothing)
grep -rn $'—\|–' web/*.html web/app.js web/data/demo.json examples/website/*.py

# screenshots: cached Chromium + CDP script (no install needed;
# python pkg 'websockets' from the repo venv; also emulates dark mode,
# clicks the picker/toggle, and takes 390px mobile shots)
.venv/bin/python /tmp/opencode/shoot.py   # writes /tmp/opencode/*.png
```

`/tmp/opencode/shoot.py` drives
`~/.cache/ms-playwright/chromium_headless_shell-1234/.../chrome-headless-shell`
via `--remote-debugging-port` and `websockets`. `/tmp` is ephemeral; if the
script is gone, recreate it from this spec's description (targets: all 5
pages light, index + use-cases dark, all 4 cases, after-fix states, mobile).

## 10. Known non-issues (do not "fix")

- Emoji renders as tofu boxes in headless screenshots only (no emoji font
  in the sandbox). Real browsers render them.
- `main { animation: page-in }` makes instant CLI screenshots look faded;
  the CDP harness sleeps 2.2s before capturing. Real users see a 0.4s fade.
- The gauge at entropy 0 shows a dot, not an arc (intentional, see §5).
- `demo.json` timestamps/trace_ids change on regeneration (expected; tests
  only check derived consistency).
- The dev server on port 8090 was left running for preview; stop with
  `pkill -f dev_server.py`.
